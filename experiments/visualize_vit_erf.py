from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from experiments.cv_vit_rrlsso_cifar100 import build_model, make_loaders
from lsso.modules_v2 import apply_rank_rotary


def normalize01(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float().cpu()
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def map_stats(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().abs().flatten().cpu()
    p = x / x.sum().clamp_min(1e-12)
    k = max(1, int(0.1 * p.numel()))
    entropy = -(p * p.clamp_min(1e-12).log()).sum() / torch.log(torch.tensor(float(p.numel())))
    return {
        "peak": float(p.max()),
        "top10_frac": float(p.topk(k).values.sum()),
        "entropy01": float(entropy),
    }


def load_checkpoint_model(
    ckpt_path: Path,
    *,
    kind: str,
    rank: int,
    image_size: int,
    patch_size: int,
    device: torch.device,
) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(kind, num_classes=100, rank=rank, image_size=image_size, patch_size=patch_size)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def forward_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    x = model._process_input(images)
    cls = model.class_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls, x), dim=1)
    x = model.encoder(x)
    return x[:, 0]


def forward_tokens(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    x = model._process_input(images)
    cls = model.class_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls, x), dim=1)
    return model.encoder(x)


def input_erf(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    max_samples: int,
    target: str,
) -> torch.Tensor:
    saliency_sum: torch.Tensor | None = None
    seen = 0
    model.eval()
    for images, labels in loader:
        if seen >= max_samples:
            break
        take = min(images.shape[0], max_samples - seen)
        images = images[:take].to(device).detach()
        labels = labels[:take].to(device)
        images.requires_grad_(True)
        model.zero_grad(set_to_none=True)

        if target == "feature_norm":
            features = forward_features(model, images)
            score = features.float().square().sum(dim=-1).mean()
        elif target == "true_logit":
            logits = model(images)
            score = logits.gather(1, labels[:, None]).mean()
        elif target == "pred_logit":
            logits = model(images)
            score = logits.max(dim=1).values.mean()
        elif target == "center_token_sqnorm":
            tokens = forward_tokens(model, images)
            num_patch_tokens = tokens.shape[1] - 1
            grid = int(round(num_patch_tokens**0.5))
            if grid * grid != num_patch_tokens:
                raise ValueError(f"patch token count {num_patch_tokens} is not a square")
            center_idx = 1 + (grid // 2) * grid + (grid // 2)
            score = tokens[:, center_idx].float().square().sum(dim=-1).mean()
        else:
            raise ValueError(f"unknown target {target!r}")

        score.backward()
        saliency = images.grad.detach().float().square().mean(dim=1)
        batch_sum = saliency.sum(dim=0)
        saliency_sum = batch_sum if saliency_sum is None else saliency_sum + batch_sum
        seen += take

    if saliency_sum is None:
        raise RuntimeError("no samples were processed")
    return saliency_sum / float(seen)


def encoder_layer_input(model: nn.Module, images: torch.Tensor, layer: int) -> tuple[nn.Module, torch.Tensor]:
    x = model._process_input(images)
    cls = model.class_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls, x), dim=1)
    x = model.encoder.dropout(x + model.encoder.pos_embedding)
    layers = list(model.encoder.layers.children())
    for i in range(layer):
        x = layers[i](x)
    return layers[layer], layers[layer].ln_1(x)


@torch.no_grad()
def mixer_matrix(
    model: nn.Module,
    images: torch.Tensor,
    *,
    kind: str,
    layer: int,
) -> torch.Tensor:
    block, x = encoder_layer_input(model, images, layer)
    if kind == "mha":
        _out, weights = block.self_attention(x, x, x, need_weights=True, average_attn_weights=False)
        return weights.detach().float().mean(dim=(0, 1))

    rr = block.self_attention.lsso
    B, N, D = x.shape
    H = rr.num_heads
    r = rr.rank
    UC = rr.w_uc(x)
    U, _C = UC.split((H * r, D), dim=-1)
    U = U.view(B, N, H, r).transpose(1, 2).contiguous()
    if rr.normalize_u:
        U = U * torch.rsqrt(torch.mean(U * U, dim=-1, keepdim=True) + rr.eps)
    U = apply_rank_rotary(U, None, base=rr.rope_base, scale=rr.rope_scale)

    calc = torch.float32
    U = U.to(calc)
    mu = (torch.nn.functional.softplus(rr.theta_mu) + rr.eps).to(calc).view(1, H, 1, 1)
    gamma = (rr.gamma_max * torch.sigmoid(rr.theta_gamma)).to(calc).view(1, H, 1, 1)
    if rr.no_global:
        gamma = torch.zeros_like(gamma)

    eye_r = torch.eye(r, device=U.device, dtype=calc).view(1, 1, r, r)
    eye_n = torch.eye(N, device=U.device, dtype=calc).view(1, 1, N, N)
    UtU = U.transpose(-1, -2) @ U
    G = eye_r + (gamma / mu) * UtU
    solved_ut = torch.linalg.solve_ex(G.reshape(B * H, r, r), U.transpose(-1, -2).reshape(B * H, r, N), check_errors=False)[0]
    solved_ut = solved_ut.view(B, H, r, N)
    mix = eye_n / mu - (gamma / (mu * mu)) * (U @ solved_ut)
    return mix.detach().float().abs().mean(dim=(0, 1))


def save_heatmap_grid(items: dict[tuple[str, str], torch.Tensor], *, out_path: Path, title: str, cmap: str) -> None:
    kinds = list(dict.fromkeys(k for k, _stage in items))
    stages = list(dict.fromkeys(stage for _kind, stage in items))
    fig, axes = plt.subplots(len(kinds), len(stages), figsize=(4 * len(stages), 3.6 * len(kinds)), squeeze=False)
    for r, kind in enumerate(kinds):
        for c, stage in enumerate(stages):
            ax = axes[r][c]
            data = normalize01(items[(kind, stage)])
            im = ax.imshow(data, cmap=cmap)
            ax.set_title(f"{kind} {stage}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_erf_paper_grid(
    items: dict[tuple[str, str], torch.Tensor],
    *,
    out_path: Path,
    model_order: list[str],
    stage_order: list[str],
) -> None:
    stage_labels = {"init": "Before training", "best": "After training", "last": "After training"}
    model_labels = {"mha": "MHA", "rrlsso": "RRLSSO-r32"}
    fig, axes = plt.subplots(
        len(stage_order),
        len(model_order),
        figsize=(3.2 * len(model_order) + 0.8, 3.2 * len(stage_order)),
        squeeze=False,
    )
    im = None
    for r, stage in enumerate(stage_order):
        for c, kind in enumerate(model_order):
            ax = axes[r][c]
            data = normalize01(items[(kind, stage)])
            im = ax.imshow(data, cmap="YlGn", vmin=0.0, vmax=1.0, interpolation="bicubic")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(model_labels.get(kind, kind), fontsize=16)
            if c == 0:
                ax.set_ylabel(stage_labels.get(stage, stage), fontsize=16)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
                spine.set_color("#222222")
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.026, pad=0.035)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize input ERF and first-layer mixer maps for ViT MHA/RRLSSO checkpoints.")
    parser.add_argument("--run-dir", default="runs/cv_vit_b4_cifar100_rrlsso")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--models", nargs="+", choices=["mha", "rrlsso"], default=["mha", "rrlsso"])
    parser.add_argument("--stages", nargs="+", default=["init", "best"])
    parser.add_argument("--data-dir", default="data/torchvision")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument(
        "--target",
        choices=["feature_norm", "true_logit", "pred_logit", "center_token_sqnorm"],
        default="feature_norm",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.run_dir) / "erf"
    out_dir.mkdir(parents=True, exist_ok=True)

    loader_args = SimpleNamespace(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    _train_loader, test_loader = make_loaders(loader_args)
    mixer_images, _labels = next(iter(test_loader))
    mixer_images = mixer_images[: min(args.batch_size, args.max_samples)].to(device)

    erf_items: dict[tuple[str, str], torch.Tensor] = {}
    mixer_items: dict[tuple[str, str], torch.Tensor] = {}
    stats: dict[str, dict[str, float]] = {}
    for kind in args.models:
        for stage in args.stages:
            ckpt_path = Path(args.run_dir) / kind / f"{stage}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
            model = load_checkpoint_model(
                ckpt_path,
                kind=kind,
                rank=args.rank,
                image_size=args.image_size,
                patch_size=args.patch_size,
                device=device,
            )
            erf = input_erf(model, test_loader, device=device, max_samples=args.max_samples, target=args.target)
            mix = mixer_matrix(model, mixer_images, kind=kind, layer=args.layer)
            erf_items[(kind, stage)] = erf
            mixer_items[(kind, stage)] = mix
            key = f"{kind}_{stage}"
            stats[f"{key}_erf"] = map_stats(erf)
            stats[f"{key}_mixer_abs"] = {
                **map_stats(mix),
                "diag_frac": float(mix.diag().sum() / mix.sum().clamp_min(1e-12)),
                "cls_row_top10_frac": float(
                    (mix[0].abs() / mix[0].abs().sum().clamp_min(1e-12))
                    .topk(max(1, int(0.1 * mix.shape[-1])))
                    .values.sum()
                ),
            }
            print(key, stats[f"{key}_erf"], stats[f"{key}_mixer_abs"], flush=True)

    save_heatmap_grid(
        erf_items,
        out_path=out_dir / f"input_erf_{args.target}.png",
        title=f"Input ERF, target={args.target}",
        cmap="magma",
    )
    save_erf_paper_grid(
        erf_items,
        out_path=out_dir / f"input_erf_{args.target}_paper_grid.png",
        model_order=args.models,
        stage_order=args.stages,
    )
    save_heatmap_grid(
        mixer_items,
        out_path=out_dir / f"mixer_layer{args.layer}_abs.png",
        title=f"Layer {args.layer} mixer absolute influence",
        cmap="viridis",
    )
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
