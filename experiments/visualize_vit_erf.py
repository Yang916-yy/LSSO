from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def signed_mixer_stats(mix: torch.Tensor) -> dict[str, float]:
    """Summarize signed token influence without conflating excitation/inhibition."""

    mix = mix.detach().float()
    positive = mix.clamp_min(0.0)
    negative = (-mix.clamp_max(0.0))
    magnitude = positive + negative
    signed_mean = mix.mean(dim=(0, 1))
    magnitude_mean = magnitude.mean(dim=(0, 1))
    N = mix.shape[-1]
    offdiag = ~torch.eye(N, device=mix.device, dtype=torch.bool)
    diagonal_magnitude = mix.diagonal(dim1=-2, dim2=-1).abs().sum()
    return {
        "positive_l1": float(positive.sum().cpu()),
        "negative_l1": float(negative.sum().cpu()),
        "negative_l1_frac": float(
            (negative.sum() / magnitude.sum().clamp_min(1e-12)).cpu()
        ),
        "negative_entry_frac": float((mix < 0).float().mean().cpu()),
        "offdiag_negative_entry_frac": float(
            (mix[..., offdiag] < 0).float().mean().cpu()
        ),
        "diagonal_l1_frac": float(
            (diagonal_magnitude / magnitude.sum().clamp_min(1e-12)).cpu()
        ),
        "sign_cancellation_ratio": float(
            (
                signed_mean.abs().sum()
                / magnitude_mean.sum().clamp_min(1e-12)
            ).cpu()
        ),
        "signed_row_sum_mean": float(mix.sum(dim=-1).mean().cpu()),
        "signed_row_sum_std": float(mix.sum(dim=-1).std().cpu()),
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
def mixer_maps(
    model: nn.Module,
    images: torch.Tensor,
    *,
    kind: str,
    layer: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    block, x = encoder_layer_input(model, images, layer)
    if kind == "mha":
        _out, weights = block.self_attention(x, x, x, need_weights=True, average_attn_weights=False)
        mix = weights.detach().float()
        maps = {
            "signed": mix.mean(dim=(0, 1)),
            "magnitude": mix.abs().mean(dim=(0, 1)),
            "negative_fraction": (mix < 0).float().mean(dim=(0, 1)),
            "relation_signed": mix.mean(dim=(0, 1)),
            "relation_magnitude": mix.abs().mean(dim=(0, 1)),
        }
        return maps, signed_mixer_stats(mix)

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
    if rr.length_normalize:
        U = U * (rr.length_reference / N) ** 0.5

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
    relation = -(gamma / (mu * mu)) * (U @ solved_ut)
    mix = eye_n / mu + relation
    mix = mix.detach().float()
    relation = relation.detach().float()
    maps = {
        "signed": mix.mean(dim=(0, 1)),
        "magnitude": mix.abs().mean(dim=(0, 1)),
        "negative_fraction": (mix < 0).float().mean(dim=(0, 1)),
        "relation_signed": relation.mean(dim=(0, 1)),
        "relation_magnitude": relation.abs().mean(dim=(0, 1)),
    }
    stats = signed_mixer_stats(mix)
    stats.update(
        {
            f"relation_{key}": value
            for key, value in signed_mixer_stats(relation).items()
        }
    )
    return maps, stats


def save_signed_heatmap_grid(
    items: dict[tuple[str, str], torch.Tensor],
    *,
    out_path: Path,
    title: str,
) -> None:
    """Render signed matrices with one symmetric scale shared by every panel."""

    kinds = list(dict.fromkeys(k for k, _stage in items))
    stages = list(dict.fromkeys(stage for _kind, stage in items))
    vmax = max(float(value.detach().abs().max().cpu()) for value in items.values())
    vmax = max(vmax, 1e-12)
    fig, axes = plt.subplots(
        len(kinds),
        len(stages),
        figsize=(4 * len(stages), 3.6 * len(kinds)),
        squeeze=False,
    )
    im = None
    for row, kind in enumerate(kinds):
        for col, stage in enumerate(stages):
            ax = axes[row][col]
            data = items[(kind, stage)].detach().float().cpu()
            im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"{kind} {stage}")
            ax.set_xticks([])
            ax.set_yticks([])
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.026, pad=0.035)
    fig.suptitle(title)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_fraction_heatmap_grid(
    items: dict[tuple[str, str], torch.Tensor],
    *,
    out_path: Path,
    title: str,
) -> None:
    """Render frequencies on a fixed 0..1 scale shared across panels."""

    kinds = list(dict.fromkeys(k for k, _stage in items))
    stages = list(dict.fromkeys(stage for _kind, stage in items))
    fig, axes = plt.subplots(
        len(kinds),
        len(stages),
        figsize=(4 * len(stages), 3.6 * len(kinds)),
        squeeze=False,
    )
    im = None
    for row, kind in enumerate(kinds):
        for col, stage in enumerate(stages):
            ax = axes[row][col]
            data = items[(kind, stage)].detach().float().cpu()
            im = ax.imshow(data, cmap="coolwarm", vmin=0.0, vmax=1.0)
            ax.set_title(f"{kind} {stage}")
            ax.set_xticks([])
            ax.set_yticks([])
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.026, pad=0.035)
    fig.suptitle(title)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


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
    mixer_items: dict[str, dict[tuple[str, str], torch.Tensor]] = {
        "signed": {},
        "magnitude": {},
        "negative_fraction": {},
        "relation_signed": {},
        "relation_magnitude": {},
    }
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
            mix_maps, mix_stats = mixer_maps(
                model, mixer_images, kind=kind, layer=args.layer
            )
            erf_items[(kind, stage)] = erf
            for map_name, values in mix_maps.items():
                mixer_items[map_name][(kind, stage)] = values
            key = f"{kind}_{stage}"
            stats[f"{key}_erf"] = map_stats(erf)
            stats[f"{key}_mixer_signed"] = mix_stats
            print(
                key,
                stats[f"{key}_erf"],
                stats[f"{key}_mixer_signed"],
                flush=True,
            )

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
    save_signed_heatmap_grid(
        mixer_items["signed"],
        out_path=out_dir / f"mixer_layer{args.layer}_signed.png",
        title=f"Layer {args.layer} signed total mixer",
    )
    save_signed_heatmap_grid(
        mixer_items["relation_signed"],
        out_path=out_dir / f"mixer_layer{args.layer}_relation_signed.png",
        title=f"Layer {args.layer} signed relational component",
    )
    save_heatmap_grid(
        mixer_items["magnitude"],
        out_path=out_dir / f"mixer_layer{args.layer}_magnitude_normalized.png",
        title=f"Layer {args.layer} normalized influence magnitude (secondary)",
        cmap="viridis",
    )
    save_fraction_heatmap_grid(
        mixer_items["negative_fraction"],
        out_path=out_dir / f"mixer_layer{args.layer}_negative_fraction.png",
        title=f"Layer {args.layer} fraction of negative influence across samples and heads",
    )
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
