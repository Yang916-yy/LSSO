from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from examples.models.dit import LatentDiT
from train_latent_dit_tiny import LatentCacheDataset, load_vae, normalize_path


def build_model(ckpt_args: dict, num_classes: int, device: torch.device) -> LatentDiT:
    image_size = int(ckpt_args.get("image_size", 256))
    patch_size = int(ckpt_args.get("patch_size", 2))
    model = LatentDiT(
        latent_size=image_size // 8,
        patch_size=patch_size,
        in_channels=4,
        hidden_size=int(ckpt_args.get("hidden_size", 96)),
        depth=int(ckpt_args.get("depth", 2)),
        num_heads=int(ckpt_args.get("heads", 6)),
        rank=int(ckpt_args.get("rank", 16)),
        num_classes=num_classes,
        mixer=str(ckpt_args.get("mixer", "lsso")),
        gamma_max=float(ckpt_args.get("gamma_max", 0.3)),
        theta_gamma_init=float(ckpt_args.get("theta_gamma_init", -4.0)),
    )
    return model.to(device)


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    scaling = float(getattr(vae.config, "scaling_factor", 0.18215))
    dtype = next(vae.parameters()).dtype
    latents = latents.to(dtype=dtype)
    images = vae.decode(latents / scaling).sample
    return (images.clamp(-1, 1) + 1) * 0.5


@torch.no_grad()
def sample_model(
    model: LatentDiT,
    num_images: int,
    num_classes: int,
    steps: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from diffusers import DDIMScheduler

    generator = torch.Generator(device=device).manual_seed(seed)
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=False,
    )
    scheduler.set_timesteps(steps, device=device)

    latents = torch.randn(
        num_images,
        4,
        model.latent_size,
        model.latent_size,
        device=device,
        generator=generator,
    )
    labels = torch.arange(num_images, device=device) % num_classes
    model.eval()
    for t in scheduler.timesteps:
        timesteps = torch.full((num_images,), int(t.item()), device=device, dtype=torch.long)
        noise_pred = model(latents, timesteps, labels)
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    return latents, labels


def save_reconstruction_grid(cache_dir: Path, vae, out_path: Path, num_images: int, device: torch.device) -> None:
    dataset = LatentCacheDataset(cache_dir, "val")
    latents = dataset.latents[:num_images].to(device).float()
    images = decode_latents(vae, latents)
    save_image(images, out_path, nrow=int(num_images**0.5), padding=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--latent-cache", type=str, default="")
    parser.add_argument("--num-classes", type=int, default=200)
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--recon-out", type=str, default="")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()

    ckpt_path = normalize_path(args.ckpt)
    out_path = normalize_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_dtype = torch.float16 if args.amp and device.type == "cuda" else torch.float32

    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    cache_dir = normalize_path(args.latent_cache) if args.latent_cache else None
    num_classes = args.num_classes
    if cache_dir is not None and (cache_dir / "meta.json").exists():
        meta = json.loads((cache_dir / "meta.json").read_text())
        num_classes = len(meta["wnids"])

    model = build_model(ckpt_args, num_classes, device)
    model.load_state_dict(ckpt["model"], strict=True)
    if "ema" in ckpt and not args.no_ema:
        model_state = model.state_dict()
        for name, tensor in ckpt["ema"].items():
            if name.startswith("_"):
                continue
            if name in model_state:
                model_state[name].copy_(tensor)
    vae = load_vae(args.vae, device, vae_dtype, args.local_files_only)

    with torch.amp.autocast(device_type="cuda", enabled=args.amp and device.type == "cuda"):
        latents, labels = sample_model(model, args.num_images, num_classes, args.steps, device, args.seed)
        images = decode_latents(vae, latents)
    nrow = int(args.num_images**0.5)
    save_image(images, out_path, nrow=nrow, padding=2)
    print(json.dumps({"out": str(out_path), "labels": labels.cpu().tolist()}, ensure_ascii=False), flush=True)

    if args.recon_out:
        if cache_dir is None:
            raise ValueError("--latent-cache is required for --recon-out")
        recon_out = normalize_path(args.recon_out)
        recon_out.parent.mkdir(parents=True, exist_ok=True)
        save_reconstruction_grid(cache_dir, vae, recon_out, args.num_images, device)
        print(json.dumps({"recon_out": str(recon_out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
