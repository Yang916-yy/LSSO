from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from examples.models.dit import LatentDiT


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_path(path: str) -> Path:
    if os.name != "nt" and len(path) >= 3 and path[1:3] == ":\\":
        drive = path[0].lower()
        rest = path[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(path)


class TinyImageNet(Dataset):
    def __init__(self, root: Path, split: str, image_size: int) -> None:
        self.root = root
        self.split = split
        self.wnids = [line.strip() for line in (root / "wnids.txt").read_text().splitlines() if line.strip()]
        self.class_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}
        self.samples = self._load_samples()
        if not self.samples:
            raise RuntimeError(f"no images found under {root / split}")

        if split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.RandomCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

    def _load_samples(self) -> list[tuple[Path, int]]:
        split_dir = self.root / self.split
        if self.split == "train":
            samples = []
            for wnid in self.wnids:
                image_dir = split_dir / wnid / "images"
                for path in sorted(image_dir.iterdir()):
                    if path.suffix.lower() in IMAGE_EXTS:
                        samples.append((path, self.class_to_idx[wnid]))
            return samples

        ann_path = split_dir / "val_annotations.txt"
        if ann_path.exists():
            samples = []
            for line in ann_path.read_text().splitlines():
                fields = line.split("\t")
                if len(fields) < 2:
                    continue
                image_name, wnid = fields[0], fields[1]
                path = split_dir / "images" / image_name
                if path.exists():
                    samples.append((path, self.class_to_idx[wnid]))
            return samples

        samples = []
        for wnid in self.wnids:
            image_dir = split_dir / wnid
            for path in sorted(image_dir.rglob("*")):
                if path.suffix.lower() in IMAGE_EXTS:
                    samples.append((path, self.class_to_idx[wnid]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        with Image.open(path) as im:
            image = im.convert("RGB")
        return self.transform(image), label


class LatentCacheDataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        self.root = root
        self.split = split
        meta_path = root / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"missing latent cache metadata: {meta_path}")
        self.meta = json.loads(meta_path.read_text())
        self.wnids = self.meta["wnids"]
        split_meta = self.meta["splits"][split]
        latents = []
        labels = []
        for shard_name in split_meta["shards"]:
            shard = torch.load(root / shard_name, map_location="cpu")
            latents.append(shard["latents"])
            labels.append(shard["labels"])
        self.latents = torch.cat(latents, dim=0).contiguous()
        self.labels = torch.cat(labels, dim=0).long().contiguous()
        self.latent_size = int(self.latents.shape[-1])
        if len(self.latents) != int(split_meta["num_samples"]):
            raise RuntimeError(f"{split} cache sample count mismatch")

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.latents[idx], int(self.labels[idx])


def make_noise_schedule(device: torch.device, steps: int) -> torch.Tensor:
    betas = torch.linspace(1e-4, 2e-2, steps, device=device)
    return torch.cumprod(1.0 - betas, dim=0)


def add_noise(latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor, alphas_cumprod: torch.Tensor) -> torch.Tensor:
    a = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    return a.sqrt() * latents + (1.0 - a).sqrt() * noise


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    val_loss: float | None
    val_loss_ema: float | None
    seconds: float
    lr: float


class EMAModel:
    def __init__(self, model: nn.Module, decay: float, warmup: bool = True) -> None:
        self.decay = decay
        self.warmup = warmup
        self.num_updates = 0
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        decay = self.decay
        if self.warmup:
            decay = min(decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        self.backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if name in self.shadow
        }

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        state = {name: tensor.detach().cpu() for name, tensor in self.shadow.items()}
        state["_ema_num_updates"] = torch.tensor(self.num_updates, dtype=torch.long)
        return state


def load_vae(model_id: str, device: torch.device, dtype: torch.dtype, local_files_only: bool) -> nn.Module:
    try:
        from diffusers import AutoencoderKL
    except ImportError as exc:
        raise RuntimeError("diffusers is required: python -m pip install diffusers") from exc

    vae = AutoencoderKL.from_pretrained(model_id, local_files_only=local_files_only).to(device=device, dtype=dtype)
    vae.eval()
    vae.requires_grad_(False)
    return vae


@torch.no_grad()
def encode_latents(vae: nn.Module, images: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    scaling = float(getattr(vae.config, "scaling_factor", 0.18215))
    posterior = vae.encode(images.to(dtype=dtype)).latent_dist
    return posterior.sample() * scaling


def run_eval(
    model: nn.Module,
    vae: nn.Module | None,
    loader: DataLoader,
    alphas_cumprod: torch.Tensor,
    device: torch.device,
    vae_dtype: torch.dtype,
    amp: bool,
    max_batches: int,
    cached_latents: bool,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for step, (batch, labels) in enumerate(loader):
            if step >= max_batches:
                break
            labels = labels.to(device, non_blocking=True)
            if cached_latents:
                latents = batch.to(device, non_blocking=True).float()
            else:
                if vae is None:
                    raise RuntimeError("vae is required when cached_latents is false")
                images = batch.to(device, non_blocking=True)
                latents = encode_latents(vae, images, vae_dtype).float()
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, alphas_cumprod.numel(), (latents.shape[0],), device=device)
            noisy = add_noise(latents, noise, timesteps, alphas_cumprod)
            with torch.amp.autocast(device_type="cuda", enabled=amp):
                pred = model(noisy, timesteps, labels)
                loss = F.mse_loss(pred.float(), noise.float())
            losses.append(float(loss.item()))
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--latent-cache", type=str, default="")
    parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--mixer",
        choices=["lsso", "mha", "hybrid", "hybrid-lsso-first", "top2-mha", "bottom2-mha"],
        default="lsso",
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--gamma-max", type=float, default=0.3)
    parser.add_argument("--theta-gamma-init", type=float, default=-4.0)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-no-warmup", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    data_root = normalize_path(args.data_root) if args.data_root else None
    latent_cache = normalize_path(args.latent_cache) if args.latent_cache else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_dtype = torch.float16 if args.amp and device.type == "cuda" else torch.float32

    if latent_cache is not None:
        train_set = LatentCacheDataset(latent_cache, "train")
        val_set = LatentCacheDataset(latent_cache, "val")
        latent_size = train_set.latent_size
        cached_latents = True
    else:
        if data_root is None:
            raise ValueError("--data-root is required when --latent-cache is not set")
        train_set = TinyImageNet(data_root, "train", args.image_size)
        val_set = TinyImageNet(data_root, "val", args.image_size)
        latent_size = args.image_size // 8
        cached_latents = False
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, min(args.workers, 4)),
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    vae = None if cached_latents else load_vae(args.vae, device, vae_dtype, args.local_files_only)
    model = LatentDiT(
        latent_size=latent_size,
        patch_size=args.patch_size,
        in_channels=4,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.heads,
        rank=args.rank,
        num_classes=len(train_set.wnids),
        mixer=args.mixer,
        gamma_max=args.gamma_max,
        theta_gamma_init=args.theta_gamma_init,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = EMAModel(model, args.ema_decay, warmup=not args.ema_no_warmup) if args.ema else None
    alphas_cumprod = make_noise_schedule(device, args.timesteps)

    run_name = (
        f"latentdit_tiny_{args.mixer}_r{args.rank}_img{args.image_size}_"
        f"d{args.hidden_size}_L{args.depth}_h{args.heads}_s{args.seed}"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{run_name}.jsonl"

    print(
        json.dumps(
            {
                "event": "start",
                "device": str(device),
                "train_samples": len(train_set),
                "val_samples": len(val_set),
                "latent_size": latent_size,
                "tokens": (latent_size // args.patch_size) ** 2,
                "params": sum(p.numel() for p in model.parameters()),
                "log_path": str(log_path),
                "cached_latents": cached_latents,
                "ema": args.ema,
                "ema_decay": args.ema_decay if args.ema else None,
                "ema_warmup": (not args.ema_no_warmup) if args.ema else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    global_step = 0
    model.train()
    with log_path.open("w", encoding="utf-8") as log_f:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            losses = []
            for batch, labels in train_loader:
                labels = labels.to(device, non_blocking=True)
                if cached_latents:
                    latents = batch.to(device, non_blocking=True).float()
                else:
                    images = batch.to(device, non_blocking=True)
                    with torch.no_grad():
                        latents = encode_latents(vae, images, vae_dtype).float()
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, args.timesteps, (latents.shape[0],), device=device)
                noisy = add_noise(latents, noise, timesteps, alphas_cumprod)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", enabled=args.amp and device.type == "cuda"):
                    pred = model(noisy, timesteps, labels)
                    loss = F.mse_loss(pred.float(), noise.float())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)

                losses.append(float(loss.item()))
                global_step += 1
                if args.max_train_steps and global_step >= args.max_train_steps:
                    break

            val_loss = run_eval(
                model,
                vae,
                val_loader,
                alphas_cumprod,
                device,
                vae_dtype,
                amp=args.amp and device.type == "cuda",
                max_batches=args.val_batches,
                cached_latents=cached_latents,
            )
            val_loss_ema = None
            if ema is not None:
                ema.store(model)
                ema.copy_to(model)
                val_loss_ema = run_eval(
                    model,
                    vae,
                    val_loader,
                    alphas_cumprod,
                    device,
                    vae_dtype,
                    amp=args.amp and device.type == "cuda",
                    max_batches=args.val_batches,
                    cached_latents=cached_latents,
                )
                ema.restore(model)
            stats = EpochStats(
                epoch=epoch,
                train_loss=float(sum(losses) / max(1, len(losses))),
                val_loss=val_loss,
                val_loss_ema=val_loss_ema,
                seconds=time.time() - t0,
                lr=optimizer.param_groups[0]["lr"],
            )
            line = json.dumps(asdict(stats), ensure_ascii=False)
            print(line, flush=True)
            log_f.write(line + "\n")
            log_f.flush()

            if args.max_train_steps and global_step >= args.max_train_steps:
                break

    ckpt_path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{run_name}.pt"
    payload = {"model": model.state_dict(), "args": vars(args)}
    if ema is not None:
        payload["ema"] = ema.state_dict()
    torch.save(payload, ckpt_path)
    print(json.dumps({"event": "saved", "ckpt_path": str(ckpt_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
