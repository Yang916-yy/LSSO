from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from train_latent_dit_tiny import TinyImageNet, load_vae, normalize_path


@torch.no_grad()
def encode_mean_latents(vae, images: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    scaling = float(getattr(vae.config, "scaling_factor", 0.18215))
    posterior = vae.encode(images.to(dtype=dtype)).latent_dist
    return posterior.mean * scaling


def deterministic_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


class Food101Images(Dataset):
    def __init__(self, root: Path, split: str, image_size: int) -> None:
        self.root = root
        self.split = split
        self.classes = [
            line.strip()
            for line in (root / "meta" / "classes.txt").read_text().splitlines()
            if line.strip()
        ]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        meta_name = "train.txt" if split == "train" else "test.txt"
        self.samples = []
        for line in (root / "meta" / meta_name).read_text().splitlines():
            rel = line.strip()
            if not rel:
                continue
            class_name = rel.split("/", 1)[0]
            self.samples.append((root / "images" / f"{rel}.jpg", self.class_to_idx[class_name]))
        self.transform = deterministic_transform(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        with Image.open(path) as im:
            image = im.convert("RGB")
        return self.transform(image), label


def make_dataset(name: str, root: Path, split: str, image_size: int):
    if name == "tiny-imagenet":
        dataset = TinyImageNet(root, split, image_size)
        dataset.transform = deterministic_transform(image_size)
        class_names = dataset.wnids
    elif name == "food101":
        dataset = Food101Images(root, split, image_size)
        class_names = dataset.classes
    else:
        raise ValueError(f"unknown dataset: {name}")
    return dataset, class_names


def cache_split(args, split: str, vae, device: torch.device, vae_dtype: torch.dtype, out_dir: Path) -> dict:
    dataset, _ = make_dataset(args.dataset, args.data_root, split, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )

    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    latents_buf = []
    labels_buf = []
    shard_idx = 0
    written = 0
    t0 = time.time()

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        latents = encode_mean_latents(vae, images, vae_dtype).to(torch.float16).cpu()
        latents_buf.append(latents)
        labels_buf.append(labels.long().cpu())

        buffered = sum(x.shape[0] for x in latents_buf)
        if buffered >= args.shard_size or step == len(loader):
            shard_latents = torch.cat(latents_buf, dim=0).contiguous()
            shard_labels = torch.cat(labels_buf, dim=0).contiguous()
            shard_name = f"{split}/shard_{shard_idx:05d}.pt"
            torch.save({"latents": shard_latents, "labels": shard_labels}, out_dir / shard_name)
            shards.append(shard_name)
            written += int(shard_labels.numel())
            latents_buf.clear()
            labels_buf.clear()
            shard_idx += 1
            print(
                json.dumps(
                    {
                        "split": split,
                        "written": written,
                        "total": len(dataset),
                        "shards": shard_idx,
                        "seconds": time.time() - t0,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    return {"num_samples": len(dataset), "shards": shards}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--dataset", choices=["tiny-imagenet", "food101"], default="tiny-imagenet")
    parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    args.data_root = normalize_path(args.data_root)
    out_dir = normalize_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_dtype = torch.float16 if args.amp and device.type == "cuda" else torch.float32
    vae = load_vae(args.vae, device, vae_dtype, args.local_files_only)

    _, class_names = make_dataset(args.dataset, args.data_root, "train", args.image_size)
    meta = {
        "dataset": args.dataset,
        "vae": args.vae,
        "image_size": args.image_size,
        "latent_size": args.image_size // 8,
        "latent_dtype": "float16",
        "latent_source": "posterior_mean",
        "scaling_factor": float(getattr(vae.config, "scaling_factor", 0.18215)),
        "wnids": class_names,
        "splits": {},
    }

    print(json.dumps({"event": "start", "out_dir": str(out_dir), "device": str(device)}, ensure_ascii=False), flush=True)
    meta["splits"]["train"] = cache_split(args, "train", vae, device, vae_dtype, out_dir)
    meta["splits"]["val"] = cache_split(args, "val", vae, device, vae_dtype, out_dir)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", "meta": str(out_dir / "meta.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
