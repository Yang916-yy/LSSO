#!/usr/bin/env python3
"""Package a prebuilt LSSO MathDx shared library as a platform wheel."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile


BACKEND_ABI = 1
DIST_NAME = "lsso_mathdx_runtime"
PACKAGE_NAME = "lsso_mathdx_runtime"
PLATFORM_TAG = "linux_x86_64"


def normalized(value: str) -> str:
    return re.sub(r"[-_.]+", "_", value)


def digest(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={encoded.decode()}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="0.2.0")
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--cuda-version", choices=("12.8", "13.0"), required=True)
    parser.add_argument("--mathdx-version", required=True)
    parser.add_argument("--mathdx-license", type=Path, required=True)
    parser.add_argument(
        "--project-license",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "LICENSE",
    )
    parser.add_argument("--architectures", default="80,86,87,89,90,100,120")
    args = parser.parse_args()

    if not args.library.is_file():
        parser.error(f"shared library does not exist: {args.library}")
    if not args.mathdx_license.is_file():
        parser.error(f"MathDx license does not exist: {args.mathdx_license}")
    if not args.project_license.is_file():
        parser.error(f"project license does not exist: {args.project_license}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    local = f"torch{args.torch_version.replace('.', '')}cu{args.cuda_version.replace('.', '')}"
    wheel_version = f"{args.version}+{local}"
    wheel_name = (
        f"{normalized(DIST_NAME)}-{wheel_version}-py3-none-{PLATFORM_TAG}.whl"
    )
    dist_info = f"{normalized(DIST_NAME)}-{wheel_version}.dist-info"
    wheel_path = args.output_dir / wheel_name

    metadata = {
        "backend_abi": BACKEND_ABI,
        "lsso_version": args.version,
        "torch_version": args.torch_version,
        "cuda_version": args.cuda_version,
        "mathdx_version": args.mathdx_version,
        "architectures": args.architectures.split(","),
    }
    init_py = f'''"""Precompiled LSSO MathDx runtime; generated at release time."""
from pathlib import Path

BACKEND_ABI = {BACKEND_ABI}
LSSO_VERSION = {args.version!r}
TORCH_VERSION = {args.torch_version!r}
CUDA_VERSION = {args.cuda_version!r}
MATHDX_VERSION = {args.mathdx_version!r}
ARCHITECTURES = {tuple(args.architectures.split(','))!r}

def library_path() -> Path:
    return Path(__file__).with_name("lib") / "lsso_mathdx.so"
'''.encode()
    package_metadata = (json.dumps(metadata, indent=2) + "\n").encode()
    wheel_metadata = (
        "Wheel-Version: 1.0\n"
        "Generator: LSSO build_mathdx_runtime_wheel.py\n"
        "Root-Is-Purelib: false\n"
        f"Tag: py3-none-{PLATFORM_TAG}\n"
    ).encode()
    core_metadata = (
        "Metadata-Version: 2.3\n"
        "Name: lsso-mathdx-runtime\n"
        f"Version: {wheel_version}\n"
        "Summary: Precompiled CUDA/MathDx backend for LSSO\n"
        "Requires-Python: >=3.10\n"
        f"Requires-Dist: torch=={args.torch_version}\n"
        "License-File: licenses/LICENSE\n"
        "License-File: licenses/LICENSE.NVIDIA-MATHDX\n"
    ).encode()
    notices = (
        "This wheel contains an LSSO application binary compiled using NVIDIA "
        "MathDx. MathDx is governed by the NVIDIA Math Libraries SDK license: "
        "https://docs.nvidia.com/cuda/mathdx/license.html\n"
    ).encode()

    files = {
        f"{PACKAGE_NAME}/__init__.py": init_py,
        f"{PACKAGE_NAME}/metadata.json": package_metadata,
        f"{PACKAGE_NAME}/lib/lsso_mathdx.so": args.library.read_bytes(),
        f"{dist_info}/METADATA": core_metadata,
        f"{dist_info}/WHEEL": wheel_metadata,
        f"{dist_info}/THIRD_PARTY_NOTICES": notices,
        f"{dist_info}/licenses/LICENSE": args.project_license.read_bytes(),
        f"{dist_info}/licenses/LICENSE.NVIDIA-MATHDX": args.mathdx_license.read_bytes(),
    }
    records: list[list[str]] = []
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for name, data in files.items():
            wheel.writestr(name, data)
            records.append([name, digest(data), str(len(data))])
        record_name = f"{dist_info}/RECORD"
        records.append([record_name, "", ""])
        stream = io.StringIO()
        csv.writer(stream, lineterminator="\n").writerows(records)
        wheel.writestr(record_name, stream.getvalue().encode())

    print(wheel_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
