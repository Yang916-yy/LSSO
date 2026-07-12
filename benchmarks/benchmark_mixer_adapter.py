from __future__ import annotations

import torch

from lsso import MixerAdapter


def time_cuda(fn, warmup: int = 30, iterations: int = 150) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / iterations


def peak_mib(fn) -> float:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del out
    return (peak - baseline) / 2**20


def run_case(mixer: str, batch: int, height: int, width: int, dim: int, heads: int):
    n = height * width + 1
    layer = MixerAdapter(
        dim=dim, num_heads=heads, mixer=mixer, rank=32, bias=False
    ).cuda().bfloat16().eval()
    x = torch.randn(batch, n, dim, device="cuda", dtype=torch.bfloat16)

    def no_2d():
        return layer(x)

    def with_2d():
        return layer(x, spatial_shape=(height, width), num_prefix_tokens=1)

    no_ms = time_cuda(no_2d)
    two_ms = time_cuda(with_2d)
    no_mem = peak_mib(no_2d)
    two_mem = peak_mib(with_2d)

    layer.train()
    train_x = x.detach().requires_grad_(True)

    def train_no_2d():
        layer.zero_grad(set_to_none=True)
        train_x.grad = None
        layer(train_x).float().square().mean().backward()

    def train_with_2d():
        layer.zero_grad(set_to_none=True)
        train_x.grad = None
        layer(
            train_x, spatial_shape=(height, width), num_prefix_tokens=1
        ).float().square().mean().backward()

    train_no_ms = time_cuda(train_no_2d, warmup=10, iterations=50)
    train_two_ms = time_cuda(train_with_2d, warmup=10, iterations=50)
    return no_ms, two_ms, no_mem, two_mem, train_no_ms, train_two_ms


def main():
    print(torch.cuda.get_device_name())
    print("mixer shape       infer_base infer_2d ratio peak_base peak_2d train_base train_2d ratio")
    for mixer in ("mha", "rrlsso"):
        for batch, height, width, dim, heads in (
            (32, 14, 14, 384, 6),
            (4, 32, 32, 384, 6),
        ):
            values = run_case(mixer, batch, height, width, dim, heads)
            a, b, c, d, e, f = values
            print(
                f"{mixer:6s} {batch}x{height}x{width:<2} "
                f"{a:10.3f} {b:8.3f} {b/a:5.2f}x "
                f"{c:9.1f} {d:7.1f} {e:10.3f} {f:8.3f} {f/e:5.2f}x"
            )


if __name__ == "__main__":
    main()
