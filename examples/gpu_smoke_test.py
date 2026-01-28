"""Simple GPU smoke test for CUDA matmul + synchronize."""
from __future__ import annotations

import sys

import torch


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is not available. Set up a compatible PyTorch build or use CPU fallback.")
        return 1

    device = torch.device("cuda")
    try:
        a = torch.randn((128, 128), device=device)
        b = torch.randn((128, 128), device=device)
        _ = a @ b
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - surface CUDA failures explicitly
        print(f"CUDA matmul smoke test failed: {exc}")
        return 1

    print("CUDA matmul smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
