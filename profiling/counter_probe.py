import torch, triton  # triton import so the shim path matches the real workload

# Known-traffic kernels on a 1 GiB fp32 buffer:
#   sum  x4 -> ~4.295 GB reads, ~0 writes
#   mul  x4 -> ~4.295 GB reads + ~4.295 GB writes
big = torch.randn(268435456, device="cuda", dtype=torch.float32)
dst = torch.empty_like(big)
torch.cuda.synchronize()
s = 0.0
for _ in range(4):
    s += big.sum().item()
for _ in range(4):
    torch.mul(big, 2.0, out=dst)
torch.cuda.synchronize()
print(f"probe-done {s:.1f}")
