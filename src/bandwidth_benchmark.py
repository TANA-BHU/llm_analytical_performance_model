import torch, time

n = 256 * 1024 * 1024 // 2          # 256 MB of fp16
a = torch.empty(n, device="cuda", dtype=torch.float16).normal_()
b = torch.empty_like(a)

for _ in range(10):
    b.copy_(a)
torch.cuda.synchronize()

iters = 100
t0 = time.perf_counter()
for _ in range(iters):
    b.copy_(a)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / iters

moved = 2 * a.numel() * 2           # read + write, 2 bytes per element
print(f"Measured bandwidth: {moved / dt / 1e9:.1f} GB/s")