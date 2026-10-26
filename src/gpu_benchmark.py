import argparse, time

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=1024)
parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--iterations", type=int, default=50)
args = parser.parse_args()

try:
    import torch
except ImportError:
    raise SystemExit("PyTorch is not installed...")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch.")

device = torch.device("cuda:0")
dtype = torch.float16 if args.dtype == "fp16" else torch.float32
a = torch.randn((args.size, args.size), device=device, dtype=dtype)
b = torch.randn((args.size, args.size), device=device, dtype=dtype)     

for _ in range(args.warmup):
    c = a @ b
torch.cuda.synchronize()          # wait for the GPU to finish

start = time.perf_counter()
for _ in range(args.iterations):
    c = a @ b
torch.cuda.synchronize()          # must sync BEFORE stopping the clock
elapsed = time.perf_counter() - start

seconds_per_matmul = elapsed / args.iterations
flops_per_matmul = 2.0 * args.size**3       # 2*M*N*K with M=N=K
achieved_tflops = flops_per_matmul / seconds_per_matmul / 1e12 

del a, b, c
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Matrix: {args.size} x {args.size}, dtype={args.dtype}")
print(f"Average matmul: {seconds_per_matmul * 1e3:.3f} ms")
print(f"Achieved throughput: {achieved_tflops:.3f} TFLOP/s")
torch.cuda.empty_cache()