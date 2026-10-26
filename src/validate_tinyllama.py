import torch, time
from transformers import AutoModelForCausalLM

name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()

PROMPT, NEW = 512, 128
ids = torch.randint(100, 30000, (4, PROMPT), device="cuda")

def run(n_new):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(ids, min_new_tokens=n_new, max_new_tokens=n_new, do_sample=False)
    torch.cuda.synchronize()
    return time.perf_counter() - t0

run(4)                                   # warmup
t_prefill = min(run(1) for _ in range(3))
t_total = min(run(NEW + 1) for _ in range(3))
t_decode = (t_total - t_prefill) / NEW

print(f"Prefill {PROMPT} tokens: {t_prefill*1e3:.1f} ms")
print(f"Decode step: {t_decode*1e3:.2f} ms  ->  {1/t_decode:.1f} tokens/s")