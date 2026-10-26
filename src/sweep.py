from specs import load
from inference import decode_step, prefill, fits

models, hws, run = load()
m, hw = models[run["model"]], hws[run["hardware"]]
wb, kb = run["weight_bytes"], run["kv_bytes"]

ok, gb = fits(m, hw, run["batch"], run["context_len"], wb, kb)
print(f"params={m.params()/1e9:.2f}B  memory need={gb:.2f} GB  fits={ok}")
print("prefill:", prefill(m, hw, run["batch"], run["prompt_len"], wb, kb))
for b in (1, 4, 16, 64, 128):
    print(b, decode_step(m, hw, b, run["context_len"], wb, kb))