import csv, copy
from specs import load
from inference import decode_step_tp

models, hws, run = load()
m = models["llama3_8b"]
base = hws["hypothetical_big_gpu"]

rows = []
for lat in (1, 2, 5, 10, 20):
    for bw in (150, 450, 900):
        for n in (1, 2, 4, 8):
            hw = copy.copy(base)
            hw.link_latency_us, hw.interconnect_gb_s = lat, bw
            r = decode_step_tp(m, hw, 32, 2048, 2, 2, n)
            rows.append(dict(latency_us=lat, link_gb_s=bw, tp=n,
                             step_ms=round(r["ms"], 3), comm_ms=round(r["comm_ms"], 3),
                             bound=r["bound"], tok_s=round(r["tokens_s"])))

import os; os.makedirs("results", exist_ok=True)
with open("results/tp_whatif.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} rows to results/tp_whatif.csv")