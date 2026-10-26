import csv, os
from collections import defaultdict
import matplotlib.pyplot as plt

BW = 450  # which link bandwidth to plot (150, 450 or 900)

series = defaultdict(list)
with open("results/tp_whatif.csv") as f:
    for r in csv.DictReader(f):
        if int(r["link_gb_s"]) == BW:
            series[int(r["latency_us"])].append((int(r["tp"]), float(r["tok_s"])))

plt.figure(figsize=(7, 4.5))
for lat, pts in sorted(series.items()):
    pts.sort()
    plt.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"{lat} µs")

plt.xscale("log", base=2)
plt.xticks([1, 2, 4, 8], [1, 2, 4, 8])
plt.xlabel("Tensor-parallel degree (GPUs)")
plt.ylabel("Decode throughput (tokens/s)")
plt.title(f"Llama-3-8B decode, batch 32, link {BW} GB/s (placeholder HW)")
plt.legend(title="Link latency")
plt.grid(alpha=0.3)
plt.tight_layout()

os.makedirs("results", exist_ok=True)
plt.savefig("results/tp_scaling.png", dpi=150)
print("saved results/tp_scaling.png")