from specs import load
from inference import decode_step_tp

models, hws, run = load()
m, hw = models["llama3_8b"], hws["hypothetical_big_gpu"]
for batch in (1, 32, 128):
    for n in (1, 2, 4, 8):
        r = decode_step_tp(m, hw, batch, 2048, 2, 2, n)
        print(f"batch={batch:3d} TP={n} step={r['ms']:6.2f} ms  mem={r['mem_ms']:5.2f}  "
              f"comm={r['comm_ms']:5.2f}  bound={r['bound']:7s} tok/s={r['tokens_s']:8.0f}")