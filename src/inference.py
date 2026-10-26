def _times(flops, nbytes, hw):
    t_c = flops / (hw.peak_tflops * 1e12 * hw.compute_eff)
    t_m = nbytes / (hw.mem_bw_gb_s * 1e9 * hw.mem_eff)
    return t_c, t_m

def decode_step(m, hw, batch, ctx, weight_bytes, kv_bytes):
    p = m.params()
    flops = 2 * p * batch + 4 * m.layers * m.heads * m.head_dim * ctx * batch
    nbytes = p * weight_bytes + batch * ctx * m.kv_bytes_per_token(kv_bytes)
    t_c, t_m = _times(flops, nbytes, hw)
    t = max(t_c, t_m) + hw.step_overhead_ms * 1e-3
    return dict(ms=t * 1e3, compute_ms=t_c * 1e3, mem_ms=t_m * 1e3,
                bound="compute" if t_c >= t_m else "memory",
                tokens_s=batch / t, intensity=flops / nbytes)

def prefill(m, hw, batch, prompt, weight_bytes, kv_bytes):
    p = m.params()
    flops = 2 * p * prompt * batch \
          + 2 * m.layers * m.heads * m.head_dim * prompt**2 * batch   # causal attn
    nbytes = p * weight_bytes + batch * prompt * m.kv_bytes_per_token(kv_bytes)
    t_c, t_m = _times(flops, nbytes, hw)
    t = max(t_c, t_m) + hw.step_overhead_ms * 1e-3
    return dict(ms=t * 1e3, bound="compute" if t_c >= t_m else "memory",
                tokens_s=batch * prompt / t)

def fits(m, hw, batch, ctx, weight_bytes, kv_bytes):
    need = m.params() * weight_bytes + batch * ctx * m.kv_bytes_per_token(kv_bytes)
    return need / 1e9 <= hw.mem_capacity_gb, need / 1e9