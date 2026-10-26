from dataclasses import dataclass

@dataclass
class Hardware:
    name: str
    compute_tflops: float     # TFLOP/s  (math speed)
    memory_bw_gb_s: float     # GB/s     (memory speed)

def estimate_decode_step(model_params_b, weight_bytes, batch_size, hardware):
    params = model_params_b * 1e9                  # 7 -> 7,000,000,000
    flops = 2.0 * params * batch_size              # each weight: 1 multiply + 1 add
    weight_bytes_total = params * weight_bytes     # 7e9 * 2 bytes = 14 GB
    compute_flops_s = hardware.compute_tflops * 1e12
    memory_bytes_s = hardware.memory_bw_gb_s * 1e9

    compute_time_s = flops / compute_flops_s
    memory_time_s = weight_bytes_total / memory_bytes_s     # weights read ONCE per step
    estimated_step_s = max(compute_time_s, memory_time_s)   # slower one wins

    arithmetic_intensity = flops / weight_bytes_total       # FLOP per byte
    roofline_tflops = min(
        hardware.compute_tflops,
        (hardware.memory_bw_gb_s * 1e9 * arithmetic_intensity) / 1e12,
    )

    return {
        "compute_ms": compute_time_s * 1e3,
        "weight_read_ms": memory_time_s * 1e3,
        "estimated_step_ms": estimated_step_s * 1e3,
        "arithmetic_intensity_flop_byte": arithmetic_intensity,
        "roofline_tflops": roofline_tflops,
        "bottleneck": "compute" if compute_time_s >= memory_time_s else "memory",
        "approx_aggregate_tokens_s": batch_size / estimated_step_s if estimated_step_s else 0.0,
    }


def main():
    MODEL_PARAMS_B = 1.0
    WEIGHT_BYTES = 2.0
    COMPUTE_TFLOPS = 10.0
    MEMORY_BW_GB_S = 150.0
    baseline = Hardware("Hypothetical baseline", COMPUTE_TFLOPS, MEMORY_BW_GB_S)

    for batch in (1, 2, 4, 8):                       # batch sweep
        r = estimate_decode_step(MODEL_PARAMS_B, WEIGHT_BYTES, batch, baseline)
        print(batch, r["estimated_step_ms"], r["bottleneck"])

    configs = [                                      # hardware what-if
        Hardware("Baseline", 20, 300),
        Hardware("2x compute", 40, 300),
        Hardware("2x bandwidth", 20, 600),
        Hardware("2x compute + BW", 40, 600),
    ]
    for hw in configs:
        r = estimate_decode_step(MODEL_PARAMS_B, WEIGHT_BYTES, 1, hw)
        print(hw.name, r["estimated_step_ms"], r["bottleneck"])

if __name__ == "__main__":
    main()