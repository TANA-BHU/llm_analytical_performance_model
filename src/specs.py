import yaml
from dataclasses import dataclass

@dataclass
class ModelSpec:
    layers: int; hidden: int; heads: int; kv_heads: int
    head_dim: int; ffn: int; vocab: int

    def params(self):
        attn = 2 * self.hidden * self.heads * self.head_dim \
             + 2 * self.hidden * self.kv_heads * self.head_dim   # Q,O + K,V
        ffn = 3 * self.hidden * self.ffn                          # SwiGLU
        return self.layers * (attn + ffn) + 2 * self.vocab * self.hidden

    def kv_bytes_per_token(self, kv_bytes):
        return 2 * self.layers * self.kv_heads * self.head_dim * kv_bytes

@dataclass
class Hardware:
    peak_tflops: float; mem_bw_gb_s: float; mem_capacity_gb: float
    compute_eff: float = 1.0; mem_eff: float = 1.0
    step_overhead_ms: float = 0.0
    interconnect_gb_s: float = 0.0
    link_latency_us: float = 5.0

def load(path="/home/elyssa/llm_analytical_performance_model/configs/experiment.yaml"):
    cfg = yaml.safe_load(open(path))
    models = {k: ModelSpec(**v) for k, v in cfg["models"].items()}
    hws = {k: Hardware(**v) for k, v in cfg["hardware"].items()}
    return models, hws, cfg["run"]