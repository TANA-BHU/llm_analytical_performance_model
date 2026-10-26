def ring_allreduce_s(nbytes, n, link_gb_s, latency_us):
    return 2 * (n - 1) / n * nbytes / (link_gb_s * 1e9) + 2 * (n - 1) * latency_us * 1e-6

def alltoall_s(nbytes_per_gpu, n, link_gb_s, latency_us):
    return (n - 1) / n * nbytes_per_gpu / (link_gb_s * 1e9) + (n - 1) * latency_us * 1e-6