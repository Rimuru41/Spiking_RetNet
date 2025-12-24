import torch
import time
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

class SNNProfiler:
    def __init__(self):
        self.E_MAC = 4.6e-12 
        self.E_AC  = 0.9e-12
        self.reset()

    def reset(self):
        self.total_spikes = 0
        self.total_neurons = 0
        self.dense_flops = 0.0
        self.sparse_sops_cap = 0.0
        self.start_time = 0

    def spike_hook(self, module, input, output):
        self.total_spikes += output.detach().sum().item()
        self.total_neurons += output.numel()

    def ops_hook(self, module, input, output):
        x_in = input[0]
        ops = x_in.numel() * module.out_features
        
        # Heuristic to check for dense vs sparse based on your code logic
        # First layer (embedding/pos) is usually dense
        sample = x_in.flatten()[:100]
        has_decimals = (sample.abs() % 1 > 1e-4).any()
        
        if has_decimals:
            self.dense_flops += ops
        else:
            self.sparse_sops_cap += ops

    def start(self):
        self.reset()
        self.start_time = time.perf_counter()

    def stop(self):
        latency = (time.perf_counter() - self.start_time) * 1000 # ms
        firing_rate = self.total_spikes / self.total_neurons if self.total_neurons > 0 else 0
        actual_sops = self.sparse_sops_cap * firing_rate
        
        energy_j = (self.dense_flops * self.E_MAC) + (actual_sops * self.E_AC)
        
        return {
            "energy": f"{energy_j * 1000:.4f}", # mJ
            "sparsity": f"{(1.0 - firing_rate)*100:.2f}%",
            "latency": f"{latency:.2f}",
            "sops": f"{int(actual_sops):,}"
        }