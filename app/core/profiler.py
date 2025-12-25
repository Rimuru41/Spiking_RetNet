import torch
import time
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

class SNNProfiler:
    def __init__(self):
        """
        Initializes the profiler with neuromorphic hardware constants.
        E_MAC: Energy per Multiply-Accumulate (Dense/Float ops) - 4.6 pJ
        E_AC: Energy per Accumulate (Sparse/Spiking ops) - 0.9 pJ
        """
        self.E_MAC = 4.6e-12 
        self.E_AC  = 0.9e-12
        self.reset()

    def reset(self):
        """Resets all counters before a new inference."""
        self.total_spikes = 0
        self.total_neurons = 0
        self.dense_flops = 0.0  # Acts as MACs count
        self.sparse_sops_cap = 0.0
        self.start_time = 0

    def spike_hook(self, module, input, output):
        """Counts actual binary spikes fired and total available neurons."""
        self.total_spikes += output.detach().sum().item()
        self.total_neurons += output.numel()

    def ops_hook(self, module, input, output):
        """Differentiates between MACs (Dense) and SOPs Capacity (Sparse)."""
        x_in = input[0]
        
        # Calculate theoretical operations
        if hasattr(module, 'out_features'):
            # Linear Layer
            ops = x_in.numel() * module.out_features
        elif hasattr(module, 'out_channels'):
            # Convolutional Layer
            k = module.kernel_size[0] * module.kernel_size[1]
            ops = x_in.shape[1] * module.out_channels * k * output.shape[-1] * output.shape[-2]
        else:
            return

        # Heuristic: Check if input is float (MACs) or binary (SOPs)
        sample = x_in.flatten()[:100]
        has_decimals = (sample.abs() % 1 > 1e-4).any()
        
        if has_decimals:
            self.dense_flops += ops
        else:
            self.sparse_sops_cap += ops

    def start(self):
        """Marks the start time."""
        self.reset()
        self.start_time = time.perf_counter()

    def stop(self):
        """Calculates final metrics for the UI."""
        latency = (time.perf_counter() - self.start_time) * 1000 
        
        firing_rate = self.total_spikes / self.total_neurons if self.total_neurons > 0 else 0
        actual_sops = self.sparse_sops_cap * firing_rate
        
        # Energy = (MACs * 4.6pJ) + (SOPs * 0.9pJ)
        energy_j = (self.dense_flops * self.E_MAC) + (actual_sops * self.E_AC)
        
        return {
            "energy": f"{energy_j * 1000:.4f}",
            "sparsity": f"{(1.0 - firing_rate)*100:.2f}%",
            "latency": f"{latency:.2f}",
            "sops": f"{int(actual_sops):,}",
            "macs": f"{int(self.dense_flops):,}",
            "neurons": f"{int(self.total_neurons):,}"
        }