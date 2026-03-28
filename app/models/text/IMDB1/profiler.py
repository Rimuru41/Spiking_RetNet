import torch
import time
import numpy as np

# Fix for SpikingJelly compatibility with newer numpy versions
np.int = int
np.float = float
np.bool = bool

class SNNProfiler:
    def __init__(self):
        """
        Initializes the profiler with neuromorphic hardware constants (45nm CMOS).
        E_MAC: Energy per Multiply-Accumulate (Dense/Float ops) - 4.6 pJ
        E_AC: Energy per Accumulate (Sparse/Spiking ops) - 0.9 pJ
        """
        self.E_MAC = 4.6e-12 
        self.E_AC  = 0.9e-12
        self.reset()

    def reset(self):
        """Resets all counters for a fresh single-sample inference."""
        self.total_spikes = 0
        self.total_neurons = 0
        self.dense_flops_capacity = 0.0  
        self.sparse_sops_capacity = 0.0
        self.start_time = 0

    def spike_hook(self, module, input, output):
        """
        Counts actual spikes fired by LIF Nodes.
        Output shape in SpikingJelly is usually[T, B, ...].
        """
        self.total_spikes += output.detach().sum().item()
        self.total_neurons += output.numel()

    def ops_hook(self, module, input, output):
        """
        Classifies each layer as Dense or Sparse based on whether its input 
        contains floating-point values (dense) or binary spike values (sparse).
        """
        x_in = input[0].detach() # Detach to be safe with memory
        
        # 1. Calculate theoretical operations
        # Your logic: ops = x_in.numel() * module.out_features
        ops = 0.0
        if isinstance(module, torch.nn.Linear):
            ops = x_in.numel() * module.out_features
        elif isinstance(module, torch.nn.Conv2d): # Fallback just in case
            k = module.kernel_size[0] * module.kernel_size[1]
            ops = x_in.numel() * k * module.out_channels
        else:
            return

        # 2. Take a random sample to speed up check
        if x_in.numel() > 1000:
            # torch.randint requires size as a tuple, hence (1000,)
            idx = torch.randint(0, x_in.numel(), (1000,), device=x_in.device)
            sample = x_in.flatten()[idx]
        else:
            sample = x_in.flatten()

        # 3. Check for decimals (float embeddings) vs integers (spikes)
        # Added .item() to safely evaluate the boolean
        has_decimals = (sample.abs() % 1 > 1e-4).any().item()
        
        if has_decimals:
            self.dense_flops_capacity += ops
        else:
            self.sparse_sops_capacity += ops

    def start(self):
        """Starts the timer and resets stats."""
        self.reset()
        self.start_time = time.perf_counter()

    def stop(self):
        """
        Calculates the Detailed Math Metrics.
        Matches the energy and complexity logic from the standalone script.
        """
        latency_ms = (time.perf_counter() - self.start_time) * 1000 
        
        # 1. Neuromorphic Metrics
        firing_rate = self.total_spikes / self.total_neurons if self.total_neurons > 0 else 0.0
        sparsity = 1.0 - firing_rate
        
        # 2. Operations Count
        # SOPs are only 'actual' when a spike occurs (Capacity * Firing Rate)
        actual_sops = self.sparse_sops_capacity * firing_rate
        dense_macs = self.dense_flops_capacity
        
        # 3. Energy Formulas (45nm CMOS)
        energy_encoder_j = dense_macs * self.E_MAC
        energy_snn_core_j = actual_sops * self.E_AC
        total_energy_j = energy_encoder_j + energy_snn_core_j
        
        return {
            "neurons": int(self.total_neurons),
            "total_spikes": int(self.total_spikes),
            "firing_rate": firing_rate,
            "sparsity": f"{sparsity*100:.2f}%",
            "macs": f"{int(dense_macs):,}",
            "sops": f"{int(actual_sops):,}",
            "energy_encoder": f"{energy_encoder_j:.4e}",
            "energy_core": f"{energy_snn_core_j:.4e}",
            "energy": f"{total_energy_j * 1000:.4f}", # Convert to mJ
            "latency": f"{latency_ms:.2f}"
        }