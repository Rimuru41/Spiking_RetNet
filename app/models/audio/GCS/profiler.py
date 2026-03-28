import torch
import time
import numpy as np

# Fix for SpikingJelly compatibility
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
        """Counts binary spikes fired (total_spikes) and total possible neuron steps."""
        # output shape is usually (T, B, ...)
        self.total_spikes += output.detach().sum().item()
        self.total_neurons += output.numel()

    def ops_hook(self, module, input, output):
        """
        Calculates theoretical operations capacity.
        Differentiates between Dense MACs (Encoder/Float) and Sparse SOPs (SNN Core/Binary).
        """
        x_in = input[0]
        
        # 1. Calculate Operations Capacity based on layer type
        if isinstance(module, torch.nn.Linear):
            ops = module.in_features * module.out_features
            if x_in.dim() > 2: # Handle (T, B, N, C) or (T, B, C)
                ops *= (x_in.numel() // x_in.shape[-1])
                
        elif isinstance(module, torch.nn.Conv1d):
            # Audio: output features * kernel size * input channels
            k = module.kernel_size[0]
            ops = output.numel() * k * module.in_channels
            
        elif isinstance(module, torch.nn.Conv2d):
            k = module.kernel_size[0] * module.kernel_size[1]
            ops = output.numel() * k * module.in_channels
        else:
            return

        # 2. Category Heuristic: MACs (Float) vs SOPs (Binary Spikes)
        # Check a sample of input to see if it contains floating point values (non 0/1)
        sample = x_in.flatten()[:100]
        is_binary = torch.all((sample == 0) | (sample == 1))
        
        if is_binary:
            # These are Synaptic Operations (SOPs) waiting for a spike to trigger
            self.sparse_sops_capacity += ops
        else:
            # These are Dense MACs (typically the initial Convolution/Encoder)
            self.dense_flops_capacity += ops

    def start(self):
        """Starts the timer and resets stats."""
        self.reset()
        self.start_time = time.perf_counter()

    def stop(self):
        """
        Calculates the Detailed Math Metrics.
        Matches the energy and complexity logic from the single_sample_profiling report.
        """
        latency_ms = (time.perf_counter() - self.start_time) * 1000 
        
        # 1. Neuromorphic Metrics
        firing_rate = self.total_spikes / self.total_neurons if self.total_neurons > 0 else 0
        sparsity = 1.0 - firing_rate
        
        # 2. Operations Count
        # SOPs are only 'actual' when a spike occurs (Capacity * Firing Rate)
        actual_sops = self.sparse_sops_capacity * firing_rate
        dense_macs = self.dense_flops_capacity
        
        # 3. Energy Calculation
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
            "energy": f"{total_energy_j * 1000:.4f}",
            "latency": f"{latency_ms:.2f}"
        }
        
  