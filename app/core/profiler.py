import torch
import time
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

class SNNProfiler:
    def __init__(self):
        """
        Initializes the profiler with neuromorphic hardware constants.
        E_MAC: Energy per Multiply-Accumulate (Dense/Float ops)
        E_AC: Energy per Accumulate (Sparse/Spiking ops)
        """
        self.E_MAC = 4.6e-12  # 4.6 pJ (picojoules)
        self.E_AC  = 0.9e-12  # 0.9 pJ (picojoules)
        self.reset()

    def reset(self):
        """Resets all metrics before a new inference run."""
        self.total_spikes = 0
        self.total_neurons = 0
        self.dense_flops = 0.0
        self.sparse_sops_cap = 0.0
        self.start_time = 0

    def spike_hook(self, module, input, output):
        """
        Hook registered to LIF Nodes.
        Counts actual binary spikes (1s) fired across all timesteps.
        """
        self.total_spikes += output.detach().sum().item()
        self.total_neurons += output.numel()

    def ops_hook(self, module, input, output):
        """
        Hook registered to Linear and Conv layers.
        Differentiates between Dense FLOPs (Frame Encoding) 
        and Sparse SOPs (Synaptic Operations between spikes).
        """
        x_in = input[0]
        
        # Calculate theoretical operations (Connections * Inputs)
        if hasattr(module, 'out_features'):
            # For Linear Layers
            ops = x_in.numel() * module.out_features
        elif hasattr(module, 'out_channels'):
            # For Conv Layers (simplified estimate)
            k = module.kernel_size[0] * module.kernel_size[1]
            ops = x_in.shape[1] * module.out_channels * k * output.shape[-1] * output.shape[-2]
        else:
            return

        # Heuristic: Check if input is float (Dense) or spikes (Sparse)
        # We check a small sample to see if values are integers/binary
        sample = x_in.flatten()[:100]
        has_decimals = (sample.abs() % 1 > 1e-4).any()
        
        if has_decimals:
            self.dense_flops += ops
        else:
            self.sparse_sops_cap += ops

    def start(self):
        """Marks the start time of the inference."""
        self.reset()
        self.start_time = time.perf_counter()

    def stop(self):
        """
        Stops timing and calculates final neuromorphic metrics.
        Returns a dictionary of formatted strings for the UI.
        """
        # Calculate latency in milliseconds
        latency = (time.perf_counter() - self.start_time) * 1000 
        
        # Calculate firing rate and actual Synaptic Operations (SOPs)
        firing_rate = self.total_spikes / self.total_neurons if self.total_neurons > 0 else 0
        actual_sops = self.sparse_sops_cap * firing_rate
        
        # Energy = (Dense FLOPs * E_MAC) + (Sparse SOPs * E_AC)
        energy_j = (self.dense_flops * self.E_MAC) + (actual_sops * self.E_AC)
        
        return {
            "energy": f"{energy_j * 1000:.4f}", # Convert Joules to mJ
            "sparsity": f"{(1.0 - firing_rate)*100:.2f}%",
            "latency": f"{latency:.2f}",
            "sops": f"{int(actual_sops):,}"
        }