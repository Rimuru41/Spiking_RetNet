"""
profiler.py
-----------
Neuromorphic energy & sparsity profiler for SpikingRetNetText.
Uses 45 nm CMOS constants:
    E_MAC = 4.6 pJ  (dense multiply-accumulate)
    E_AC  = 0.9 pJ  (sparse spike accumulate)
"""

import time
import torch
import torch.nn as nn
import numpy as np
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

# SpikingJelly compatibility shim for older numpy
np.int   = int
np.float = float
np.bool  = bool


class SNNProfiler:
    # 45 nm CMOS energy constants
    E_MAC = 4.6e-12   # J per multiply-accumulate  (dense / float)
    E_AC  = 0.9e-12   # J per accumulate            (sparse / spike)

    def __init__(self):
        self.reset()

    # ──────────────────────────────────────────────────────────────────────────
    # State
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self):
        """Clears all counters. Call before each new inference."""
        self.total_spikes         = 0
        self.total_neurons        = 0
        self.dense_flops_capacity = 0.0
        self.sparse_sops_capacity = 0.0
        self.start_time           = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Hooks
    # ──────────────────────────────────────────────────────────────────────────

    def spike_hook(self, module, input, output):
        """
        Counts binary spikes fired (total_spikes) and
        total possible neuron steps (total_neurons).
        Output shape from MultiStepLIFNode: [T, B, ...]
        """
        self.total_spikes  += output.detach().sum().item()
        self.total_neurons += output.numel()

    def ops_hook(self, module, input, output):
        """
        Calculates theoretical operations capacity.
        Differentiates between Dense MACs (Encoder/Float)
        and Sparse SOPs (SNN Core/Binary).
        """
        x_in = input[0]

        # 1. Calculate ops capacity based on layer type
        if isinstance(module, nn.Linear):
            ops = module.in_features * module.out_features
            if x_in.dim() > 2:   # handles [T, B, N, C] or [T*B, N, C]
                ops *= (x_in.numel() // x_in.shape[-1])

        elif isinstance(module, nn.Conv1d):
            k   = module.kernel_size[0]
            ops = output.numel() * k * module.in_channels

        elif isinstance(module, nn.Conv2d):
            k   = module.kernel_size[0] * module.kernel_size[1]
            ops = output.numel() * k * module.in_channels

        else:
            return

        # 2. Dense vs Sparse heuristic
        # Sample first 100 elements — if all 0 or 1 it's binary spikes
        sample    = x_in.flatten()[:100]
        is_binary = torch.all((sample == 0) | (sample == 1)).item()

        if is_binary:
            self.sparse_sops_capacity += ops   # SOPs — triggered only on spikes
        else:
            self.dense_flops_capacity += ops   # MACs — always paid

    # ──────────────────────────────────────────────────────────────────────────
    # Timer
    # ──────────────────────────────────────────────────────────────────────────

    def start(self):
        """Resets counters and starts the wall-clock timer."""
        self.reset()
        self.start_time = time.perf_counter()

    def stop(self):
        """
        Stops the timer, computes all metrics, returns a flat dict.

        Keys
        ----
        neurons         int   total neuron time-steps observed
        total_spikes    int   total spikes fired
        firing_rate     float fraction of neurons that fired  (0-1)
        sparsity        str   "XX.XX%"
        macs            str   dense MAC count  (comma-formatted)
        sops            str   actual sparse AC count  (comma-formatted)
        energy_encoder  str   dense energy  (scientific notation, J)
        energy_core     str   sparse energy (scientific notation, J)
        energy          str   total energy in mJ  ("X.XXXX")
        latency         str   wall-clock latency in ms  ("X.XX")
        """
        latency_ms = (time.perf_counter() - self.start_time) * 1000

        # Neuromorphic metrics
        firing_rate = (self.total_spikes / self.total_neurons
                       if self.total_neurons > 0 else 0.0)
        sparsity = 1.0 - firing_rate

        # Operations
        dense_macs  = self.dense_flops_capacity
        actual_sops = self.sparse_sops_capacity * firing_rate  # capacity * firing rate

        # Energy
        energy_encoder_j = dense_macs  * self.E_MAC
        energy_core_j    = actual_sops * self.E_AC
        total_energy_j   = energy_encoder_j + energy_core_j

        return {
            "neurons":        int(self.total_neurons),
            "total_spikes":   int(self.total_spikes),
            "firing_rate":    firing_rate,
            "sparsity":       f"{sparsity * 100:.2f}%",
            "macs":           f"{int(dense_macs):,}",
            "sops":           f"{int(actual_sops):,}",
            "energy_encoder": f"{energy_encoder_j:.4e}",
            "energy_core":    f"{energy_core_j:.4e}",
            "energy":         f"{total_energy_j * 1000:.4f}",   # mJ
            "latency":        f"{latency_ms:.2f}",
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Convenience: register + deregister hooks in one call
    # ──────────────────────────────────────────────────────────────────────────

    def register_hooks(self, model):
        """
        Registers spike_hook on all MultiStepLIFNode modules and
        ops_hook on all Linear / Conv1d / Conv2d modules.
        Returns list of hook handles — call handle.remove() on each when done.
        """
        handles = []
        for m in model.modules():
            if isinstance(m, MultiStepLIFNode):
                handles.append(m.register_forward_hook(self.spike_hook))
            elif isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                handles.append(m.register_forward_hook(self.ops_hook))
        return handles