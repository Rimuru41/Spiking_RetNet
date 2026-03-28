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
        """Clears all counters.  Call before each new inference."""
        self.total_spikes         = 0
        self.total_neurons        = 0
        self.dense_flops_capacity = 0.0
        self.sparse_sops_capacity = 0.0
        self._start_time          = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Hooks  (register these on the model before each forward pass)
    # ──────────────────────────────────────────────────────────────────────────

    def spike_hook(self, module, inp, output):
        """
        Counts spikes fired by any MultiStepLIFNode.
        Output shape: [T, B, ...]
        """
        spikes = output.detach()
        self.total_spikes  += spikes.sum().item()
        self.total_neurons += spikes.numel()

    def ops_hook(self, module, inp, output):
        """
        Classifies each Linear layer as Dense (MACs) or Sparse (SOPs) by
        checking whether the layer input contains floating-point values
        (dense embedding output) or binary spike values (0 / 1).
        """
        x_in = inp[0].detach()

        # Theoretical op count for this layer
        if isinstance(module, nn.Linear):
            ops = x_in.numel() * module.out_features
        elif isinstance(module, nn.Conv2d):
            kh, kw = module.kernel_size
            ops = x_in.numel() * kh * kw * module.out_channels
        else:
            return

        # Sample up to 1 000 elements to decide float vs binary
        flat   = x_in.flatten()
        n      = flat.numel()
        sample = (flat[torch.randint(0, n, (min(1000, n),), device=flat.device)]
                  if n > 1000 else flat)

        has_decimals = (sample.abs() % 1 > 1e-4).any().item()

        if has_decimals:
            self.dense_flops_capacity += ops
        else:
            self.sparse_sops_capacity += ops

    # ──────────────────────────────────────────────────────────────────────────
    # Timer
    # ──────────────────────────────────────────────────────────────────────────

    def start(self):
        """Resets counters and starts the wall-clock timer."""
        self.reset()
        self._start_time = time.perf_counter()

    def stop(self):
        """
        Stops the timer, computes all metrics, and returns a flat dict.

        Keys
        ----
        neurons         int   total neuron time-steps observed
        total_spikes    int   total spikes fired
        firing_rate     float fraction of neurons that fired  (0–1)
        sparsity        str   "XX.XX%"
        macs            str   dense MAC count (comma-formatted)
        sops            str   actual sparse AC count (comma-formatted)
        energy_encoder  str   dense energy in scientific notation  (J)
        energy_core     str   sparse energy in scientific notation  (J)
        energy          str   total energy in mJ  ("X.XXXX")
        latency         str   wall-clock latency in ms  ("X.XX")
        """
        latency_ms = (time.perf_counter() - self._start_time) * 1000

        # Sparsity
        firing_rate = (self.total_spikes / self.total_neurons
                       if self.total_neurons > 0 else 0.0)
        sparsity = 1.0 - firing_rate

        # Operations
        dense_macs  = self.dense_flops_capacity
        actual_sops = self.sparse_sops_capacity * firing_rate

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
        ops_hook on all Linear modules.  Returns list of hook handles
        — call handle.remove() on each when done.
        """
        handles = []
        for m in model.modules():
            if isinstance(m, MultiStepLIFNode):
                handles.append(m.register_forward_hook(self.spike_hook))
            elif isinstance(m, nn.Linear):
                handles.append(m.register_forward_hook(self.ops_hook))
        return handles