"""
processor.py
------------
Loads the trained SpikingRetNetText and exposes a single inference function.

Usage (standalone):
    from processor import run_inference
    result = run_inference("This movie was absolutely fantastic!")
    print(result)

Usage (as package):
    from .processor import run_inference
"""

import os
import torch
from transformers import BertTokenizer
from spikingjelly.clock_driven import functional

# ── Local imports — works both as a package and as a standalone module ────────
try:
    from .model    import SpikingRetNetText
    from .profiler import SNNProfiler
except ImportError:
    from model    import SpikingRetNetText
    from profiler import SNNProfiler


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEVICE       = torch.device("cpu")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")

MODEL_CONFIG = dict(
    vocab_size      = 30522,
    max_len         = 256,
    num_classes     = 2,
    embed_dims      = 256,
    num_heads       = 8,
    depths          = 2,
    T               = 1,
    backend         = "torch",
    dropout         = 0.0,
    drop_path_rate  = 0.0,
)

MAX_LENGTH = 512   # tokenizer truncation length

LABEL_MAP = {0: "Negative Sentiment", 1: "Positive Sentiment"}


# ──────────────────────────────────────────────────────────────────────────────
# Initialise once at import time
# ──────────────────────────────────────────────────────────────────────────────

print("[processor] Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

print("[processor] Building model...")
model = SpikingRetNetText(**MODEL_CONFIG).to(DEVICE)

if os.path.exists(WEIGHTS_PATH):
    print(f"[processor] Loading weights from {WEIGHTS_PATH}")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    # Support both raw state_dict and checkpoint dict saved during training
    state_dict = (checkpoint.get("model_state_dict") or
                  checkpoint.get("model") or
                  checkpoint)
    model.load_state_dict(state_dict, strict=False)
    print("[processor] Weights loaded OK.")
else:
    print(f"[processor] WARNING: {WEIGHTS_PATH} not found — using random weights.")

model.eval()

profiler = SNNProfiler()


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_IMDB_FINALS_inference(text: str) -> dict:
    """
    Runs a single spiking inference on the provided text.

    Parameters
    ----------
    text : str
        Raw review / sentence to classify.

    Returns
    -------
    dict with keys:
        prediction  str   "Positive Sentiment" | "Negative Sentiment"
        confidence  str   "XX.XX"  (percentage, no % sign)
        metrics     dict  full SNNProfiler output (energy, sparsity, latency …)
    """
    # 1. Tokenise (dynamic padding to avoid wasted compute on short texts)
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(DEVICE)

    input_ids      = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 2. Reset SNN membrane state before inference
    functional.reset_net(model)

    # 3. Register profiler hooks
    handles = profiler.register_hooks(model)

    # 4. Forward pass
    profiler.start()
    with torch.no_grad():
        logits       = model(input_ids, attention_mask=attention_mask)
        probs        = torch.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    metrics = profiler.stop()

    # 5. Cleanup
    functional.reset_net(model)
    for h in handles:
        h.remove()

    return {
        "prediction": LABEL_MAP.get(prediction.item(), str(prediction.item())),
        "confidence": f"{confidence.item() * 100:.2f}",
        "metrics":    metrics,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Quick test when run directly
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    samples = [
        "This movie was absolutely fantastic! One of the best films I have ever seen.",
        "Terrible film. Boring, poorly acted, and a complete waste of time.",
        "It was okay. Some good moments but overall pretty average.",
    ]

    for text in samples:
        result = run_inference(text)
        print("\n" + "─" * 60)
        print(f"Text      : {text[:80]}...")
        print(f"Prediction: {result['prediction']}  ({result['confidence']}% confidence)")
        m = result["metrics"]
        print(f"Latency   : {m['latency']} ms")
        print(f"Energy    : {m['energy']} mJ")
        print(f"Sparsity  : {m['sparsity']}")
        print(f"MACs      : {m['macs']}")
        print(f"SOPs      : {m['sops']}")