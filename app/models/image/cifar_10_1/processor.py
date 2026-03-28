"""
processor.py  -  Image / cifar_100
Function name: run_cifar_100_inference  (matches folder name cifar_100)
Loads model exactly the same way as the Kaggle inference script.
"""

import io
import os
import tarfile

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

from .model import spiking_retnet
from .profiler import SNNProfiler

# ─────────────────────────────────────────────────────────────────────────────
# 1.  SETTINGS  (matches Kaggle config exactly)
# ─────────────────────────────────────────────────────────────────────────────
DEVICE   = torch.device("cpu")
IMG_SIZE = 32
CLASSES  = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD  = [0.2470, 0.2435, 0.2616]

_DIR             = os.path.dirname(__file__)
WEIGHTS_PATH_PTH = os.path.join(_DIR, "best_model.pth")
WEIGHTS_PATH_TAR = os.path.join(_DIR, "best_model.pth.tar")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
])

# ─────────────────────────────────────────────────────────────────────────────
# 2.  CHECKPOINT LOADER  — handles .pth and .pth.tar
# ─────────────────────────────────────────────────────────────────────────────
def _load_checkpoint():
    if os.path.exists(WEIGHTS_PATH_PTH):
        print(f"--> [cifar_100] Loading {WEIGHTS_PATH_PTH}")
        return torch.load(WEIGHTS_PATH_PTH, map_location=DEVICE, weights_only=False)

    if os.path.exists(WEIGHTS_PATH_TAR):
        print(f"--> [cifar_100] Loading {WEIGHTS_PATH_TAR} directly (not a real tar)")
        return torch.load(WEIGHTS_PATH_TAR, map_location=DEVICE, weights_only=False)

    return None

# ─────────────────────────────────────────────────────────────────────────────
# 3.  MODEL INIT  — same call as Kaggle: model_retnet.spiking_retnet(...)
# ─────────────────────────────────────────────────────────────────────────────
model = spiking_retnet(
    img_size_h=IMG_SIZE, img_size_w=IMG_SIZE,
    patch_size=4, embed_dims=384, num_heads=12,
    mlp_ratios=4, in_channels=3, num_classes=10,
    depths=4, T=4
).to(DEVICE)

checkpoint = _load_checkpoint()
if checkpoint is not None:
    # Exactly like Kaggle: prefer 'state_dict' key, else use raw dict
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    print("--> [cifar_100] Weights loaded OK.")
else:
    print("!! [cifar_100] No weights found — using random weights.")

model.eval()
profiler = SNNProfiler()

# ─────────────────────────────────────────────────────────────────────────────
# 4.  INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def run_cifar_10_1_inference(file_storage):
    """
    Accepts a Flask FileStorage object.
    Returns { prediction, confidence, metrics }.
    """
    file_storage.seek(0)
    img        = Image.open(io.BytesIO(file_storage.read())).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # register profiler hooks
    hooks = []
    for _, m in model.named_modules():
        if isinstance(m, MultiStepLIFNode):
            hooks.append(m.register_forward_hook(profiler.spike_hook))
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(profiler.ops_hook))

    profiler.start()
    with torch.no_grad():
        functional.reset_net(model)
        outputs       = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    metrics = profiler.stop()

    functional.reset_net(model)
    for h in hooks:
        h.remove()

    return {
        "prediction": CLASSES[prediction.item()],
        "confidence": f"{confidence.item() * 100:.2f}",
        "metrics":    metrics,
    }