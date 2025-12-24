import torch
import os
import io
from PIL import Image
from torchvision import transforms
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

# Import local architecture and global profiler
from .model import SpikingRetNet
from app.core.profiler import SNNProfiler

# --- 1. SETTINGS & DEVICE ---
DEVICE = torch.device("cpu")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pth')
CLASSES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD  = [0.2023, 0.1994, 0.2010]

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])

# --- 2. INITIALIZE COMPONENTS ---
model = SpikingRetNet(
    img_size=32, num_classes=10, embed_dims=384, 
    num_heads=12, depths=4, T=4, backend='torch'
).to(DEVICE)

# Load weights
if os.path.exists(WEIGHTS_PATH):
    print(f"--> [CIFAR10] Loading weights into CPU...")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
else:
    print(f"!! [CIFAR10] Weights not found at {WEIGHTS_PATH}")

model.eval()
profiler = SNNProfiler()

def run_cifar_10_inference(file_storage):
    """
    Handles image processing, inference, and profiling.
    """
    # 1. Pre-process
    file_storage.seek(0)
    img = Image.open(io.BytesIO(file_storage.read())).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # 2. Register Profiler Hooks
    hooks = []
    for name, m in model.named_modules():
        # Track Spikes
        if isinstance(m, MultiStepLIFNode):
            hooks.append(m.register_forward_hook(profiler.spike_hook))
        # Track Operations (Energy)
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(m.register_forward_hook(profiler.ops_hook))

    # 3. Execution & Profiling
    profiler.start()
    with torch.no_grad():
        # Clear SNN states
        functional.reset_net(model)
        
        outputs = model(img_tensor)
        
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        
    metrics = profiler.stop()

    # 4. Cleanup
    functional.reset_net(model)
    for h in hooks:
        h.remove()

    # 5. Result
    return {
        "prediction": CLASSES[prediction.item()],
        "confidence": f"{confidence.item()*100:.2f}",
        "metrics": metrics
    }