import torch
import os
import io
from PIL import Image
from torchvision import transforms
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from .model import SpikingRetNet
from app.core.profiler import SNNProfiler

# 1. FORCE DEVICE TO CPU
DEVICE = torch.device("cpu")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pth')
CLASSES = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# 2. DEFINE THE TRANSFORM (Including Resize to 32x32)
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # <--- THIS CONVERTS UPLOADED IMAGE TO 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# 3. INITIALIZE MODEL (Matching training config)
model = SpikingRetNet(
    img_size=32, 
    embed_dims=384, 
    num_heads=12, 
    depths=4, 
    T=4, 
    backend='torch'
).to(DEVICE)

# 4. LOAD WEIGHTS
if os.path.exists(WEIGHTS_PATH):
    print(f"--> [CIFAR-10] Loading weights into CPU...")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    # Clean keys if they were saved with DataParallel 'module.' prefix
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)

model.eval()
profiler = SNNProfiler()

def run_cifar_10_inference(file_storage):
    """
    Resizes uploaded image to 32x32, runs spiking inference, and returns metrics.
    """
    # 1. Load and Transform Image
    # Reset file pointer to beginning just in case
    file_storage.seek(0)
    img = Image.open(io.BytesIO(file_storage.read())).convert('RGB')
    
    # transform(img) handles the Resize to 32x32 and Normalization
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # 2. Attach Hooks for Profiling
    hooks = []
    for m in model.modules():
        if isinstance(m, MultiStepLIFNode):
            hooks.append(m.register_forward_hook(profiler.spike_hook))
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(m.register_forward_hook(profiler.ops_hook))

    # 3. Execution
    profiler.start()
    with torch.no_grad():
        outputs = model(img_tensor)
        prob = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(prob, 1)
    
    metrics = profiler.stop()

    # 4. Clean Up
    functional.reset_net(model)
    for h in hooks: 
        h.remove()

    return {
        "prediction": CLASSES[pred.item()],
        "confidence": f"{conf.item()*100:.2f}",
        "metrics": metrics
    }