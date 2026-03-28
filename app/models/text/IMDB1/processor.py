import torch
import os
from transformers import BertTokenizer
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

# Import local architecture and local profiler
from .model import SpikingRetNetText
from .profiler import SNNProfiler

# --- 1. SETTINGS & DEVICE ---
DEVICE = torch.device("cpu")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pth')

# --- 2. INITIALIZE COMPONENTS ---
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = SpikingRetNetText(
    vocab_size=30522, 
    max_len=512, 
    num_classes=2,
    embed_dims=256, 
    num_heads=8, 
    depths=2, 
    T=4, 
    backend='torch' 
).to(DEVICE)

# Load trained weights
if os.path.exists(WEIGHTS_PATH):
    print(f"--> [IMDB] Loading weights into CPU from {WEIGHTS_PATH}")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
else:
    print(f"!! [IMDB] Warning: best_model.pth not found!")

model.eval()
profiler = SNNProfiler()

def run_IMDB1_inference(text):
    """
    Takes raw text, runs spiking inference, and returns metrics + result.
    """
    # 1. PRE-PROCESS (Dynamic padding to save CPU time)
    inputs = tokenizer(
        text, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    ).to(DEVICE)
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # 2. RESET BEFORE START (Crucial for SNN state)
    functional.reset_net(model)

    # 3. REGISTER PROFILER HOOKS
    hooks = []
    for m in model.modules():
        if isinstance(m, MultiStepLIFNode):
            hooks.append(m.register_forward_hook(profiler.spike_hook))
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(profiler.ops_hook))

    # 4. EXECUTION
    profiler.start()
    with torch.no_grad():
        # Mask-aware forward call
        outputs = model(input_ids, attention_mask=attention_mask)
        
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        
    metrics = profiler.stop()

    # 5. CLEANUP
    functional.reset_net(model)
    for h in hooks:
        h.remove()

    return {
        "prediction": "Positive Sentiment" if prediction.item() == 1 else "Negative Sentiment",
        "confidence": f"{confidence.item()*100:.2f}",
        "metrics": metrics
    }