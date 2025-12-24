import torch
import os
from transformers import BertTokenizer
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

# Import local architecture and global profiler
from .model import SpikingRetNetText
from app.core.profiler import SNNProfiler

# --- 1. SETTINGS & DEVICE ---
DEVICE = torch.device("cpu")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pth')

# --- 2. INITIALIZE COMPONENTS ---
# Load the BERT tokenizer once during server startup
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Instantiate the model architecture
# Note: Ensure these parameters match your training config exactly
model = SpikingRetNetText(
    vocab_size=30522, 
    max_len=512, 
    num_classes=2,
    embed_dims=256, 
    num_heads=8, 
    depths=2, 
    T=4, 
    backend='torch'  # Force CPU backend
).to(DEVICE)

# Load the trained weights
if os.path.exists(WEIGHTS_PATH):
    print(f"--> [IMDB] Loading weights into CPU from {WEIGHTS_PATH}")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    
    # Handle both full checkpoints and raw state_dicts
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
else:
    print(f"!! [IMDB] Warning: best_model.pth not found in {os.path.dirname(WEIGHTS_PATH)}")

model.eval()

# Global profiler instance for this processor
profiler = SNNProfiler()

def run_imdb_inference(text):
    """
    Takes raw text, runs spiking inference, and returns metrics + result.
    """
    # 1. PRE-PROCESS
    # Tokenize and pad/truncate to 512 tokens
    inputs = tokenizer(
        text, 
        padding="max_length", 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    ).to(DEVICE)
    input_ids = inputs['input_ids']

    # 2. REGISTER PROFILER HOOKS
    # We register hooks dynamically to track this specific inference run
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, MultiStepLIFNode):
            hooks.append(m.register_forward_hook(profiler.spike_hook))
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(profiler.ops_hook))

    # 3. EXECUTION & PROFILING
    profiler.start()
    with torch.no_grad():
        outputs = model(input_ids)
        
        # Calculate result and confidence
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        
    # Stop timing and extract metrics
    metrics = profiler.stop()

    # 4. CLEANUP
    # Reset membrane potentials (Crucial for SNNs)
    functional.reset_net(model)
    # Remove hooks to prevent memory leak/interference with next run
    for h in hooks:
        h.remove()

    # 5. FORMAT OUTPUT
    return {
        "prediction": "Positive Sentiment" if prediction.item() == 1 else "Negative Sentiment",
        "confidence": f"{confidence.item()*100:.2f}",
        "metrics": metrics
    }