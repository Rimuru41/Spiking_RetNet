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

# def run_imdb_inference(text):
#     """
#     Takes raw text, runs spiking inference, and returns metrics + result.
#     """
#     # 1. PRE-PROCESS 
#     # FIX: Change padding from "max_length" to True. 
#     # This prevents 500+ empty tokens from diluting your sentiment signal.
#     inputs = tokenizer(
#         text, 
#         padding=True,        # Only pad to the length of this specific sentence
#         truncation=True, 
#         max_length=512, 
#         return_tensors="pt"
#     ).to(DEVICE)
#     input_ids = inputs['input_ids']

#     # 2. RESET BEFORE START (Crucial)
#     # Ensure any residual "electrical charge" from the last sentence is gone
#     functional.reset_net(model)

#     # 3. REGISTER PROFILER HOOKS
#     hooks = []
#     for name, m in model.named_modules():
#         if isinstance(m, MultiStepLIFNode):
#             hooks.append(m.register_forward_hook(profiler.spike_hook))
#         if isinstance(m, torch.nn.Linear):
#             hooks.append(m.register_forward_hook(profiler.ops_hook))

#     # 4. EXECUTION
#     profiler.start()
#     with torch.no_grad():
#         # The model now only processes the actual words, making the "Mean" 
#         # calculation much more accurate.
#         outputs = model(input_ids)
        
#         probabilities = torch.softmax(outputs, dim=1)
#         confidence, prediction = torch.max(probabilities, 1)
        
#     metrics = profiler.stop()

#     # 5. CLEANUP
#     # Reset again so the model is clean for the next request
#     functional.reset_net(model)
    
#     for h in hooks:
#         h.remove()

#     return {
#         "prediction": "Positive Sentiment" if prediction.item() == 1 else "Negative Sentiment",
#         "confidence": f"{confidence.item()*100:.2f}",
#         "metrics": metrics
#     }
import torch
import os
from transformers import BertTokenizer
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

# Import local architecture and your global profiler
from .model import SpikingRetNetText
from app.core.profiler import SNNProfiler

# --- 1. SETTINGS & DEVICE ---
DEVICE = torch.device("cpu")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pth')
VOCAB_SIZE = 30522 
MAX_LEN = 512

# --- 2. INITIALIZE COMPONENTS ---
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = SpikingRetNetText(
    vocab_size=VOCAB_SIZE, 
    max_len=MAX_LEN, 
    num_classes=2,
    embed_dims=256, 
    num_heads=8, 
    depths=2, 
    T=4, 
    backend='torch' 
).to(DEVICE)

# Load weights
if os.path.exists(WEIGHTS_PATH):
    print(f"--> [IMDB] Loading weights from {WEIGHTS_PATH}")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
model.eval()

# Initialize your profiler
profiler = SNNProfiler()

def run_imdb_inference(text):
    """
    Takes raw text, runs spiking inference with MASK and PROFILER.
    """
    # 1. PRE-PROCESS
    inputs = tokenizer(
        text, 
        padding=True, # Dynamic padding for speed
        truncation=True, 
        max_length=MAX_LEN, 
        return_tensors="pt"
    ).to(DEVICE)
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask'] # Used to fix 50.6% confidence

    # 2. RESET BEFORE START
    functional.reset_net(model)

    # 3. REGISTER PROFILER HOOKS
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, MultiStepLIFNode):
            hooks.append(m.register_forward_hook(profiler.spike_hook))
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(profiler.ops_hook))

    # 4. EXECUTION & PROFILING
    profiler.start()
    with torch.no_grad():
        # Passing 'mask' here ensures high confidence scores!
        outputs = model(input_ids, mask=attention_mask)
        
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        
    # Stop timing and extract metrics (Energy, Spikes, etc.)
    metrics = profiler.stop()

    # 5. CLEANUP
    functional.reset_net(model) # Clear membrane potential
    for h in hooks:
        h.remove() # Prevent memory leaks

    # 6. FORMAT OUTPUT
    return {
        "prediction": "Positive Sentiment" if prediction.item() == 1 else "Negative Sentiment",
        "confidence": f"{confidence.item()*100:.2f}",
        "metrics": metrics # Now includes your profiler data!
    }