import torch
import os
from transformers import BertTokenizer
from spikingjelly.clock_driven import functional
from .model import SpikingRetNetText
from app.core.profiler import SNNProfiler

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pth')

# Initialize Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = SpikingRetNetText(
    vocab_size=30522, max_len=512, num_classes=2,
    embed_dims=256, num_heads=8, depths=2, T=4, backend='cupy'
).to(DEVICE)

# Load Weights
if os.path.exists(WEIGHTS_PATH):
    print(f"--> Loading IMDB Weights from {WEIGHTS_PATH}")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

profiler = SNNProfiler()

def run_imdb_inference(text):
    # 1. Prepare Data
    inputs = tokenizer(text, padding="max_length", truncation=True, 
                       max_length=512, return_tensors="pt").to(DEVICE)
    input_ids = inputs['input_ids']

    # 2. Register Hooks for this specific run
    hooks = []
    for m in model.named_modules():
        if isinstance(m[1], MultiStepLIFNode):
            hooks.append(m[1].register_forward_hook(profiler.spike_hook))
        if isinstance(m[1], torch.nn.Linear):
            hooks.append(m[1].register_forward_hook(profiler.ops_hook))

    # 3. Execution
    profiler.start()
    with torch.no_grad():
        outputs = model(input_ids)
        prob = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(prob, 1)
    
    metrics = profiler.stop()
    
    # 4. Clean Up
    functional.reset_net(model)
    for h in hooks: h.remove()

    return {
        "prediction": "Positive Review" if pred.item() == 1 else "Negative Review",
        "confidence": f"{conf.item()*100:.2f}",
        "metrics": metrics
    }