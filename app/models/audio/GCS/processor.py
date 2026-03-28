import torch
import os
import librosa
import tempfile
import torchaudio.transforms as T_audio
from spikingjelly.activation_based import functional, neuron

# Local imports
from .model import SRN_KWS
from .profiler import SNNProfiler

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pth')
CLASSES = ['SILENCE', 'UNKNOWN', 'YES', 'NO', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'ON', 'OFF', 'STOP', 'GO']

# Mel transformation matching training
mel_transform = T_audio.MelSpectrogram(sample_rate=16000, n_mels=40, hop_length=160)

# Initialize Model
model = SRN_KWS(num_classes=12, embed_dims=128, T=4).to(DEVICE)

# Load Weights
if os.path.exists(WEIGHTS_PATH):
    print(f"--> [AUDIO] Loading weights from {WEIGHTS_PATH}")
    ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    # Handle both checkpoint formats (dict with 'model' key or direct state_dict)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
else:
    print(f"--> [WARNING] No weights found at {WEIGHTS_PATH}")

model.eval()
profiler = SNNProfiler()

def run_GCS_inference(file_storage):
    """
    Handles audio inference from a file-like object.
    """
    # 1. Save buffer to temp file to support librosa format detection
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        file_storage.seek(0)
        temp_audio.write(file_storage.read())
        temp_path = temp_audio.name

    try:
        # 2. Load and resample to 16kHz mono
        data, _ = librosa.load(temp_path, sr=16000, mono=True)
        waveform = torch.from_numpy(data).float().unsqueeze(0)
    except Exception as e:
        return {"error": f"Audio processing failed: {str(e)}"}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # 3. Standardize length to 1 second (16,000 samples)
    if waveform.shape[1] < 16000:
        waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
    else:
        waveform = waveform[:, :16000]

    # 4. Mel-Spectrogram Extraction & Normalization
    mel = (mel_transform(waveform).squeeze(0) + 1e-9).log()
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    mel_tensor = mel.unsqueeze(0).to(DEVICE) # Shape: (1, 40, 101)

    # 5. Profiling Hooks
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, neuron.LIFNode):
            hooks.append(m.register_forward_hook(profiler.spike_hook))
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
            hooks.append(m.register_forward_hook(profiler.ops_hook))

    # 6. Inference
    profiler.start()
    with torch.no_grad():
        functional.reset_net(model) # Crucial for SNNs: clear membrane potentials
        outputs = model(mel_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        
    metrics = profiler.stop()
    for h in hooks: h.remove()
    functional.reset_net(model)

    return {
        "prediction": CLASSES[prediction.item()],
        "confidence": f"{confidence.item()*100:.2f}",
        "metrics": metrics
    }