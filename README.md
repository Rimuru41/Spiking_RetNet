# Spiking Retentive Networks for Efficient Multi-Modal Sequence Modeling

This repository provides a practical Flask deployment of Spiking Retentive Networks (SRN) across multiple modalities.

The app lets you run inference for:
- Text sentiment analysis
- Image classification
- Audio keyword spotting

Alongside predictions, it reports neuromorphic profiling metrics such as energy estimate, sparsity, latency, MACs, SOPs, and neuron activity.

The theoretical motivation is documented in the included paper:
- [Spiking Retentive Networks for Efficient Multi-Modal Sequence Modeling.pdf](Spiking%20Retentive%20Networks%20for%20Efficient%20Multi-Modal%20Sequence%20Modeling.pdf)

## What This Project Is

Spiking Retentive Networks combine sequence modeling ideas (retention-style processing) with spiking neural computation to improve efficiency.

In this implementation:
- Each modality has its own processor module under [app/models](app/models)
- Models are loaded at app startup from processor files
- Inference requests are routed dynamically based on selected category and dataset
- A profiler tracks spike activity and estimates compute and energy behavior

This is an inference-and-demo platform intended for local testing, benchmarking, and showcasing multi-modal SRN behavior.

## Supported Models

Model choices are discovered automatically from folder structure in [app/models](app/models).

Current categories and datasets in this repository:
- Text: IMDB, IMDB1, IMDB_FINAL, IMDB_FINALS
- Image: cifar_10, cifar_10_1
- Audio: GCS

## Architecture Overview

Main runtime components:
- App entry point: [run.py](run.py)
- Flask app factory: [app/__init__.py](app/__init__.py)
- Route/controller logic: [app/routes.py](app/routes.py)
- Global profiler: [app/core/profiler.py](app/core/profiler.py)
- Frontend dashboard: [app/templates/index.html](app/templates/index.html)

High-level request flow:
1. App starts and preloads processors from each model folder.
2. User selects category/model in web UI and submits text or file input.
3. The selected processor function executes inference.
4. Response returns prediction, confidence, and neuromorphic metrics.

## Setup for Local Hosting

### 1. Clone the repository

```bash
git clone https://github.com/Rimuru41/Spiking_RetNet.git
cd Spiking_RetNet
```

### 2. Create a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download pretrained model weights

Model binaries are intentionally not stored in Git due size limits.

Run:

```bash
python download_models.py
```

This script fetches weights from Hugging Face into the expected folders under [app/models](app/models).

### 5. Start the app

```bash
python run.py
```

By default, Flask serves at:
- http://127.0.0.1:5000

Open that URL in your browser.

## How To Use the Dashboard

1. Select a Data Category.
2. Select a Target Model.
3. Provide input:
- Text models: paste text in the Sequence Input box.
- Image models: upload an image file.
- Audio model (GCS): use microphone recording in the UI.
4. Click Run Analytics.
5. Read prediction plus metrics panel.

## API Notes

Primary endpoint:
- POST /predict

Form fields:
- category: text, image, audio
- model: dataset folder name for that category
- input_data: required for text
- file_data: required for file-based modalities

Typical JSON response shape:

```json
{
	"prediction": "Positive Sentiment",
	"confidence": "97.31",
	"metrics": {
		"energy": "0.1234",
		"sparsity": "91.52%",
		"latency": "18.45",
		"sops": "12,345",
		"macs": "456,789",
		"neurons": 102400
	}
}
```

## Project Structure

```text
Spiking_RetNet/
├── run.py
├── download_models.py
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── routes.py
│   ├── core/
│   │   └── profiler.py
│   ├── models/
│   │   ├── audio/
│   │   ├── image/
│   │   └── text/
│   ├── static/
│   └── templates/
└── README.md
```

## Troubleshooting

### Missing model files

If inference fails due missing checkpoints, rerun:

```bash
python download_models.py
```

### Slow first startup

The app preloads all processor modules on startup in [app/routes.py](app/routes.py), so initial launch can take time while tokenizers/models load.

### GitHub push fails for large model files

Do not commit model binaries larger than 100 MB. Keep them out of Git and fetch them at runtime using [download_models.py](download_models.py).
