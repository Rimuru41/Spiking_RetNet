import os
import requests

HF_BASE_URL = "https://huggingface.co/WhiteTestarossa1/SRN/resolve/main"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_TO_DOWNLOAD = {
    "GCS_best_model.pth": "app/models/audio/GCS/best_model.pth",
    "IMDB_best_model.pth": "app/models/text/IMDB/best_model.pth",
    "IMDB1_best_model.pth": "app/models/text/IMDB1/best_model.pth",
    "IMDB_FINAL_best_model.pth": "app/models/text/IMDB_FINAL/best_model.pth",
    "IMDB_FINALS_best_model.pth": "app/models/text/IMDB_FINALS/best_model.pth",
    "CIFAR10_best_model.pth": "app/models/image/cifar_10/best_model.pth",
    "CIFAR101_best_model.pth.tar": "app/models/image/cifar_10_1/best_model.pth.tar",
}


def download_file(url, dest):
    tmp = dest + ".tmp"

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    os.replace(tmp, dest)


def main():
    print("Checking models...\n")

    for hf_name, relative_path in MODELS_TO_DOWNLOAD.items():
        local_path = os.path.join(BASE_DIR, relative_path)

        if os.path.exists(local_path):
            print(f"✓ Exists: {relative_path}")
            continue

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        url = f"{HF_BASE_URL}/{hf_name}"
        print(f"Downloading {hf_name}...")

        try:
            download_file(url, local_path)
            print(f"✓ Downloaded -> {relative_path}\n")
        except Exception as e:
            print(f"✗ Failed: {hf_name}")
            print(e)


if __name__ == "__main__":
    main()
