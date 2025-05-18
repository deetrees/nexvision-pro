import os
import subprocess
import tarfile
from pathlib import Path

SDXL_URL_MAP = {
    "sdxl-1.0": "https://your-model-url.com/your-sdxl-model.tar",  # <-- replace this with your real URL
}

def download_and_extract(url: str, target_dir: Path):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    tar_path = "/src/tmp.tar"

    print(f"[INFO] Downloading: {url}")
    subprocess.check_call(["wget", url, "-O", tar_path])  # ← simple and portable

    print(f"[INFO] Extracting to: {target_dir}")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=target_dir)

    print("[INFO] Cleanup")
    os.remove(tar_path)


def install_t2i_adapter_cache(model_name: str, cache_path: Path = Path("/src/model-cache")):
    if model_name not in SDXL_URL_MAP:
        raise ValueError(f"Unknown model key: {model_name}")

    model_base_cache = Path(cache_path)
    model_base_cache.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Checking model cache: {model_base_cache}")
    if not any(model_base_cache.iterdir()):
        print(f"[INFO] Cache is empty. Downloading model '{model_name}'...")
        download_and_extract(SDXL_URL_MAP[model_name], model_base_cache)
    else:
        print(f"[INFO] Model cache found. Skipping download.")
``

