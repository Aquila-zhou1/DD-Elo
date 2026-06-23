# scripts/download_data.py
import os
from huggingface_hub import snapshot_download

HF_TOKEN = None  # Replace with your Hugging Face token if needed

def download_dataset():
    print("⏳ Downloading data and cache from Hugging Face...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(project_root, "data")
    snapshot_download(
        repo_id="tyzhou-cs/DD-Elo-Data", 
        repo_type="dataset", 
        local_dir=target_dir,
        token=HF_TOKEN
    )
    print(f"✅ Done! All data saved into: {target_dir}")

if __name__ == "__main__":
    download_dataset()