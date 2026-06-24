# scripts/download_data.py
import os
from huggingface_hub import snapshot_download

def download_dataset():
    print("⏳ Downloading data and cache from Hugging Face...")
    
    # 获取当前脚本所在目录的上一级，即项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(project_root, "data")
    
    # 自动下载并完美还原目录结构
    snapshot_download(
        repo_id="tyzhou-cs/DD-Elo-Data", 
        repo_type="dataset", 
        local_dir=target_dir
    )
    print(f"✅ Done! All data saved into: {target_dir}")

if __name__ == "__main__":
    download_dataset()