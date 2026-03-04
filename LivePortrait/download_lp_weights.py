import os
from huggingface_hub import snapshot_download

def download_weights():
    # Define the target directory relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(base_dir, "pretrained_weights")
    
    print(f"Downloading LivePortrait weights to: {target_dir}")
    print("This may take a while (approx 2-3 GB)...")
    
    try:
        # Download the main LivePortrait repository
        snapshot_download(
            repo_id="KlingTeam/LivePortrait",
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.git*", "README.md", "docs", "*.jpg", "*.mp4", "*.png"]
        )
        
        print("\n[SUCCESS] Weights downloaded successfully!")
        print(f"Files are located in: {target_dir}")
        
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("Ensure you have an active internet connection.")

if __name__ == "__main__":
    download_weights()