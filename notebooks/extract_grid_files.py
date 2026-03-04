import os
import zipfile
import tarfile
from tqdm import tqdm

def extract_limited_data(start_speaker=1, end_speaker=34, limit=500):
    # --- ROBUST RELATIVE PATHS ---
    # Get the directory where this script is running
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths relative to the script location
    GRID_BASE = os.path.join(BASE_DIR, "gridcorpus")
    
    raw_audio_base = os.path.join(GRID_BASE, "raw", "audio")
    raw_video_base = os.path.join(GRID_BASE, "raw", "video")
    target_audio_base = os.path.join(GRID_BASE, "audio")
    target_video_base = os.path.join(GRID_BASE, "video")
    # -----------------------------

    os.makedirs(target_audio_base, exist_ok=True)
    os.makedirs(target_video_base, exist_ok=True)

    print(f"🚀 Starting Extraction in: {GRID_BASE}")
    print(f"   Limit: {limit} files per speaker")

    for i in range(start_speaker, end_speaker + 1):
        s_id = f"s{i}"
        print(f"\nProcessing Speaker {s_id}...")

        # 1. EXTRACT VIDEOS
        zip_path = os.path.join(raw_video_base, f"{s_id}.zip")
        video_out_dir = os.path.join(target_video_base, s_id)
        
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as z:
                all_videos = [f for f in z.namelist() if f.endswith('.mpg')]
                all_videos.sort()
                selected_videos = all_videos[:limit]
                
                if selected_videos:
                    os.makedirs(video_out_dir, exist_ok=True)
                    print(f"   -> Extracting {len(selected_videos)} videos...")
                    for file in tqdm(selected_videos, leave=False):
                        z.extract(file, target_video_base)
        else:
            print(f"   ⚠️ Video archive not found: {zip_path}")

        # 2. EXTRACT AUDIO
        tar_path = os.path.join(raw_audio_base, f"{s_id}.tar")
        audio_out_dir = os.path.join(target_audio_base, s_id)

        if os.path.exists(tar_path):
            with tarfile.open(tar_path, 'r') as t:
                all_audios = [m for m in t.getmembers() if m.name.endswith('.wav')]
                all_audios.sort(key=lambda x: x.name)
                selected_audios = all_audios[:limit]
                
                if selected_audios:
                    os.makedirs(audio_out_dir, exist_ok=True)
                    print(f"   -> Extracting {len(selected_audios)} audios...")
                    t.extractall(path=target_audio_base, members=selected_audios)
        else:
            print(f"   ⚠️ Audio archive not found: {tar_path}")

    print("\n✅ Extraction Complete.")

if __name__ == "__main__":
    try:
        s_start = int(input("Enter starting speaker (e.g. 1): "))
        s_end = int(input("Enter ending speaker (e.g. 34): "))
        extract_limited_data(s_start, s_end, limit=500)
    except ValueError:
        print("Invalid input. Using defaults (s1-s34).")
        extract_limited_data()