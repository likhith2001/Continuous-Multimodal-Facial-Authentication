import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from src.extract_optical_flow import extract_region

def generate_synthetic_dataset(base_dir, dataset_name, videos_limit=500):
    print(f"\n[Synthetic] Generating Time-Shifted Fakes for {dataset_name.upper()}...")

    if dataset_name == 'grid':
        real_video_root = os.path.join(base_dir, "video")
    elif dataset_name == 'mobio':
        real_video_root = base_dir 
    elif dataset_name == 'faceforensics':
        real_video_root = os.path.join(base_dir, "original_sequences", "youtube", "c23", "videos")
        if not os.path.exists(real_video_root):
            real_video_root = os.path.join(base_dir, "original_sequences")
    else:
        print(f"Dataset {dataset_name} not supported for synthetic generation.")
        return

    output_root = os.path.join(base_dir, "optical_flow", "combined", "fake_synthetic")
    os.makedirs(output_root, exist_ok=True)

    all_videos = []
    if not os.path.exists(real_video_root):
        print(f"Error: Video root not found: {real_video_root}")
        return

    print(f"   -> Scanning for videos in: {real_video_root}")
    for root, _, files in os.walk(real_video_root):
        for f in files:
            if f.endswith(('.mp4', '.avi', '.mpg')):
                all_videos.append(os.path.join(root, f))
    
    if not all_videos:
        print("No videos found to process.")
        return

    print(f"   -> Found {len(all_videos)} real videos.")
    
    random.shuffle(all_videos)
    if videos_limit and len(all_videos) > videos_limit:
        all_videos = all_videos[:videos_limit]

    generated_count = 0
    
    for vid_path in tqdm(all_videos, desc="Creating Synthetic Fakes"):
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        
        save_path = os.path.join(output_root, f"synth_{vid_name}.npy")
        if os.path.exists(save_path): 
            generated_count += 1
            continue

        cap = cv2.VideoCapture(vid_path)
        frames_eye = []
        frames_lip = []
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            eye = extract_region(frame, mode='eye', dataset_name=dataset_name, target_size=(64, 32))
            lip = extract_region(frame, mode='lip', dataset_name=dataset_name, target_size=(64, 32))
            
            if eye is not None and lip is not None:
                frames_eye.append(cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY))
                frames_lip.append(cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY))
        cap.release()

        if len(frames_eye) < 40: continue

        shift = random.randint(5, 15)
        if random.random() > 0.5:
            eyes_stream = frames_eye[:-shift]
            lips_stream = frames_lip[shift:]
        else:
            eyes_stream = frames_eye[shift:]
            lips_stream = frames_lip[:-shift]

        min_len = min(len(eyes_stream), len(lips_stream))
        eyes_stream = eyes_stream[:min_len]
        lips_stream = lips_stream[:min_len]

        flow_frames = []
        for i in range(min_len - 1):
            flow_e = cv2.calcOpticalFlowFarneback(eyes_stream[i], eyes_stream[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_l = cv2.calcOpticalFlowFarneback(lips_stream[i], lips_stream[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            combined_flow = np.vstack([flow_e, flow_l])
            flow_frames.append(combined_flow)

        if len(flow_frames) > 10:
            np.save(save_path, np.array(flow_frames))
            generated_count += 1

    print(f"   [Done] Generated {generated_count} synthetic fake videos.")