import os
import subprocess
import sys
import random
import cv2
import glob
import numpy as np
import dlib
from imutils import face_utils

def extract_audio_from_video(video_path, audio_output_path):
    command = [
        "ffmpeg", 
        "-y",
        "-i", video_path,   
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-loglevel", "error",
        audio_output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def restore_wav2lip_background(real_vid_path, fake_vid_path, out_vid_path, detector, predictor):
    cap_real = cv2.VideoCapture(real_vid_path)
    cap_fake = cv2.VideoCapture(fake_vid_path)
    
    fps = cap_real.get(cv2.CAP_PROP_FPS)
    w = int(cap_real.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_real.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    temp_silent_path = out_vid_path.replace(".avi", "_silent_blend.avi")
    writer = cv2.VideoWriter(temp_silent_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
    
    while True:
        ret_r, frame_r = cap_real.read()
        ret_f, frame_f = cap_fake.read()
        if not ret_r or not ret_f: break
        
        gray = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        
        if len(rects) > 0:
            shape = predictor(gray, rects[0])
            shape = face_utils.shape_to_np(shape)
            mouth_pts = shape[48:68]
            
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, mouth_pts, 255)
            mask = cv2.dilate(mask, np.ones((15,15), np.uint8), iterations=1)
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            
            mask_float = mask.astype(np.float32) / 255.0
            mask_inv = 1.0 - mask_float
            
            mask_3ch = cv2.merge([mask_float, mask_float, mask_float])
            inv_3ch = cv2.merge([mask_inv, mask_inv, mask_inv])
            
            frame_r_float = frame_r.astype(np.float32)
            frame_f_float = frame_f.astype(np.float32)
            
            blended = (frame_f_float * mask_3ch + frame_r_float * inv_3ch).astype(np.uint8)
            writer.write(blended)
        else:
            writer.write(frame_f)
            
    cap_real.release()
    cap_fake.release()
    writer.release()
    
    command = [
        "ffmpeg", "-y", "-i", temp_silent_path, "-i", fake_vid_path,
        "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
        "-loglevel", "error", out_vid_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        pass
    
    if os.path.exists(temp_silent_path):
        os.remove(temp_silent_path)

def generate_fakes_wav2lip(
    base_dir, 
    wav2lip_dir, 
    dataset_name, 
    videos_per_user, 
    all_speakers, 
    step,
    predictor_path=None
):
    output_base_dir = os.path.join(base_dir, "fake_dataset_w2l")
    os.makedirs(output_base_dir, exist_ok=True)
    
    detector = None
    predictor = None
    if predictor_path and os.path.exists(predictor_path):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
    
    temp_dir = os.path.join(wav2lip_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_audio_path = os.path.join(temp_dir, "temp_extracted.wav")

    checkpoint_path = os.path.join(wav2lip_dir, "checkpoints/wav2lip.pth")
    print(f"   -> Checking Wav2Lip Data (Target: {output_base_dir})...")
    
    skipped = 0; generated = 0; errors = 0

    for spk in all_speakers:
        if dataset_name == 'faceforensics':
            video_dir = os.path.join(base_dir, "original_sequences/youtube/c23/videos")
            user_id = spk
            video_files = [os.path.join(video_dir, spk + ".mp4")]
            audio_dir = None
        elif dataset_name == 'grid':
            base_vid_path = os.path.join(base_dir, "video", spk)
            nested_vid_path = os.path.join(base_vid_path, spk)
            video_dir = nested_vid_path if os.path.exists(nested_vid_path) else base_vid_path
            base_aud_path = os.path.join(base_dir, "audio", spk)
            nested_aud_path = os.path.join(base_aud_path, spk)
            audio_dir = nested_aud_path if os.path.exists(nested_aud_path) else base_aud_path
            user_id = spk
            video_files = []
            if os.path.exists(video_dir):
                for root, dirs, files in os.walk(video_dir):
                    for f in files:
                        if f.lower().endswith(('.mp4', '.avi', '.mpg')):
                            video_files.append(os.path.join(root, f))
        else:
            parts = spk.split('/')
            video_dir = os.path.join(base_dir, *parts) 
            audio_dir = None
            user_id = parts[-1] 
            video_files = []
            if os.path.exists(video_dir):
                for root, dirs, files in os.walk(video_dir):
                    for f in files:
                        if f.lower().endswith(('.mp4', '.avi', '.mpg')):
                            video_files.append(os.path.join(root, f))

        speaker_out_dir = os.path.join(output_base_dir, user_id)
        os.makedirs(speaker_out_dir, exist_ok=True)
        
        video_files.sort()
        selected_videos = video_files[::step][:videos_per_user]

        for video_path in selected_videos:
            if not os.path.exists(video_path): continue
            vid_name = os.path.splitext(os.path.basename(video_path))[0]
            if dataset_name == 'mobio':
                parent = os.path.basename(os.path.dirname(video_path))
                save_name = f"fake_w2l_{parent}_{vid_name}.avi"
            else:
                save_name = f"fake_w2l_{vid_name}.avi"

            output_path = os.path.join(speaker_out_dir, save_name)
            
            if os.path.exists(output_path):
                skipped += 1
                continue

            if dataset_name == 'grid':
                wav_name = vid_name + '.wav'
                audio_path = os.path.join(audio_dir, wav_name)
                if not os.path.exists(audio_path): continue 
            else:
                if not extract_audio_from_video(video_path, temp_audio_path):
                    errors += 1
                    continue
                audio_path = temp_audio_path

            temp_w2l_out = os.path.join(temp_dir, f"raw_{save_name}")
            
            command = [
                "python", "inference.py",
                "--checkpoint_path", checkpoint_path,
                "--face", os.path.abspath(video_path),
                "--audio", os.path.abspath(audio_path),
                "--outfile", os.path.abspath(temp_w2l_out),
                "--pads", "0", "10", "0", "0", 
                "--resize_factor", "1",
                "--nosmooth"
            ]
            
            try:
                subprocess.run(command, capture_output=True, cwd=wav2lip_dir, text=True, check=True)
                
                if os.path.exists(temp_w2l_out):
                    if detector and predictor:
                        restore_wav2lip_background(video_path, temp_w2l_out, output_path, detector, predictor)
                    else:
                        if os.path.exists(output_path): os.remove(output_path)
                        os.rename(temp_w2l_out, output_path)
                    
                    if os.path.exists(temp_w2l_out): os.remove(temp_w2l_out)
                    generated += 1
                else:
                    errors += 1
            except subprocess.CalledProcessError:
                errors += 1
                continue

    return generated, skipped, errors

def generate_fakes_fomm(
    base_dir, 
    fomm_dir, 
    dataset_name, 
    videos_per_user, 
    all_speakers, 
    step
):
    output_base_dir = os.path.join(base_dir, "fake_dataset_fomm")
    os.makedirs(output_base_dir, exist_ok=True)

    print(f"   -> Checking FOMM Data (Target: {output_base_dir})...")
    config_path = os.path.join(fomm_dir, "config/vox-256.yaml")
    checkpoint_path = os.path.join(fomm_dir, "vox-cpk.pth.tar")
    inference_script = os.path.join(fomm_dir, "inference_batch.py")
    
    if not os.path.exists(inference_script):
        print(" Error: inference_batch.py not found!")
        return 0, 0, 1

    all_vid_paths = []
    if dataset_name == 'faceforensics':
        v_root = os.path.join(base_dir, "original_sequences/youtube/c23/videos")
        if os.path.exists(v_root):
             all_vid_paths = [os.path.join(v_root, f) for f in os.listdir(v_root) if f.endswith('.mp4')]
    else:
        for spk in all_speakers:
            path = os.path.join(base_dir, "video", spk) if dataset_name == 'grid' else os.path.join(base_dir, *spk.split('/'))
            if os.path.exists(path):
                for root, _, files in os.walk(path):
                    for f in files:
                        if f.endswith(('.mp4','.avi','.mpg')):
                            all_vid_paths.append(os.path.join(root, f))
    
    if len(all_vid_paths) < 2: return 0, 0, 0

    skipped = 0; generated = 0; errors = 0

    for spk in all_speakers:
        if dataset_name == 'faceforensics':
            user_id = spk
            video_dir = os.path.join(base_dir, "original_sequences/youtube/c23/videos")
            spk_videos = [os.path.join(video_dir, f"{spk}.mp4")]
        elif dataset_name == 'grid':
            video_dir = os.path.join(base_dir, "video", spk)
            user_id = spk
            spk_videos = []
        else:
            video_dir = os.path.join(base_dir, *spk.split('/'))
            user_id = spk.split('/')[-1]
            spk_videos = []

        if dataset_name != 'faceforensics' and os.path.exists(video_dir):
            for root, _, files in os.walk(video_dir):
                for f in files:
                    if f.endswith(('.mp4','.avi','.mpg')):
                        spk_videos.append(os.path.join(root, f))

        speaker_out_dir = os.path.join(output_base_dir, user_id)
        os.makedirs(speaker_out_dir, exist_ok=True)
        
        selected = spk_videos[::step][:videos_per_user]
        
        for source_vid in selected:
            if not os.path.exists(source_vid): continue
            vid_name = os.path.splitext(os.path.basename(source_vid))[0]
            save_name = f"fake_fomm_{vid_name}.avi"
            output_path = os.path.join(speaker_out_dir, save_name)

            if os.path.exists(output_path):
                skipped += 1
                continue

            driving_vid = random.choice(all_vid_paths)
            while driving_vid == source_vid:
                driving_vid = random.choice(all_vid_paths)
            
            command = [
                "python", inference_script,
                "--config", config_path,
                "--checkpoint", checkpoint_path,
                "--source_image", os.path.abspath(source_vid),
                "--driving_video", os.path.abspath(driving_vid),
                "--result_video", os.path.abspath(output_path)
            ]
            
            try:
                subprocess.run(command, capture_output=True, cwd=fomm_dir, text=True, check=True)
                generated += 1
            except subprocess.CalledProcessError:
                errors += 1
                continue
                
    return generated, skipped, errors

def generate_fakes_liveportrait(
    base_dir, 
    lp_dir, 
    dataset_name, 
    videos_per_user, 
    all_speakers, 
    step
):
    output_base_dir = os.path.join(base_dir, "fake_dataset_lp")
    os.makedirs(output_base_dir, exist_ok=True)
    
    inference_script = os.path.join(lp_dir, "inference.py")
    if not os.path.exists(inference_script):
        print(f" Error: LivePortrait inference script not found at {inference_script}")
        return 0, 0, 1

    print(f"   -> LivePortrait Generation (Target: {output_base_dir})...")

    all_vid_paths = []
    if dataset_name == 'faceforensics':
        v_root = os.path.join(base_dir, "original_sequences/youtube/c23/videos")
        if os.path.exists(v_root):
             all_vid_paths = [os.path.join(v_root, f) for f in os.listdir(v_root) if f.endswith('.mp4')]
    else:
        for spk in all_speakers:
            if dataset_name == 'grid':
                path = os.path.join(base_dir, "video", spk) 
            else:
                path = os.path.join(base_dir, *spk.split('/'))
            
            if os.path.exists(path):
                for root, _, files in os.walk(path):
                    for f in files:
                        if f.endswith(('.mp4','.avi','.mpg')):
                            all_vid_paths.append(os.path.join(root, f))

    if not all_vid_paths: 
        print(" Error: No driving videos found.")
        return 0, 0, 0

    skipped = 0; generated = 0; errors = 0

    my_env = os.environ.copy()
    my_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    my_env["PYTHONIOENCODING"] = "utf-8"

    for spk in all_speakers:
        if dataset_name == 'faceforensics':
            user_id = spk
            video_dir = os.path.join(base_dir, "original_sequences/youtube/c23/videos")
            spk_videos = [os.path.join(video_dir, f"{spk}.mp4")]
        elif dataset_name == 'grid':
            video_dir = os.path.join(base_dir, "video", spk)
            user_id = spk
            spk_videos = []
        else:
            video_dir = os.path.join(base_dir, *spk.split('/'))
            user_id = spk.split('/')[-1]
            spk_videos = []

        if dataset_name != 'faceforensics' and os.path.exists(video_dir):
            for root, _, files in os.walk(video_dir):
                for f in files:
                    if f.endswith(('.mp4','.avi','.mpg')):
                        spk_videos.append(os.path.join(root, f))

        speaker_out_dir = os.path.join(output_base_dir, user_id)
        os.makedirs(speaker_out_dir, exist_ok=True)
        
        selected = spk_videos[::step][:videos_per_user]

        for source_vid in selected:
            if not os.path.exists(source_vid): continue
            
            vid_name = os.path.splitext(os.path.basename(source_vid))[0]
            final_out_path = os.path.join(speaker_out_dir, f"fake_lp_{vid_name}.mp4")
            
            if os.path.exists(final_out_path):
                skipped += 1
                continue

            driver = random.choice(all_vid_paths)
            while driver == source_vid: driver = random.choice(all_vid_paths)
            driver_name = os.path.splitext(os.path.basename(driver))[0]

            temp_src_img = os.path.join(lp_dir, "temp_source.jpg")
            cap = cv2.VideoCapture(source_vid)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                errors += 1; continue
            cv2.imwrite(temp_src_img, frame)

            command = [
                "python", "inference.py",
                "--source", os.path.abspath(temp_src_img),
                "--driving", os.path.abspath(driver),
                "--output-dir", os.path.abspath(speaker_out_dir),
                "--flag_do_crop",
                "--flag_pasteback"
            ]
            
            try:
                result = subprocess.run(
                    command, 
                    capture_output=True, 
                    cwd=lp_dir, 
                    text=True, 
                    check=True,
                    env=my_env,
                    encoding='utf-8',
                    errors='replace'
                )
                
                expected_pattern = f"temp_source--{driver_name}*.mp4"
                search_path = os.path.join(speaker_out_dir, expected_pattern)
                found_files = glob.glob(search_path)
                
                if not found_files:
                     all_mp4s = glob.glob(os.path.join(speaker_out_dir, "*.mp4"))
                     if all_mp4s:
                         candidates = [f for f in all_mp4s if "fake_lp_" not in os.path.basename(f)]
                         if candidates:
                             found_files = [max(candidates, key=os.path.getctime)]

                if found_files:
                    generated_file = found_files[0]
                    if os.path.exists(final_out_path): os.remove(final_out_path)
                    try:
                        os.rename(generated_file, final_out_path)
                        generated += 1
                    except OSError:
                        errors += 1
                else:
                    print(f"\n[Error] LP finished but no file found in {speaker_out_dir}")
                    print(f"Stdout: {result.stdout[:300]}...") 
                    errors += 1

            except subprocess.CalledProcessError as e:
                errors += 1
                print(f"\n[CRASH] LivePortrait Failed for {vid_name}!")
                print(f"Error Message: {e.stderr.strip()}") 
                continue

    return generated, skipped, errors

def generate_fake_videos(base_dir, wav2lip_dir, fomm_dir, liveportrait_dir, dataset_name, videos_per_user, limit_speakers, tool, predictor_path=None, step=1):
    
    if dataset_name == 'faceforensics':
        v_root = os.path.join(base_dir, "original_sequences/youtube/c23/videos")
        if os.path.exists(v_root):
            all_speakers = sorted([os.path.splitext(f)[0] for f in os.listdir(v_root) if f.endswith('.mp4')])
        else:
            print(f" Error: FF++ folder not found at {v_root}"); return
    elif dataset_name == 'grid':
        video_root = os.path.join(base_dir, "video")
        if os.path.exists(video_root):
            all_speakers = sorted([d for d in os.listdir(video_root) if d.startswith('s')])
        else:
            print(f" Error: GRID video folder not found at {video_root}"); return
    else:
        sources = ['idiap', 'unis']
        all_speakers = []
        for source in sources:
            path = os.path.join(base_dir, source)
            if os.path.exists(path):
                spks = [f"{source}/{s}" for s in os.listdir(path) if os.path.isdir(os.path.join(path, s))]
                all_speakers.extend(spks)

    if limit_speakers:
        limit_set = set(s.replace('\\', '/') for s in limit_speakers)
        all_speakers = [s for s in all_speakers if s.replace('\\', '/') in limit_set]

    print(f"   -> Processing {len(all_speakers)} speakers...")
    
    gen_w, skip_w, err_w = 0, 0, 0
    gen_f, skip_f, err_f = 0, 0, 0
    gen_lp, skip_lp, err_lp = 0, 0, 0

    if tool in ['wav2lip', 'both']:
        g, s, e = generate_fakes_wav2lip(
            base_dir, wav2lip_dir, dataset_name, 
            videos_per_user, all_speakers, step, predictor_path
        )
        gen_w, skip_w, err_w = g, s, e

    if tool in ['fomm', 'both']:
        g, s, e = generate_fakes_fomm(
            base_dir, fomm_dir, dataset_name, 
            videos_per_user, all_speakers, step
        )
        gen_f, skip_f, err_f = g, s, e
        
    if tool in ['liveportrait', 'both']:
        g, s, e = generate_fakes_liveportrait(
            base_dir, liveportrait_dir, dataset_name,
            videos_per_user, all_speakers, step
        )
        gen_lp, skip_lp, err_lp = g, s, e

    print(f" Fakes Generation Complete.")
    if tool == 'wav2lip':
        print(f"   Wav2Lip: New {gen_w}, Skipped {skip_w}, Errors {err_w}")
    elif tool == 'fomm':
        print(f"   FOMM   : New {gen_f}, Skipped {skip_f}, Errors {err_f}")
    elif tool == 'liveportrait':
        print(f"   LiveP  : New {gen_lp}, Skipped {skip_lp}, Errors {err_lp}")
    else:
        print(f"   Wav2Lip: {gen_w} | FOMM: {gen_f} | LiveP: {gen_lp}")