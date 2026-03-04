import os
import cv2
import dlib
import numpy as np
from imutils import face_utils
from tqdm import tqdm
import sys

current_script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_script_dir)
PREDICTOR_PATH = os.path.join(backend_dir, "shape_predictor_68_face_landmarks.dat")

if not os.path.exists(PREDICTOR_PATH):
    if os.path.exists("shape_predictor_68_face_landmarks.dat"):
        PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
    else:
        print(f"\n CRITICAL ERROR: Shape Predictor not found at: {PREDICTOR_PATH}")
        sys.exit(1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
haar_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(haar_path)

extraction_errors = 0

def stabilize_face(frame, shape, desired_face_width=256):
    left_eye = shape[36:42]
    right_eye = shape[42:48]
    left_eye_center = left_eye.mean(axis=0).astype("int")
    right_eye_center = right_eye.mean(axis=0).astype("int")

    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))

    desired_left_eye_pos = (0.35, 0.35)
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desired_dist = (1.0 - 2 * desired_left_eye_pos[0]) * desired_face_width
    scale = desired_dist / dist

    cx = int((left_eye_center[0] + right_eye_center[0]) // 2)
    cy = int((left_eye_center[1] + right_eye_center[1]) // 2)
    eyes_center = (cx, cy)

    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    tX = desired_face_width * 0.5
    tY = desired_face_width * desired_left_eye_pos[1]
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])

    return cv2.warpAffine(frame, M, (desired_face_width, desired_face_width), flags=cv2.INTER_CUBIC)

def get_simple_crop(frame, points, pad=5):
    (x, y, w, h) = cv2.boundingRect(points)
    return frame[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]

def detect_face_robust(frame, dataset_name):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    upsample = 1 if dataset_name == 'mobio' else 0
    rects = detector(gray, upsample)
    
    if len(rects) > 0:
        return rects[0], gray, frame

    if dataset_name == 'mobio':
        for rot_code in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            frame_rot = cv2.rotate(frame, rot_code)
            gray_rot = cv2.cvtColor(frame_rot, cv2.COLOR_BGR2GRAY)
            rects_rot = detector(gray_rot, upsample)
            if len(rects_rot) > 0:
                return rects_rot[0], gray_rot, frame_rot

    gray_eq = cv2.equalizeHist(gray)
    rects_eq = detector(gray_eq, upsample)
    if len(rects_eq) > 0:
        return rects_eq[0], gray_eq, frame

    faces_haar = face_cascade.detectMultiScale(gray, 1.1, 3)
    if len(faces_haar) > 0:
        faces_haar = sorted(faces_haar, key=lambda x: x[2] * x[3], reverse=True)
        (x, y, w, h) = faces_haar[0]
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        return dlib_rect, gray, frame

    return None, None, None

def extract_region(frame, mode='lip', dataset_name='mobio', target_size=(64, 64)):
    global extraction_errors
    try:
        result = detect_face_robust(frame, dataset_name)
        if result[0] is None: return None
        rect, gray, processed_frame = result
        shape = face_utils.shape_to_np(predictor(gray, rect))

        if dataset_name == 'mobio':
            aligned_face = stabilize_face(processed_frame, shape)
            if mode == 'lip':
                crop = aligned_face[130:230, 70:186]
                return cv2.resize(crop, target_size)
            elif mode == 'eye':
                crop = aligned_face[60:110, 50:206]
                return cv2.resize(crop, target_size)
            elif mode == 'combined':
                eyes = aligned_face[60:110, 50:206]
                mouth = aligned_face[130:230, 70:186]
                return np.vstack([cv2.resize(eyes,(64,32)), cv2.resize(mouth,(64,32))])
        else:
            if mode == 'lip':
                crop = get_simple_crop(processed_frame, shape[48:68])
                if crop.size == 0: return None
                return cv2.resize(crop, target_size)
            elif mode == 'eye':
                r_eye = get_simple_crop(processed_frame, shape[36:42])
                l_eye = get_simple_crop(processed_frame, shape[42:48])
                if r_eye.size==0 or l_eye.size==0: return None
                combined = np.hstack([cv2.resize(r_eye,(32,32)), cv2.resize(l_eye,(32,32))])
                return cv2.resize(combined, target_size)
            elif mode == 'combined':
                r_eye = get_simple_crop(processed_frame, shape[36:42])
                l_eye = get_simple_crop(processed_frame, shape[42:48])
                mouth = get_simple_crop(processed_frame, shape[48:68])
                if r_eye.size==0 or l_eye.size==0 or mouth.size==0: return None
                eyes = np.hstack([cv2.resize(r_eye,(32,32)), cv2.resize(l_eye,(32,32))])
                return np.vstack([eyes, cv2.resize(mouth,(64,32))])

    except Exception as e:
        if extraction_errors < 5:
            print(f"\n Extraction Error: {str(e)}")
            extraction_errors += 1
        return None
    return None

def compute_optical_flow(video_path, output_folder, mode, dataset_name, filename_prefix=""):
    vid_name = os.path.splitext(os.path.basename(video_path))[0]
    
    if dataset_name == 'mobio':
        if vid_name.startswith('fake_'):
             save_name = f"{filename_prefix}{vid_name}.npy"
        else:
             parent = os.path.basename(os.path.dirname(video_path))
             save_name = f"{filename_prefix}{parent}_{vid_name}.npy"
    else:
        save_name = f"{filename_prefix}{vid_name}.npy"

    output_path = os.path.join(output_folder, save_name)
    
    if os.path.exists(output_path): return

    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        if not ret: return

        prev = extract_region(first_frame, mode, dataset_name)
        if prev is None:
            for _ in range(10):
                ret, first_frame = cap.read()
                if ret:
                    prev = extract_region(first_frame, mode, dataset_name)
                    if prev is not None: break
            if prev is None: return
            
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        flow_maps = []

        while True:
            ret, frame = cap.read()
            if not ret: break
            curr = extract_region(frame, mode, dataset_name)
            if curr is None: continue
            
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_maps.append(flow)
            prev_gray = curr_gray.copy()

        if len(flow_maps) > 10:
            final_flow = np.array(flow_maps)
            
            if dataset_name == 'faceforensics' and 'fake_' not in save_name:
                mag = np.sqrt(final_flow[..., 0]**2 + final_flow[..., 1]**2)
                mean_mag = np.mean(mag)
                if mean_mag < 0.5:
                    return

            np.save(output_path, final_flow)
            
    except Exception:
        pass
    finally:
        if cap: cap.release()

def process_dataset_mode(base_video_dir, base_flow_dir, mode, dataset_name, limit_speakers_paths=None, max_videos=None, is_fake=False, filename_prefix=""):
    if dataset_name == 'faceforensics' and not is_fake:
         if not os.path.exists(base_video_dir): return
         speakers = sorted([os.path.splitext(f)[0] for f in os.listdir(base_video_dir) if f.endswith('.mp4')])
    
    else:
        if not os.path.exists(base_video_dir):
            print(f"   [Warning] Folder {base_video_dir} does not exist. Skipping.")
            return
        speakers = sorted(os.listdir(base_video_dir))

    if limit_speakers_paths:
        allowed = [s.replace('\\','/').split('/')[-1] for s in limit_speakers_paths]
        speakers = [s for s in speakers if s in allowed]

    total_videos_found = 0
    
    for spk_id in tqdm(speakers, desc=f"Processing {dataset_name.upper()} ({mode})"):
        video_files = []
        
        if dataset_name == 'faceforensics' and not is_fake:
            vid_path = os.path.join(base_video_dir, spk_id + ".mp4")
            if os.path.exists(vid_path): video_files.append(vid_path)
        else:
            spk_dir = os.path.join(base_video_dir, spk_id)
            if dataset_name == 'grid' and not is_fake:
                 nested = os.path.join(spk_dir, spk_id)
                 spk_dir = nested if os.path.exists(nested) else spk_dir
            
            if os.path.exists(spk_dir):
                for r, _, f in os.walk(spk_dir):
                    for file in f:
                         if file.endswith(('.mp4','.avi','.mpg')): video_files.append(os.path.join(r, file))

        video_files.sort()
        if max_videos: video_files = video_files[:max_videos]
        total_videos_found += len(video_files)
        
        out_spk_dir = os.path.join(base_flow_dir, spk_id)
        os.makedirs(out_spk_dir, exist_ok=True)
        
        for vid in video_files:
            compute_optical_flow(vid, out_spk_dir, mode, dataset_name, filename_prefix)
    
    files_created = sum([len(files) for r, d, files in os.walk(base_flow_dir)])
    print(f"   -> [Debug] Input Videos: {total_videos_found} | Extracted Flow Files: {files_created}")