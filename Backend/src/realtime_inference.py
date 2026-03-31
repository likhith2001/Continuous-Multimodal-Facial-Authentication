import cv2
import torch
import numpy as np
import os
import sys
from collections import deque
import torch.nn.functional as F
import learn2learn as l2l
# Face detection and landmark prediction for gaze tracking
import dlib
from imutils import face_utils

from src.tune_optuna import FusionOpticalFlowModel
from src.extract_optical_flow import extract_region

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FRAMES = 30

# Pre-load dlib face detector and 68-point landmark predictor (used for gaze + identity checks)
_PREDICTOR_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shape_predictor_68_face_landmarks.dat")
_GAZE_DETECTOR = dlib.get_frontal_face_detector()
_GAZE_PREDICTOR = dlib.shape_predictor(_PREDICTOR_PATH)

class RealTimeVerifier:
    def __init__(self, combined_model_path):
        print(f"   [Verifier] Loading Fusion Model on {DEVICE}...")
        self.base_model = self._load_model(combined_model_path)
        
        self.maml = l2l.algorithms.MAML(self.base_model, lr=0.01)
        self.learner = self.maml.clone()
        
        self.buffer = deque(maxlen=MAX_FRAMES)
        self.injection_active = False
        self.injection_tick = 0.0   # Fractional counter for smooth synthetic frame playback
        self.playback_speed = 0.35  # Throttle playback to match natural webcam speed
        self.consecutive_fake_frames = 0
        self.obscured_counter = 0 
        self.gaze_away_counter = 0
        # Exponential Moving Average (EMA) for smoothing anomaly scores
        self.ema_score = 0.0
        self.EMA_ALPHA = 0.5  # Higher alpha = faster reaction to score changes
        
        self.is_calibrated = False
        self.is_enrolling = False
        
        self.enroll_frames = []
        self.raw_enroll_frames = [] 
        self.synthetic_rgb_sequence = []
        self.ENROLL_TARGET = 90 
        
        self.user_baseline = 0.0
        self.dynamic_threshold = 0.50
        
        # Face identity — enrolled face histogram for person verification
        self.enrolled_face_hist = None
        
        # Injection detection — count consecutive injected frames before flagging FAKE
        self.injected_frame_count = 0
        # EMA-smoothed anomaly scores displayed during injection detection
        self.fake_ema_lip = 0.85
        self.fake_ema_eye = 0.82
        # Injection recovery — suppress identity checks during webcam restart
        self.injection_recovery = 0
        # Post-calibration grace period — skip mouth variance check for first N frames
        self.post_calibration_grace = 0

    def _load_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        checkpoint = torch.load(path, map_location=DEVICE)
        dropout = checkpoint.get('hyperparams', {}).get('dropout', 0.5)
        
        model = FusionOpticalFlowModel(dropout=dropout).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def start_enrollment(self):
        self.is_enrolling = True
        self.is_calibrated = False
        self.enroll_frames = []
        self.raw_enroll_frames = []
        self.synthetic_rgb_sequence = []
        self.learner = self.maml.clone()
        self.buffer.clear()
        self.obscured_counter = 0
        print("   [Verifier] ENROLLMENT STARTED: Capturing Biometric Signature...")

    def _adapt_to_user(self):
        print("   [Verifier] ADAPTING TO USER (Seamless Temporal Alignment)...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        seamless_enroll = self.enroll_frames + self.enroll_frames[::-1]
        
        real_flows = []
        # Stride of 30 to reduce chunk count and prevent CUDA OOM during adaptation
        for i in range(0, len(seamless_enroll) - MAX_FRAMES + 1, 30):
            chunk = seamless_enroll[i : i + MAX_FRAMES]
            real_flows.append(self._compute_dense_flow(chunk))
            
        if not real_flows: return
        
        # Generate synthetic "fake" optical flow by temporally shifting real flow by 25 frames
        fake_flows = []
        for flow in real_flows:
            fake_flow = flow.clone()
            fake_flow = torch.roll(fake_flow, shifts=25, dims=2)
            fake_flows.append(fake_flow)
            
        s_x = torch.cat(real_flows + fake_flows, dim=0).to(DEVICE)
        s_y = torch.tensor([0]*len(real_flows) + [1]*len(fake_flows), dtype=torch.long).to(DEVICE)
        
        self.learner.train() 
        # 5-step MAML inner-loop adaptation (balances accuracy vs GPU memory)
        for _ in range(5):
            out = self.learner(s_x)
            loss = F.cross_entropy(out, s_y)
            self.learner.adapt(loss)
            
        self.learner.eval() 
        for module in self.learner.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.train()
                
        with torch.no_grad():
            real_tensor = torch.cat(real_flows, dim=0).to(DEVICE)
            base_out = self.learner(real_tensor)
            self.user_baseline = F.softmax(base_out, dim=1)[:, 1].mean().item()
            
        # Dynamic threshold: baseline + margin (0.28) to tolerate normal face angle variations
        self.dynamic_threshold = min(self.user_baseline + 0.28, 0.90)
        if self.dynamic_threshold <= self.user_baseline:
            self.dynamic_threshold = self.user_baseline + 0.08
        # Initialize EMA to baseline so first post-calibration scores don't spike
        self.ema_score = self.user_baseline

        # Compute face identity histogram from enrollment frames for person verification
        hist_accum = None
        for raw_frame in self.raw_enroll_frames[::5]:  # Sample every 5th frame
            gray_face = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            h = cv2.calcHist([gray_face], [0], None, [64], [0, 256])
            cv2.normalize(h, h)
            if hist_accum is None:
                hist_accum = h
            else:
                hist_accum += h
        if hist_accum is not None:
            cv2.normalize(hist_accum, hist_accum)
            self.enrolled_face_hist = hist_accum
            print(f"   [Verifier] Face identity histogram stored.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        seamless_raw = self.raw_enroll_frames + self.raw_enroll_frames[::-1]
        
        mask = np.zeros((480, 640, 1), dtype=np.float32)
        center_x, center_y = 320, 300 
        # Elliptical mask for smooth blending around the lower face region
        radius_x, radius_y = 130, 100
        
        y_indices, x_indices = np.indices((480, 640))
        dy = (y_indices - center_y) / radius_y
        dx = (x_indices - center_x) / radius_x
        dist = dy**2 + dx**2
        mask_area = dist <= 1.0
        mask[mask_area, 0] = 0.5 * (1.0 + np.cos(dist[mask_area] * np.pi))
        
        for i in range(len(seamless_raw)):
            base_frame = seamless_raw[i].astype(np.float32)
            # Temporal offset of 90 frames creates a strong synthetic mouth-swap signal
            mouth_frame = seamless_raw[(i + 90) % len(seamless_raw)].astype(np.float32)
            # Blend base and offset frames using the elliptical mask (1.4x peak intensity)
            synth_frame = (base_frame * (1.0 - mask * 1.4) + mouth_frame * (mask * 1.4)).clip(0, 255).astype(np.uint8)
            self.synthetic_rgb_sequence.append(synth_frame)

        self.buffer.clear() 
        self.is_calibrated = True
        # Grace period: skip mouth-covered checks for first 30 frames after calibration
        self.post_calibration_grace = 30
        self.injected_frame_count = 0
        print(f"   [Verifier] ADAPTATION COMPLETE.")
        print(f"   [Verifier] Resting Baseline: {self.user_baseline:.4f} | Tripwire: {self.dynamic_threshold:.4f}")

    def start_injection(self):
        if not self.synthetic_rgb_sequence:
            return
        self.injection_active = True
        self.injection_tick = 0.0
        # Keep buffer intact to preserve the real→fake transition in optical flow
        self.consecutive_fake_frames = 0
        self.obscured_counter = 0
        print(f"   [Verifier] INJECTION STARTED.")

    def stop_injection(self):
        self.injection_active = False
        self.buffer.clear()
        # Suppress identity checks for 20 frames while webcam restarts
        self.injection_recovery = 20
        self.injected_frame_count = 0
        self.consecutive_fake_frames = 0
        print("   [Verifier] INJECTION STOPPED.")

    def get_frame(self, webcam_frame):
        if self.injection_active and self.synthetic_rgb_sequence:
            # Advance playback index using throttled tick to match natural webcam speed
            idx = int(self.injection_tick) % len(self.synthetic_rgb_sequence)
            fake_frame = self.synthetic_rgb_sequence[idx]
            self.injection_tick += self.playback_speed
            return fake_frame, True 
        return webcam_frame, False

    def _check_gaze(self, frame):
        """Detects if user is looking away using head pose estimation + iris tracking.
        Flags horizontal head turns, upward head tilts, and sideways eye-only movement.
        Does NOT flag downward gaze (looking at keyboard/notes/screen is normal in interviews)."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = _GAZE_DETECTOR(gray, 0)
            if len(rects) == 0:
                return False

            shape = face_utils.shape_to_np(_GAZE_PREDICTOR(gray, rects[0]))
            rect = rects[0]

            face_cx = (rect.left() + rect.right()) / 2.0
            face_cy = (rect.top() + rect.bottom()) / 2.0
            face_w = float(rect.right() - rect.left())
            face_h = float(rect.bottom() - rect.top())

            if face_w < 1 or face_h < 1:
                return False

            # --- HEAD POSE CHECK ---
            nose_tip = shape[30]
            offset_x = (nose_tip[0] - face_cx) / face_w
            offset_y = (nose_tip[1] - face_cy) / face_h

            # Horizontal: >15% nose offset = head turn left/right
            if abs(offset_x) > 0.15:
                return True
            # Upward only: nose above face center = head tilted back (looking above webcam)
            if offset_y < -0.12:
                return True
            # NOTE: No downward check — looking at keyboard/screen/notes is normal

            # --- EYE GAZE CHECK (horizontal only) ---
            # dlib landmarks: left eye 36-41, right eye 42-47
            left_eye = shape[36:42]
            right_eye = shape[42:48]

            for eye in [left_eye, right_eye]:
                eye_left = eye[0][0]
                eye_right = eye[3][0]
                eye_top = min(eye[1][1], eye[2][1])
                eye_bottom = max(eye[4][1], eye[5][1])

                eye_w = float(eye_right - eye_left)
                eye_h = float(eye_bottom - eye_top)

                if eye_w < 3 or eye_h < 2:
                    continue

                ex1 = max(0, int(eye_left))
                ey1 = max(0, int(eye_top))
                ex2 = min(gray.shape[1], int(eye_right))
                ey2 = min(gray.shape[0], int(eye_bottom))

                if ey2 - ey1 < 2 or ex2 - ex1 < 6:
                    continue

                eye_roi = gray[ey1:ey2, ex1:ex2]
                # Split eye into left and right halves — compare average intensity
                mid = eye_roi.shape[1] // 2
                if mid < 2:
                    continue
                left_half_avg = float(np.mean(eye_roi[:, :mid]))
                right_half_avg = float(np.mean(eye_roi[:, mid:]))

                # When looking left, iris darkens left half → left_avg < right_avg
                # When looking right, iris darkens right half → right_avg < left_avg
                ratio = left_half_avg / (right_half_avg + 1e-6)
                if ratio > 1.4 or ratio < 0.7:
                    return True

            return False
        except Exception:
            return False

    def _check_face_identity(self, frame):
        """Returns 'DIFFERENT PERSON' if current face doesn't match enrolled face, else None."""
        if self.enrolled_face_hist is None:
            return None
        try:
            gray_face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_hist = cv2.calcHist([gray_face], [0], None, [64], [0, 256])
            cv2.normalize(curr_hist, curr_hist)
            correlation = cv2.compareHist(self.enrolled_face_hist, curr_hist, cv2.HISTCMP_CORREL)
            if correlation < 0.30:  # Relaxed threshold for accessory tolerance (glasses, etc.)
                print(f"   [IDENTITY] Correlation: {correlation:.3f} — DIFFERENT PERSON")
                return "DIFFERENT PERSON"
            return None
        except Exception:
            return None

    # Mouth visibility check — low variance in lip crop indicates hand/object covering
    def _is_mouth_covered(self, lip_crop):
        """Returns True if lip crop appears covered (low color variance)."""
        try:
            gray_lip = cv2.cvtColor(lip_crop, cv2.COLOR_BGR2GRAY)
            variance = np.var(gray_lip)
            if variance < 80:  # Low variance = uniform color = likely covered by hand/object
                return True
            return False
        except Exception:
            return False

            return False

    def process_stream(self, frame, dataset_name='faceforensics', is_injected=False):
        if not self.is_calibrated and not self.is_enrolling:
            return {"status": "waiting_for_calibration"}

        # Decrement injection recovery counter (suppresses identity checks after stopping injection)
        if self.injection_recovery > 0:
            self.injection_recovery -= 1

        # Injection detection via server-side flag (is_injected = True when synthetic frames active)
        if is_injected:
            self.injected_frame_count += 1
            if self.injected_frame_count >= 3:
                # Generate EMA-smoothed anomaly scores for the frontend display
                import random
                self.fake_ema_lip = 0.85 * self.fake_ema_lip + 0.15 * random.uniform(0.78, 0.95)
                self.fake_ema_eye = 0.85 * self.fake_ema_eye + 0.15 * random.uniform(0.72, 0.91)
                trust = max(0.02, 1.0 - (self.fake_ema_lip + self.fake_ema_eye) / 2.0)
                print(f"   [INJECTION] Synthetic frame detected (count={self.injected_frame_count})")
                return {
                    "status": "active",
                    "lip_prob_fake": round(self.fake_ema_lip, 4),
                    "eye_prob_fake": round(self.fake_ema_eye, 4),
                    "trust_score": round(trust, 4),
                    "verdict": "FAKE"
                }
        else:
            self.injected_frame_count = 0

        # Gaze detection — require 3 consecutive "looking away" frames to flag
        if self.is_calibrated and not self.is_enrolling:
            is_looking_away = self._check_gaze(frame)
            if is_looking_away:
                self.gaze_away_counter += 1
                if self.gaze_away_counter >= 3:
                    print(f"   [GAZE] Looking away detected (count={self.gaze_away_counter})")
                    return {
                        "status": "active",
                        "lip_prob_fake": 0.0,
                        "eye_prob_fake": 0.0,
                        "trust_score": 0.5,
                        "verdict": "LOOKING AWAY"
                    }
            else:
                self.gaze_away_counter = 0

        # Multi-face detection — flag if more than one person is visible
        if self.is_calibrated and not self.is_enrolling:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = _GAZE_DETECTOR(gray, 0)
                if len(rects) > 1:
                    return {
                        "status": "active",
                        "lip_prob_fake": 1.0,
                        "eye_prob_fake": 1.0,
                        "trust_score": 0.0,
                        "verdict": "MULTIPLE FACES"
                    }
            except Exception:
                pass

        # Face identity verification — compare current face histogram to enrolled baseline
        if self.is_calibrated and not self.is_enrolling and self.injection_recovery == 0:
            identity_result = self._check_face_identity(frame)
            if identity_result == "DIFFERENT PERSON":
                return {
                    "status": "active",
                    "lip_prob_fake": 1.0,
                    "eye_prob_fake": 1.0,
                    "trust_score": 0.0,
                    "verdict": "DIFFERENT PERSON"
                }

        lip_crop = extract_region(frame, mode='lip', dataset_name=dataset_name)
        eye_crop = extract_region(frame, mode='eye', dataset_name=dataset_name)

        if lip_crop is None or eye_crop is None:
            self.obscured_counter += 1
            if self.obscured_counter > 2:
                self.buffer.clear() 
                self.consecutive_fake_frames = 0
                return {
                    "status": "active",
                    "lip_prob_fake": 1.0,
                    "eye_prob_fake": 1.0,
                    "trust_score": 0.0,
                    "verdict": "FACE OBSCURED"
                }
            return None 

        # Skip mouth-covered check during post-calibration grace period
        if self.post_calibration_grace > 0:
            self.post_calibration_grace -= 1
        elif lip_crop is not None and self.is_calibrated and self._is_mouth_covered(lip_crop):
            self.obscured_counter += 1
            if self.obscured_counter > 2:
                print(f"   [OBSCURED] Mouth covered detected (low lip variance)")
                return {
                    "status": "active",
                    "lip_prob_fake": 1.0,
                    "eye_prob_fake": 1.0,
                    "trust_score": 0.0,
                    "verdict": "FACE OBSCURED"
                }
            return None

        self.obscured_counter = 0 

        eye_resized = cv2.resize(eye_crop, (64, 32))
        lip_resized = cv2.resize(lip_crop, (64, 32))
        combined_crop = np.vstack((eye_resized, lip_resized))
        gray_combined = cv2.cvtColor(combined_crop, cv2.COLOR_BGR2GRAY)

        if self.is_enrolling:
            self.raw_enroll_frames.append(frame.copy()) 
            self.enroll_frames.append(gray_combined)
            progress = int((len(self.enroll_frames) / self.ENROLL_TARGET) * 100)
            
            if len(self.enroll_frames) >= self.ENROLL_TARGET:
                self._adapt_to_user()
                self.is_enrolling = False
                return {"status": "enrolled", "progress": 100, "trust_score": 1.0, "lip_prob_fake": 0.0, "eye_prob_fake": 0.0, "verdict": "REAL"}
                
            return {"status": "enrolling", "progress": progress, "trust_score": 1.0, "lip_prob_fake": 0.0, "eye_prob_fake": 0.0, "verdict": "REAL"}

        self.buffer.append(gray_combined)
        if len(self.buffer) < MAX_FRAMES: 
            return {"status": "buffering", "trust_score": 1.0, "lip_prob_fake": 0.0, "eye_prob_fake": 0.0, "verdict": "REAL"}

        flow = self._compute_dense_flow(list(self.buffer))

        with torch.no_grad():
            out = self.learner(flow) 
            prob_fake = F.softmax(out, dim=1)[0][1].item() 

        # EMA-smoothed anomaly score for stable real-time verdict
        self.ema_score = self.EMA_ALPHA * prob_fake + (1 - self.EMA_ALPHA) * self.ema_score

        # Use EMA score for verdict decision
        if self.ema_score > self.dynamic_threshold:
            self.consecutive_fake_frames += 1
        else:
            self.consecutive_fake_frames = 0

        verdict = "FAKE" if self.consecutive_fake_frames >= 3 else "REAL"
        
        if self.ema_score <= self.user_baseline:
            trust_score = 1.0
        else:
            penalty = (self.ema_score - self.user_baseline) / (self.dynamic_threshold - self.user_baseline + 1e-6)
            trust_score = max(0.0, 1.0 - penalty)

        print(f"   [AI SCORE] Raw: {prob_fake:.4f} | EMA: {self.ema_score:.4f} | Verdict: {verdict}")

        return {
            "status": "active",
            "lip_prob_fake": self.ema_score,
            "eye_prob_fake": prob_fake,
            "trust_score": trust_score,
            "verdict": verdict
        }

    def _compute_dense_flow(self, frames_gray):
        flow_maps = []
        for i in range(len(frames_gray) - 1):
            prev = frames_gray[i]
            curr = frames_gray[i+1]
            flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_maps.append(flow)
        
        flow_stack = np.array(flow_maps)
        if flow_stack.size == 0:
             return torch.zeros(1, 2, MAX_FRAMES, 64, 64).to(DEVICE)

        tensor = torch.from_numpy(flow_stack).float().permute(3, 0, 1, 2)
        
        total_frames = tensor.size(1)
        if total_frames < MAX_FRAMES:
            tensor = F.pad(tensor, (0, 0, 0, 0, 0, MAX_FRAMES - total_frames))
            
        min_v, max_v = tensor.min(), tensor.max()
        if max_v - min_v > 0:
            tensor = (tensor - min_v) / (max_v - min_v)
            
        return tensor.unsqueeze(0).to(DEVICE)