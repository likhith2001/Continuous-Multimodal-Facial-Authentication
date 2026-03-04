import cv2
import torch
import numpy as np
import os
import sys
from collections import deque
import torch.nn.functional as F

from src.tune_optuna import OpticalFlowModel
from src.extract_optical_flow import extract_region

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FRAMES = 30  

class RealTimeVerifier:
    def __init__(self, lip_model_path, eye_model_path):
        print(f"   [Verifier] Loading Models on {DEVICE}...")
        self.lip_model = self._load_model(lip_model_path)
        self.eye_model = self._load_model(eye_model_path)
        
        self.lip_buffer = deque(maxlen=MAX_FRAMES)
        self.eye_buffer = deque(maxlen=MAX_FRAMES)
        
        self.injection_active = False
        self.fake_video_cap = None
        
        self.consecutive_fake_frames = 0
        self.frame_history = deque(maxlen=5)

    def _load_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        checkpoint = torch.load(path, map_location=DEVICE)
        dropout = checkpoint.get('hyperparams', {}).get('dropout', 0.5)
        model = OpticalFlowModel(dropout=dropout).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def start_injection(self, fake_video_path):
        if not os.path.exists(fake_video_path):
            print(f"Error: Fake video not found at {fake_video_path}")
            return
        self.fake_video_cap = cv2.VideoCapture(fake_video_path)
        self.injection_active = True
        print(f"   [Verifier]  INJECTION STARTED: {os.path.basename(fake_video_path)}")

    def stop_injection(self):
        self.injection_active = False
        if self.fake_video_cap:
            self.fake_video_cap.release()
            self.fake_video_cap = None
        print("   [Verifier]  INJECTION STOPPED. Returning to Webcam.")

    def get_frame(self, webcam_frame):
        if self.injection_active and self.fake_video_cap:
            ret, fake_frame = self.fake_video_cap.read()
            if not ret:
                self.fake_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, fake_frame = self.fake_video_cap.read()
            return fake_frame, True 
        return webcam_frame, False

    def process_stream(self, frame, dataset_name='mobio'):
        lip_crop = extract_region(frame, mode='lip', dataset_name=dataset_name)
        eye_crop = extract_region(frame, mode='eye', dataset_name=dataset_name)

        if lip_crop is None or eye_crop is None:
            return None

        self.lip_buffer.append(cv2.cvtColor(lip_crop, cv2.COLOR_BGR2GRAY))
        self.eye_buffer.append(cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY))

        if len(self.lip_buffer) < 5: 
            return {"status": "buffering", "progress": len(self.lip_buffer)}

        lip_flow = self._compute_dense_flow(list(self.lip_buffer))
        eye_flow = self._compute_dense_flow(list(self.eye_buffer))

        with torch.no_grad():
            lip_out = self.lip_model(lip_flow)
            lip_prob = F.softmax(lip_out, dim=1)[0][1].item() 

            eye_out = self.eye_model(eye_flow)
            eye_prob = F.softmax(eye_out, dim=1)[0][1].item() 

        combined_score = (lip_prob * 0.4) + (eye_prob * 0.6)
        
        SUSPICION_THRESHOLD = 0.50

        if combined_score > SUSPICION_THRESHOLD:
            self.consecutive_fake_frames += 1
        else:
            self.consecutive_fake_frames = 0

        if self.consecutive_fake_frames >= 2:
            verdict = "FAKE"
            trust_score = 1.0 - combined_score
        else:
            verdict = "REAL"
            trust_score = 1.0
            
        print(f"   [DEBUG] Score: {combined_score:.2f} | Count: {self.consecutive_fake_frames} | Verdict: {verdict}")

        return {
            "status": "active",
            "lip_prob_fake": lip_prob,
            "eye_prob_fake": eye_prob,
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
             return torch.zeros(1, 2, 10, 64, 64).to(DEVICE)

        tensor = torch.from_numpy(flow_stack).float().permute(3, 0, 1, 2)
        
        min_v, max_v = tensor.min(), tensor.max()
        if max_v - min_v > 0:
            tensor = (tensor - min_v) / (max_v - min_v)
            
        return tensor.unsqueeze(0).to(DEVICE)