import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import base64
import numpy as np
import uvicorn
import threading
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.realtime_inference import RealTimeVerifier

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LIP_MODEL = "saved_models/mobio_model_lip_fomm_prod.pth"
EYE_MODEL = "saved_models/mobio_model_eye_fomm_prod.pth"
ATTACK_VIDEO = "data/my_videos/my_deepfake.avi"

verifier = None
latest_frame_to_process = None
frame_lock = threading.Lock()
current_metrics = {
    "status": "waiting", 
    "trust_score": 1.0, 
    "verdict": "REAL", 
    "lip_prob_fake": 0.0, 
    "eye_prob_fake": 0.0
}

def ai_inference_loop():
    global latest_frame_to_process, current_metrics
    
    print("   [Background]  AI Worker Thread Started")
    
    while True:
        img_input = None
        
        with frame_lock:
            if latest_frame_to_process is not None:
                img_input = latest_frame_to_process.copy()
                latest_frame_to_process = None

        if img_input is not None and verifier is not None:
            try:
                result = verifier.process_stream(img_input, dataset_name='mobio')
                if result:
                    current_metrics = result
                    print(f"   [AI SCORE] Lip: {result['lip_prob_fake']:.4f} | Eye: {result['eye_prob_fake']:.4f} | Verdict: {result['verdict']}")
            except Exception as e:
                print(f"   [Background Error] {e}")
        
        time.sleep(0.05)

@app.on_event("startup")
async def startup_event():
    global verifier
    print("\n SERVER STARTING: Loading AI Models...")
    try:
        verifier = RealTimeVerifier(LIP_MODEL, EYE_MODEL)
        print(" Models Loaded! Starting Background Worker...\n")
        
        t = threading.Thread(target=ai_inference_loop, daemon=True)
        t.start()
        
    except Exception as e:
        print(f" Error Loading Models: {e}")

@app.websocket("/ws/video")
async def video_endpoint(websocket: WebSocket):
    global latest_frame_to_process
    
    await websocket.accept()
    print(" Client Connected (Turbo Mode)")

    try:
        while True:
            data = await websocket.receive_text()
            header, encoded = data.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None: continue

            display_frame = frame
            is_injected = False
            
            if verifier:
                display_frame, is_injected = verifier.get_frame(frame)

            with frame_lock:
                latest_frame_to_process = display_frame

            _, buffer = cv2.imencode('.jpg', display_frame)
            b64_frame = base64.b64encode(buffer).decode('utf-8')
            
            response = {
                "image": f"data:image/jpeg;base64,{b64_frame}",
                "metrics": current_metrics,
                "is_injected": is_injected
            }
            
            await websocket.send_json(response)

    except WebSocketDisconnect:
        print(" Client Disconnected")
        if verifier: verifier.stop_injection()

@app.post("/api/toggle-injection")
async def toggle_injection(payload: dict):
    active = payload.get("active", False)
    if active and os.path.exists(ATTACK_VIDEO):
        verifier.start_injection(ATTACK_VIDEO)
        return {"status": "started"}
    else:
        verifier.stop_injection()
        return {"status": "stopped"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)