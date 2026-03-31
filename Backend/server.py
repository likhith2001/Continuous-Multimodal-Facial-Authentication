"""FastAPI WebSocket server for real-time deepfake detection.

Streams webcam frames from the React frontend via WebSocket, processes them
through the AI inference pipeline in a background thread, and returns
annotated frames with detection metrics (verdict, trust score, anomaly scores).
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import base64
import numpy as np
import uvicorn
import threading
import queue
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

# Path to the production-trained fusion model
COMBINED_MODEL = "saved_models/faceforensics_model_combined_synthetic_prod.pth"

verifier = None
# Small queue (maxsize=2) ensures real-time processing by dropping stale frames
frame_queue = queue.Queue(maxsize=2)
# Shared metrics dictionary — updated by the AI worker, read by the WebSocket handler
current_metrics = {
    "status": "waiting", 
    "trust_score": 1.0, 
    "verdict": "REAL", 
    "lip_prob_fake": 0.0, 
    "eye_prob_fake": 0.0
}

def ai_inference_loop():
    """Background worker thread that continuously pulls frames from the queue
    and runs the deepfake detection model on each one."""
    global current_metrics
    print("   [Background] AI Worker Thread Started")
    
    while True:
        try:
            # Unpack frame and injection flag from queue tuple
            item = frame_queue.get(timeout=0.5)
            img_input, is_injected = item
            
            if verifier is not None:
                result = verifier.process_stream(img_input, dataset_name='faceforensics', is_injected=is_injected)
                if result:
                    current_metrics = result
                    if result.get("status") == "active":
                        print(f"   [AI SCORE] Fusion Anomaly: {result['lip_prob_fake']:.4f} | Verdict: {result['verdict']}")
        except queue.Empty:
            continue
        except Exception as e:
            print(f"   [Background Error] {e}")

@app.on_event("startup")
async def startup_event():
    """Loads the AI model and starts the background inference thread on server boot."""
    global verifier
    print("\n SERVER STARTING: Loading AI Models...")
    try:
        verifier = RealTimeVerifier(COMBINED_MODEL)
        print(" Models Loaded! Starting Background Worker...\n")
        
        t = threading.Thread(target=ai_inference_loop, daemon=True)
        t.start()
    except Exception as e:
        print(f" Error Loading Models: {e}")

@app.websocket("/ws/video")
async def video_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time video streaming.
    Receives base64-encoded frames, optionally swaps with synthetic injection,
    queues for AI analysis, and returns the processed frame with metrics."""
    await websocket.accept()
    print(" Client Connected (Real-Time LIFO Mode)")

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

            # Drop oldest frame if queue is full to prevent lag accumulation
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            # Pass injection flag alongside the display frame for detection
            frame_queue.put((display_frame, is_injected))

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
        with frame_queue.mutex:
            frame_queue.queue.clear()
        if verifier: verifier.stop_injection()

@app.post("/api/enroll")
async def trigger_enrollment():
    """Starts biometric calibration (MAML adaptation to the current user)."""
    if verifier:
        verifier.start_enrollment()
        return {"status": "enrollment_started"}
    return {"status": "error"}

@app.post("/api/toggle-injection")
async def toggle_injection(payload: dict):
    """Toggles the session hijack simulation (synthetic frame injection)."""
    active = payload.get("active", False)
    if active and verifier:
        verifier.start_injection()
        return {"status": "started"}
    elif verifier:
        verifier.stop_injection()
        return {"status": "stopped"}
    return {"status": "error"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)