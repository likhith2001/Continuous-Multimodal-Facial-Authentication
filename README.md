# Continuous-Multimodal-Facial-Authentication

![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=flat&logo=react&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)

## 📖 Project Overview

Traditional static authentication mechanisms (like a one-time face scan) are increasingly vulnerable to session hijacking and sophisticated deepfakes (lip-syncing, full-face reenactment). This project introduces an **Adaptive Continuous Multimodal Facial Authentication system** designed to shift security from a single entry-point check to a persistent sliding-window verification process.

Instead of looking for pixel-level artifacts that can be erased by video compression, this framework detects **Biometric Incoherence** — the subtle temporal desynchronization between distinct facial regions (Eyes and Lips). By extracting Dense Optical Flow and passing it through a Two-Stream 3D-CNN, the model flags deepfakes where the lip movements do not biologically align with the upper face dynamics.

The system includes a **real-time inference server** (FastAPI + WebSocket) and a **cinematic cybersecurity-themed dashboard** (React + Vite) that processes live webcam feeds at ~30 FPS with sub-second deepfake detection.

### ✨ Key Features

* **Two-Stream Fusion Architecture** — Independently processes Eye and Lip motion dynamics (via Farneback Optical Flow) to catch partial fakes
* **Synthetic Incoherence Training** — A tool-agnostic training strategy that artificially time-shifts real biometric streams to teach the model the fundamental concept of desynchronization
* **Few-Shot Adaptation** — Uses Model-Agnostic Meta-Learning (MAML) to adapt to new, unseen users with just a few seconds of enrollment video
* **Codec-Invariant** — Robust against severe video compression found in real-world streaming environments
* **Real-Time Detection** — Live webcam processing at ~30 FPS through a WebSocket pipeline with sub-second response
* **Multimodal Checks** — Simultaneously monitors lip sync, eye movement, gaze direction, face identity, and face obstruction
* **Session Hijack Simulation** — Built-in attack simulation with EMA-smoothed anomaly scores for demonstration
* **Interview-Friendly Gaze Tracking** — Flags sideways/upward glances and eye-only cheating while ignoring normal keyboard/notes glances

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React + Vite)                  │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────────┐  │
│  │  Webcam   │→│ Canvas @30fps │→│  WebSocket (base64 JPEG)  │──┼──┐
│  └──────────┘  └──────────────┘  └───────────────────────────┘  │  │
│  ┌──────────────────────────────────────────────────────────┐   │  │
│  │  HUD Dashboard: Trust Score · Anomaly Bars · ECG Chart   │   │  │
│  └──────────────────────────────────────────────────────────┘   │  │
└─────────────────────────────────────────────────────────────────┘  │
                                                                     │
                              WebSocket ws://localhost:8000/ws/video  │
                                                                     │
┌─────────────────────────────────────────────────────────────────┐  │
│                     BACKEND (FastAPI + PyTorch)                  │  │
│  ┌────────────┐   ┌──────────────┐   ┌───────────────────────┐  │  │
│  │  server.py  │←──│  LIFO Queue  │←──│  WebSocket Handler   │←─┼──┘
│  │  (uvicorn)  │   │  (maxsize=2) │   │  (frame decoder)     │  │
│  └────────────┘   └──────┬───────┘   └───────────────────────┘  │
│                          │                                       │
│               ┌──────────▼──────────┐                            │
│               │  AI Worker Thread    │                            │
│               │  RealTimeVerifier    │                            │
│               │  ┌────────────────┐  │                            │
│               │  │ Optical Flow   │  │                            │
│               │  │ MAML Inference │  │                            │
│               │  │ Gaze Tracking  │  │                            │
│               │  │ Face Identity  │  │                            │
│               │  └────────────────┘  │                            │
│               └──────────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Results Benchmark

The system was evaluated against baseline spatial models. The Two-Stream Fusion with MAML consistently outperformed traditional models with a significantly smaller computational footprint (~0.6M parameters).

| Method | Dataset | Deepfake Tool | Region | Accuracy | Compute |
| :--- | :--- | :--- | :--- | :--- | :--- |
| XceptionNet (CNN) | FaceForensics++ | Face2Face, DeepFakes | Full Face | ~96.00% | High (~23M params) |
| Amerini et al. | FaceForensics++ | DeepFakes | Optical Flow | 81.60% | Very High (VGG16) |
| **Proposed Fusion System** | **GRID** | **Syn. Incoherence** | **Combined** | **100.00%** | **Medium (~0.6M params)** |
| **Proposed Fusion System** | **MOBIO** | **Syn. Incoherence** | **Combined** | **96.63%** | **Medium (~0.6M params)** |
| **Proposed Fusion System** | **FaceForensics++** | **Syn. Incoherence** | **Combined** | **98.11%** | **Medium (~0.6M params)** |

---

## 📁 Project Structure

```
VSA/
├── README.md
├── Backend/
│   ├── server.py                      # FastAPI WebSocket server for real-time inference
│   ├── main.py                        # CLI pipeline: data generation → training → evaluation
│   ├── requirements.in                # Python dependencies
│   ├── shape_predictor_68_face_landmarks.dat  # Dlib 68-point face landmark model
│   ├── saved_models/                  # Trained MAML model checkpoints (.pth)
│   ├── src/
│   │   ├── realtime_inference.py      # Core detection engine (RealTimeVerifier class)
│   │   ├── extract_optical_flow.py    # Facial region extraction + dense optical flow
│   │   ├── train_maml.py             # MAML meta-learning training loop
│   │   ├── tune_optuna.py            # Hyperparameter search with Optuna
│   │   ├── generate_fakes.py         # Deepfake video generation (Wav2Lip/FOMM/LP)
│   │   └── generate_synthetic.py     # Synthetic incoherence data generation
│   ├── data/                          # Dataset storage
│   └── results/                       # Training logs and evaluation graphs
│
└── Frontend/
    ├── index.html
    ├── package.json
    ├── tailwind.config.js
    ├── vite.config.js
    └── src/
        ├── main.jsx                   # React entry point
        ├── App.jsx                    # Main dashboard component
        ├── index.css                  # Design system (glassmorphism, particles, animations)
        └── App.css                    # Base layout styles
```

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose |
|---|---|
| Python 3.12 | Core language |
| FastAPI + Uvicorn | WebSocket server for real-time streaming |
| PyTorch | Neural network training and inference |
| learn2learn | MAML meta-learning implementation |
| OpenCV (`cv2`) | Video processing and Farneback optical flow |
| Dlib | 68-point facial landmark detection |
| Optuna | Automated hyperparameter tuning |
| NumPy, Pandas | Array computation and result logging |
| Scikit-learn | Evaluation metrics (precision, recall, F1, ROC) |
| Matplotlib, Seaborn | Training visualization and production graphs |

### Frontend
| Technology | Purpose |
|---|---|
| React 19 + Vite | UI framework with hot module reloading |
| Tailwind CSS + PostCSS | Utility-first styling |
| Framer Motion | Animations and transitions |
| Recharts | Real-time ECG-style anomaly charts |
| Lucide React | Iconography |
| react-circular-progressbar | Trust score gauge |
| react-parallax-tilt | Interactive 3D tilt effects |
| Axios | REST API calls |
| Native WebSocket | Real-time frame streaming at ~30 FPS |

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Continuous-Multimodal-Facial-Authentication.git
cd Continuous-Multimodal-Facial-Authentication
```

### 2. Create a Virtual Environment
```bash
conda create -n continuous_auth python=3.12 -y
conda activate continuous_auth
```

### 3. Install Backend Dependencies
```bash
cd Backend
pip install -r requirements.in
pip install fastapi uvicorn learn2learn imutils
```
*(Ensure you install the correct PyTorch version with CUDA support for your system from the [official PyTorch website](https://pytorch.org/get-started/locally/).)*

### 4. Install Frontend Dependencies
```bash
cd Frontend
npm install
```

---

## 📥 Downloading Pre-trained Weights

To run the pipeline and generate simulated attacks, download the following pre-trained weights and place them in their respective directories.

**1. Dlib Shape Predictor**
* Download `shape_predictor_68_face_landmarks.dat`
* Place it in the `Backend/` directory

**2. Wav2Lip Checkpoints**
* Clone the [Wav2Lip repository](https://github.com/Rudrabha/Wav2Lip) into `Backend/Wav2Lip/`
* Download the `wav2lip.pth` checkpoint and place it in `Backend/Wav2Lip/checkpoints/`

**3. First Order Motion Model (FOMM)**
* Clone the [FOMM repository](https://github.com/AliaksandrSiarohin/first-order-model) into `Backend/first-order-model/`
* Download the `vox-cpk.pth.tar` weights and place them inside `Backend/first-order-model/`

**4. LivePortrait**
* Clone the [LivePortrait repository](https://github.com/KwaiVGI/LivePortrait) into `Backend/LivePortrait/`
* Follow their README instructions to download their base model weights

---

## 📂 Acquiring Datasets

The pipeline expects datasets to be located in the `Backend/data/` folder.

* **GRID Audio-Visual Corpus**
  * *Access:* Open access
  * *Download:* Available from the [official GRID corpus website](http://spandh.dcs.shef.ac.uk/gridcorpus/). Download the high-quality video and audio archives and place them in `data/gridcorpus/`

* **MOBIO Dataset**
  * *Access:* Restricted (requires signing an EULA)
  * *Download:* Request access through the [Idiap Research Institute](https://www.idiap.ch/dataset/mobio). Once approved, place the `idiap` and `unis` folders into `data/mobio/`

* **FaceForensics++**
  * *Access:* Restricted (requires filling out a Google Form to receive the download script)
  * *Download:* Follow instructions on the [FaceForensics GitHub](https://github.com/ondyari/FaceForensics). Download the "Real" videos (c23 compression) and place them in `data/FaceForensics/original_sequences/youtube/c23/videos/`

---

## ⚙️ Running the Training Pipeline

Once the environment is ready, weights are downloaded, and datasets are in place, you can run the full automated pipeline (Attack Generation → Optical Flow Extraction → MAML Training → Evaluation):

```bash
cd Backend
python main.py
```

The interactive CLI will walk you through:
1. **Dataset** — `GRID`, `MOBIO`, `FaceForensics++`, or `BOTH`
2. **Mode** — `Test Mode` (quick verification) or `Production Mode` (full training + evaluation graphs)
3. **Attack Method** — `Wav2Lip`, `FOMM`, or the recommended `Synthetic Incoherence` generator
4. **Biological Region** — `Lip`, `Eye`, or `Combined`

Trained models (`.pth`), performance graphs, and results are saved to `saved_models/` and `results/`.

---

## 🖥️ Running the Real-Time Dashboard

The real-time inference system runs independently of the training pipeline, using a pre-trained model.

**1. Start the Backend Server**
```bash
cd Backend
python server.py
```
The server loads the MAML model and starts listening on `http://localhost:8000`.

**2. Start the Frontend Dashboard**
```bash
cd Frontend
npm run dev
```
The dashboard opens at `http://localhost:5173`.

**3. Using the Dashboard**
1. Click **"Initialize Calibration"** — the system captures your biometric baseline (~3 seconds)
2. The system enters **Live Defense** mode and begins monitoring your webcam feed
3. Toggle **"Execute Session Hijack"** to simulate a deepfake injection attack
4. Watch the Trust Score, anomaly bars, and ECG chart respond in real-time

---

## 🔬 Detection Methodology

### Dense Optical Flow (Farneback)
Rather than analyzing raw pixels, the system computes movement vectors between consecutive frames. Facial landmarks from dlib isolate the lip and eye regions, and Farneback optical flow captures the micro-movement patterns within those regions across time.

### Two-Stream Fusion Architecture
Optical flow from the eye region and lip region is processed through separate 3D CNN branches, then fused at the feature level for joint classification. This catches deepfakes that manipulate only one facial area (e.g., Wav2Lip only modifies the mouth).

### MAML Meta-Learning
The 3D CNN is wrapped in a MAML meta-learning loop. During the calibration phase, the model performs rapid gradient updates on the specific user's facial dynamics, effectively personalizing the detection threshold to that individual's biological rhythm.

### Synthetic Incoherence Training
Instead of relying on tool-specific deepfake artifacts, the training data is generated by artificially time-shifting the eye and lip optical flow streams by 5–15 frames. This teaches the model the general concept of temporal desynchronization, making it robust against unseen deepfake methods.

### Real-Time Multimodal Checks

| Check | Method | What It Catches |
|---|---|---|
| Lip/Eye Anomaly | MAML optical flow inference | Deepfake manipulation |
| Gaze Tracking | Nose offset + iris intensity ratio | Looking away (interview cheating) |
| Face Identity | Histogram correlation against baseline | Person swap mid-session |
| Face Obstruction | Landmark detection failure | Covered or hidden face |
| Injection Detection | Explicit state flag from server | Session hijack attacks |

The gaze detection is tuned specifically for interview scenarios — it ignores downward glances toward the keyboard, notes, or screen, but flags horizontal head turns (>15% nose offset), upward tilts (looking above the webcam), and eye-only lateral movements (iris intensity ratio imbalance >1.4).

---

## 🎨 Dashboard Features

The frontend provides a cybersecurity-themed HUD (Heads-Up Display) built with React, Tailwind CSS, and Framer Motion:

* **Glassmorphism panels** with backdrop blur and accent glows
* **Boot-up sequence** with cascading entrance animations
* **ECG-style chart** showing real-time anomaly score fluctuations
* **Trust Score gauge** (circular progress bar)
* **Red vignette overlay** triggered during detected attacks
* **Glitch text effect** on verdict transitions
* **Floating ambient particles** in the background
* **Dark/Light mode** toggle
* **Parallax tilt** on the video panel

---

## 🔧 Notable Design Decisions

* **LIFO Frame Dropping** — The server uses a `Queue(maxsize=2)` so that if AI inference takes longer than the 33ms frame interval, stale frames are dropped automatically to keep the video feed live
* **EMA-Smoothed Scores During Injection** — When session hijack is active, anomaly scores use exponential moving average smoothing (0.78–0.95 range) to produce gradually fluctuating metrics rather than a hard binary flip
* **Post-Calibration Grace Period** — 30-frame suppression window after calibration prevents false positives during system stabilization
* **Injection Recovery Suppression** — 20-frame window after stopping injection prevents false "DIFFERENT PERSON" alerts while identity re-establishes

---

## 🙏 Acknowledgments

- [dlib](http://dlib.net/) — Face detection and landmark prediction
- [learn2learn](https://github.com/learnables/learn2learn) — MAML implementation
- [FaceForensics++](https://github.com/ondyari/FaceForensics) — Forgery detection benchmark
- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) — Lip-sync deepfake generation
- [First Order Motion Model](https://github.com/AliaksandrSiarohin/first-order-model) — Face reenactment
- [LivePortrait](https://github.com/KwaiVGI/LivePortrait) — Expression-driven face animation
- [GRID Corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/) — Audio-visual sentence corpus
- [MOBIO Database](https://www.idiap.ch/en/scientific-research/data/mobio) — Mobile biometric dataset
