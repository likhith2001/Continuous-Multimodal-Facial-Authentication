# Continuous-Multimodal-Facial-Authentication

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)

## 📖 Project Overview
Traditional static authentication mechanisms (like a one-time face scan) are increasingly vulnerable to session hijacking and sophisticated deepfakes (lip-syncing, full-face reenactment). This project introduces an **Adaptive Continuous Multimodal Facial Authentication system** designed to shift security from a single entry-point check to a persistent sliding-window verification process. 

Instead of looking for pixel-level artifacts that can be erased by video compression, this framework detects **Biometric Incoherence**—the subtle temporal desynchronization between distinct facial regions (Eyes and Lips). By extracting Dense Optical Flow and passing it through a Two-Stream 3D-CNN, the model successfully flags deepfakes where the lip movements do not biologically align with the upper face dynamics.

### ✨ Key Features
* **Two-Stream Fusion Architecture:** Independently processes Eye and Lip motion dynamics (via Farneback Optical Flow) to catch partial fakes.
* **Synthetic Incoherence Training:** A novel, tool-agnostic training strategy that artificially time-shifts real biometric streams to teach the model the fundamental concept of desynchronization.
* **Few-Shot Adaptation:** Utilizes Model-Agnostic Meta-Learning (MAML) to adapt to new, unseen users with just 5 seconds of enrollment video.
* **Codec-Invariant:** Highly robust against severe video compression found in real-world streaming environments.

## 🛠️ Tech Stack
* **Language:** Python 3.8+
* **Deep Learning Framework:** PyTorch, learn2learn (for MAML)
* **Computer Vision:** OpenCV (`cv2`), Dlib (68-point facial landmarks)
* **Hyperparameter Tuning:** Optuna
* **Deepfake Attack Generation:** Wav2Lip, First Order Motion Model (FOMM), LivePortrait
* **Data Handling & Metrics:** NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn

## 📊 Results Benchmark

The system was evaluated against baseline spatial models. Our Two-Stream Fusion with MAML consistently outperformed traditional models with a significantly smaller computational footprint (~0.6M parameters).

| Method | Dataset | Deepfake Tool | Region | Accuracy | Compute |
| :--- | :--- | :--- | :--- | :--- | :--- |
| XceptionNet (CNN) | FaceForensics++ | Face2Face, DeepFakes | Full Face | ~96.00% | High (~23M params) |
| Amerini et al. | FaceForensics++ | DeepFakes | Optical Flow | 81.60% | Very High (VGG16) |
| **Proposed Fusion System** | **GRID** | **Syn. Incoherence** | **Combined** | **100.00%** | **Medium (~0.6M params)** |
| **Proposed Fusion System** | **MOBIO** | **Syn. Incoherence** | **Combined** | **96.63%** | **Medium (~0.6M params)** |
| **Proposed Fusion System** | **FaceForensics++** | **Syn. Incoherence** | **Combined** | **98.11%** | **Medium (~0.6M params)** |

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/Continuous-Multimodal-Facial-Authentication.git](https://github.com/yourusername/Continuous-Multimodal-Facial-Authentication.git)
cd Continuous-Multimodal-Facial-Authentication
```

### 2. Create a Virtual Environment
```bash
conda create -n continuous_auth python=3.9 -y
conda activate continuous_auth
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: Ensure you install the correct PyTorch version with CUDA support for your system from the [official PyTorch website](https://pytorch.org/get-started/locally/).)*

---

## 📥 Downloading Pre-trained Weights

To run the pipeline and generate simulated attacks, you must download several pre-trained weights and place them in their respective directories.

**1. Dlib Shape Predictor**
* Download the `shape_predictor_68_face_landmarks.dat` file.
* Place it in the root directory of this repository.

**2. Wav2Lip Checkpoints**
* Clone the [official Wav2Lip repository](https://github.com/Rudrabha/Wav2Lip) into a folder named `Wav2Lip` in the root directory.
* Download the `wav2lip.pth` checkpoint and place it in `Wav2Lip/checkpoints/`.

**3. First Order Motion Model (FOMM)**
* Clone the [official FOMM repository](https://github.com/AliaksandrSiarohin/first-order-model) into a folder named `first-order-model`.
* Download the `vox-cpk.pth.tar` weights and place them inside the `first-order-model/` directory.

**4. LivePortrait**
* Clone the [official LivePortrait repository](https://github.com/KwaiVGI/LivePortrait) into a folder named `LivePortrait`.
* Follow their specific README instructions to download their base model weights.

---

## 📂 Acquiring Datasets

The project pipeline expects datasets to be located in a `data/` folder in the root directory.

* **GRID Audio-Visual Corpus** * *Access:* Open access.
  * *Download:* Available from the [official GRID corpus website](http://spandh.dcs.shef.ac.uk/gridcorpus/). Download the high-quality video and audio archives and place them in `data/gridcorpus/`.
* **MOBIO Dataset**
  * *Access:* Restricted (Requires signing an End User License Agreement / EULA).
  * *Download:* Request access through the [Idiap Research Institute](https://www.idiap.ch/dataset/mobio). Once approved, place the `idiap` and `unis` folders into `data/mobio/`.
* **FaceForensics++**
  * *Access:* Restricted (Requires filling out a Google Form to receive the download script).
  * *Download:* Follow instructions on the [FaceForensics GitHub](https://github.com/ondyari/FaceForensics). Download the "Real" videos (c23 compression) and place them in `data/FaceForensics/original_sequences/youtube/c23/videos/`.

---

## ⚙️ Running the Pipeline

Once your environment is set up, weights are downloaded, and data is in place, you can run the entire automated pipeline (Attack Generation -> Optical Flow Extraction -> MAML Training -> Evaluation).

```bash
python main.py
```

You will be greeted with an interactive CLI menu:
1. Select the Dataset (`GRID`, `MOBIO`, `FaceForensics++`, or `BOTH`).
2. Select the Mode (`Test Mode` for quick verification, `Production Mode` for full training and metric graphs).
3. Select the Attack Method (`Wav2Lip`, `FOMM`, or the recommended `Synthetic Incoherence` generator).
4. Select the Biological Region (`Lip`, `Eye`, or `Combined`).

Results, trained PyTorch models (`.pth`), and performance graphs will be automatically saved to the `saved_models/` and `results/` directories.

## 🔮 Future Work (Phase 3)
Active development is currently focused on wrapping the trained meta-learning models into a FastAPI backend and developing a responsive frontend dashboard for real-time, sliding-window webcam authentication.
