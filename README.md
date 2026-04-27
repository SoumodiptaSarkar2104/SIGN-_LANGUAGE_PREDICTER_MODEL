# Sign_Language_Detector

# 🎧 EAR MODEL FOR REAL-TIME SIGN LANGUAGE DETECTION

> A lightweight, single-ear wearable assistive device for real-time sign language to speech translation using onboard AI/ML processing.

---

## 📌 Project Overview

This project introduces a **compact, ergonomic, and standalone ear-worn device** capable of translating sign language gestures into natural speech in real time.

Unlike traditional bulky systems (glove-based, external cameras, or smartphone-dependent setups), this solution provides:

- ✅ Lightweight wearable form factor  
- ✅ Fully standalone processing (No smartphone required)  
- ✅ Real-time gesture-to-speech conversion  
- ✅ Onboard AI/ML inference  
- ✅ Multi-sign language support (ISL-ready, expandable to ASL/BSL)

---

## 🚀 Key Features

- 📷 Front-facing mini camera for gesture capture  
- 🤖 Onboard AI/ML hand landmark recognition  
- 🧠 84-dimensional normalized feature extraction  
- 🔊 Integrated text-to-speech engine  
- 🔋 Rechargeable Li-Po battery  
- 📡 Optional Bluetooth/Wi-Fi for firmware updates  
- 🧩 Modular architecture for future upgrades  

---

## 🏗 System Architecture

```
Camera Capture
      ↓
Frame Preprocessing
      ↓
MediaPipe Hand Detection
      ↓
Landmark Feature Extraction (84D Vector)
      ↓
ML Model Prediction
      ↓
Stability / Debouncing Logic
      ↓
Text Output
      ↓
Text-to-Speech Engine
      ↓
Speaker Output
```

---

## 🧠 Feature Extraction Details

Each frame is processed using **MediaPipe Hands**:

- 21 landmarks per hand  
- 2 hands maximum  
- (x, y) coordinates per landmark  
- Wrist-relative normalization  
- Zero padding for missing hands  

Final Feature Vector Size:

```
2 hands × 21 landmarks × 2 coordinates = 84 features
```

This ensures:
- Distance invariance  
- Scale invariance  
- Robust real-time performance  

---

## 📊 Model Performance

| Metric | Old Model | New Model |
|--------|-----------|-----------|
| Average Accuracy | 55.4% | 75.2% |
| Improvement | — | +19.8% |

🔥 ~20% improvement achieved through optimized feature normalization and training pipeline.

---

## 🛠 Tech Stack

### 🔹 AI / ML
- TensorFlow
- Keras
- MediaPipe
- OpenCV
- NumPy
- Scikit-learn

### 🔹 Computer Vision
- MediaPipe Hands
- OpenCV DNN
- YOLO (optional enhancement)

### 🔹 Hardware Components
- Microprocessor / Microcontroller (ESP32 / Raspberry Pi Zero)
- Forward-facing mini camera
- Infrared emitter + IR sensor (optional)
- Acoustic driver (mini speaker)
- Audio processor / DAC
- Rechargeable Li-Po battery
- Flash memory
- Bluetooth / Wi-Fi module

---

## 📁 Project Structure

```
EAR-Model/
│
├── dataset/              # Training dataset (Images/Videos)
├── models/               # Saved trained models (.h5)
├── training/             # Model training scripts
├── realtime/             # Real-time inference scripts
├── hardware/             # Hardware integration docs
├── requirements.txt
└── README.md
```

---

## ⚙ Installation Guide

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/EAR-Model.git
cd EAR-Model
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install tensorflow keras opencv-python mediapipe numpy scikit-learn pyttsx3
```

---

## 🧪 Training the Model

```bash
python train.py
```

After training, the model will be saved as:

```
isl_model.h5
```

---

## 🎥 Running Real-Time Detection

```bash
python realtime.py
```

- Press `q` to exit the program.
- Final output text will be converted to speech automatically.

---

## 🔁 Real-Time Decoding Logic

The inference engine performs:

1. Frame capture
2. Feature extraction
3. Model prediction
4. Confidence threshold check
5. Stability (repeat frame confirmation)
6. Space insertion when no hand detected
7. Text overlay on live feed
8. Final speech synthesis

---

## 🎯 Use Cases

- 🏫 Educational Institutions  
- 🏢 Workplace Communication  
- 🛍 Daily Conversations  
- 🚨 Emergency Situations  
- 🏥 Medical Communication  

---

## 🔮 Future Improvements

- Transformer-based gesture decoding  
- Vision-Language Model (LLM) integration  
- TensorFlow Lite quantization for edge deployment  
- Custom ASIC / Embedded chip deployment  
- Mobile companion application  
- Multilingual speech output  

---

## 👨‍💻 Authors

- **Jishnu Paul**
- **Sagnik Basu**
- **Soumodipta Sarkar**

Department of CSE (AI & ML)  
Narula Institute of Technology  
2025–2026
