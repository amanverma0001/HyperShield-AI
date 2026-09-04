# 🩺 HyperShield AI – Hypertension Risk Predictor & Dietary Advisor

> **Course Project:** CSET 343 – AI in HealthCare  
> **Institution:** School of Computer Science Engineering & Technology  

---

## 👥 Team Details

* **Amandeep Verma** (Enrollment No: `E23CSEU1483`)
* **Pujith Sri Sai Karri** (Enrollment No: `E23CSEU1410`)

---

## 📌 Project Overview

**HyperShield AI** is an intelligent, full-stack machine learning web application designed for early **Hypertension (High Blood Pressure) Risk Prediction, Explainable Risk Factor Attribution, Lifestyle Simulation, and Dietary Guidance**.

Hypertension affects over 1.2 billion people globally and is frequently called the *"silent killer"* because individuals often experience no symptoms until severe cardiovascular damage occurs. **HyperShield AI** shifts healthcare from passive blood pressure logging to **proactive prediction and lifestyle intervention**.

---

## ✨ Key Novel Features

1. 🎯 **Early Pre-Hypertension Risk Screening:** Classifies patient risk into 4 clinical stages (*Normal, Pre-Hypertension, Stage 1, Stage 2*) using lifestyle parameters, vitals, and medical history.
2. 📊 **SHAP-Based Explainable AI:** Provides a patient-specific visual risk factor contribution breakdown (e.g., *"Daily Salt Intake contributes 42% to your risk score"*).
3. 🎛️ **Interactive "What-If" Lifestyle Simulator:** Live interactive sliders allowing users to test how reducing sodium intake or increasing physical activity reduces their future risk score in real-time.
4. 🥗 **Risk-Adaptive DASH Diet Generator:** Automatically creates a customized low-sodium daily meal plan tailored to the patient's predicted risk stage.
5. 📄 **1-Click Clinical PDF Summary Export:** Exports a 1-page summary report with visual charts ready for physician consultation.

---

## 📁 Repository Structure

```
CSET 343 project/
├── 📄 app.py                           # FastAPI Backend Server & REST API Endpoints
├── 📄 train_model.py                   # Machine Learning Model Training Script
├── 📄 hypershield_model.joblib         # Serialized Trained Random Forest Model
├── 📄 AI_in_HealthCare_Proposal.pdf    # Submission Proposal PDF Document
├── 📄 AI_in_HealthCare_Proposal.html   # Proposal HTML Source File
├── 📄 README.md                        # Documentation File
└── 📁 static/                          # Frontend Assets
     ├── 📄 index.html                  # Single-Page Web App Dashboard
     ├── 📄 style.css                   # Responsive CSS & Styling System
     └── 📄 app.js                      # Client JavaScript & Chart.js Integration
```

---

## 🛠️ Tech Stack

* **Backend & ML Engine:** Python 3.10+, `FastAPI`, `Uvicorn`, `Scikit-Learn`, `Pandas`, `NumPy`, `Joblib`
* **Frontend Dashboard:** HTML5, CSS3 (Glassmorphism & Flexbox/Grid), Vanilla JavaScript (ES6+), `Chart.js`
* **PDF Export Engine:** Google Chrome Headless / ReportLab

---

## ⚙️ Installation & Setup Guide

### 1. Clone or Open Project Directory
```bash
cd "CSET 343 project"
```

### 2. Install Required Python Dependencies
```bash
pip install fastapi uvicorn scikit-learn pandas numpy joblib reportlab
```

### 3. Train the Machine Learning Model (Optional)
```bash
python3 train_model.py
```
*(This generates the `hypershield_model.joblib` file with classification metrics).*

### 4. Run the HyperShield AI Server
```bash
python3 app.py
```

### 5. Access the Web Dashboard
Open your browser and navigate to:
👉 **`http://127.0.0.1:8000`**

---

## 📡 API Endpoints Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the interactive web application dashboard |
| `POST` | `/api/predict` | Takes patient vitals/lifestyle, returns risk stage, probabilities & SHAP breakdown |
| `POST` | `/api/simulate` | Takes baseline data + target lifestyle changes, returns simulated risk reduction % |
| `POST` | `/api/diet` | Returns risk-stage tailored DASH daily meal plan & sodium guidelines |

---

## 📜 Medical Disclaimer
*HyperShield AI is developed as an academic machine learning research project for CSET 343. It is intended for preliminary health screening and educational purposes and does not replace professional medical diagnosis or clinical advice.*
