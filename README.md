Explainable EEG-Based ADHD Screening using Machine Learning and LLMs



📌 Overview

This project presents an Explainable AI framework for ADHD screening using multi-channel EEG signals.
The system combines signal processing, feature optimization, deep learning, and Large Language Models (LLMs) to deliver accurate and interpretable predictions.

Unlike traditional black-box models, this approach emphasizes transparency and trust, providing both feature-level explanations (SHAP) and human-readable insights (LLaMA-3).

---

🚀 Key Features

- 🧪 EEG Signal Processing
  
  - NeuroDCT-ICA-based preprocessing
  - Artifact removal using ICA
  - Signal normalization
  - Epoch segmentation

- 📊 Feature Engineering
  
  - Frequency transformation using I-DCT
  - Time-domain, DCT, and PSD feature extraction

- ⚙️ Feature Optimization
  
  - Criss-Cross Optimization (CCO) for selecting discriminative features

- 🤖 Machine Learning Models

  - Logistic Regression (baseline model)
  - Support Vector Machine (SVM)
  - Random Forest (best performing model)
  - XGBoost (optional)

Random Forest achieved the highest performance and was selected for further explainability analysis.
- 🔍 Explainable AI (XAI)
  
  - SHAP-based feature importance visualization
  - LLaMA-3 powered explanation (Clinician + Parent level)

- 🖥️ Interactive Dashboard
  
  - Built with Streamlit
  - Real-time EEG analysis and interpretation

---

🏗️ System Architecture

EEG Data → Preprocessing → Epoching → I-DCT → Feature Extraction  
→ Criss-Cross Optimization → Deep Learning Model  
→ Prediction → SHAP Explanation → LLaMA Interpretation → Dashboard


📊 Dataset

- EEG dataset sourced from Kaggle:
  https://www.kaggle.com/datasets/inancigdem/eeg-data-for-mental-attention-state-detection

---

▶️ How to Run

1️⃣ Clone the repository

git clone https://github.com/your-username/ADHD-EEG-XAI-System.git
cd ADHD-EEG-XAI-System

2️⃣ Install dependencies

pip install -r requirements.txt

3️⃣ Run Streamlit dashboard

streamlit run app/app.py (It contains all steps for running the interface)

---

📈 Results

- Achieved classification using optimized EEG features
- Reduced noise using NeuroDCT-ICA preprocessing
- Improved interpretability using SHAP + LLaMA
- Real-time explainable predictions via dashboard

---

🧠 Explainable AI Output

The system provides:

-  Top contributing EEG features (SHAP)
-  Clinician-level explanation
-  Parent-friendly explanation
-  Medical disclaimer

---

🎯 Research Contribution

- Integration of Criss-Cross Optimization with EEG signals
- Combination of Deep Learning + Explainable AI
- Novel use of LLMs for human-centered explanation generation
- End-to-end pipeline from signal → prediction → interpretation

---

⚠️ Disclaimer

This system is designed as a screening tool only and does not provide medical diagnosis.
Clinical evaluation by a qualified healthcare professional is required.

---

🔮 Future Work

- Subject-level ADHD classification
- Transformer-based EEG models
- Real-time EEG device integration
- Improved confidence calibration

---

👨‍💻 Author

Pranav R
B.Tech Computer Science & Engineering
Amrita Vishwa Vidyapeetham

---
