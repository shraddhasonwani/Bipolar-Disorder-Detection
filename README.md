# Bipolar-Disorder-Detection

**An AI-powered Bipolar Disorder Detection System** that leverages **machine learning and deep learning models** to analyze **multimodal inputs** (text, audio, and video).  
The project aims to assist in **early diagnosis and mental health assessment** using data-driven insights.

This repository contains an AI-based system built with **Python**, **Flask**, and **Deep Learning**, using **audio, text, and facial features** for multimodal mental health analysis.

---

## 🌟 Features

- Analyze **text, speech, and video** to detect bipolar tendencies.  
- Multimodal fusion of inputs using **deep learning models**.  
- **Web-based interface** built with Flask for easy interaction.  
- **Lightweight and modular** project structure for easy expansion.

---

## 🚀 Quick Start Guide

### 1️⃣ Install Dependencies
pip install -r requirements.txt
###2️⃣ Test Setup (IMPORTANT)
python test_setup.py
###3️⃣ Run the Flask App
python app.py
###4️⃣ Open in Browser
http://127.0.0.1:5000
##📦 Project Structure
Bipolar-Disorder-Detection/
│
├── app.py                  # Main Flask app
├── test_setup.py           # Test configuration script
├── models/                 # Directory for ML/DL models
├── templates/              # HTML templates
├── static/                 # CSS, JS, images
├── requirements.txt        # Python dependencies
└── README.md
##⚠️ Note on Large Model File
The main deep learning model models/best_fusion_transformer.pt is not included in this repository due to GitHub's file size limits (>100 MB).

You can download it separately:

Download the model file

After downloading, place it in the models/ folder to run the system.

##💻 Technologies Used
Python 3.x

Flask

PyTorch / TensorFlow (Deep Learning)

scikit-learn, librosa (Audio processing)

OpenCV (Video processing)

##📄 License
This project is open-source. You can freely use, modify, and share it.
