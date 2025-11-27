# 📘 Deep Learning Tutorial Series (16 Complete Python + Colab Modules)

Welcome to the **Deep Learning Tutorial Series**, a complete set of **16 hands-on, classroom-ready Jupyter notebooks** designed for:

* Teachers
* Trainers
* Engineering colleges
* Students learning ML/DL
* Skill development & FDP programs

This series covers **Image Processing → CNNs → Transfer Learning → Object Detection → Segmentation → NLP → Audio → Time Series → Deployment**.

All notebooks run seamlessly on **Google Colab**.

---

## 🧭 Table of Contents

1. [Overview](#overview)
2. [Modules](#modules)
3. [Recommended Usage](#recommended-usage)
4. [Folder Structure](#folder-structure)
5. [How to Run in Google Colab](#how-to-run-in-google-colab)
6. [Local Installation](#local-installation)
7. [Instructor Notes & Teaching Strategy](#instructor-notes--teaching-strategy)
8. [License](#license)

---

## 🌟 Overview

This repository contains a **16-module structured curriculum** that teaches deep learning using **Python**, **TensorFlow/Keras**, **PyTorch**, **OpenCV**, **Transformers**, **Ultralytics YOLO**, and **Gradio**.

Each module includes:

* Simple, reproducible code
* Demo-sized datasets (fast training in class)
* Step-by-step explanations
* Visualizations
* Instructor notes
* Exercises and projects

---

## 📚 Modules

Below is the complete list of notebooks included in the repository.

### **Module 1 — Image Processing & Augmentation**

* OpenCV basics
* Resize, crop, rotate, flip
* Color jitter, noise addition
* Albumentations augmentation
* CIFAR-10 mini CNN

---

### **Module 2 — Dataset Preparation & Annotation**

* VIA annotation
* YOLO TXT generation
* TFRecord intro
* Folder-based classification dataset pipeline

---

### **Module 3 — Flower Classification (Transfer Learning)**

* Load Flower dataset
* MobileNetV2 transfer learning
* Fine-tuning & evaluation

---

### **Module 4 — NN Fundamentals & Small CNNs**

* MLP (MNIST)
* Small CNN (CIFAR-10)
* Feature maps & filter visualization

---

### **Module 5 — Transfer Learning (ResNet / MobileNet)**

* Feature extraction
* Fine-tuning
* Comparing accuracy & training time

---

### **Module 6 — Object Detection (YOLOv8)**

* Install Ultralytics
* Create synthetic YOLO dataset
* Train YOLOv8n
* Run inference & visualize

---

### **Module 7 — Image Segmentation (UNet)**

* Create synthetic masks
* UNet implementation
* Dice loss
* Mask predictions

---

### **Module 8 — Classical CV & Embeddings**

* Canny edges
* ORB features & matching
* Simple CNN embedding extractor
* t-SNE visualization

---

### **Module 9 — NLP Basics & Embeddings**

* TF-IDF classifier
* LSTM sentiment model
* Hugging Face tokenization
* DistilBERT fine-tuning demo

---

### **Module 10 — Audio Classification**

* Generate sine/noise dataset
* Mel-spectrograms
* CNN classifier
* Evaluate & visualize

---

### **Module 11 — Time Series (LSTM/Transformer)**

* NDVI-like synthetic data
* Sliding windows
* LSTM forecaster
* Transformer forecaster
* MAE/MAPE evaluation

---

### **Module 12 — Evaluation & Explainability**

* RandomForest + SHAP
* CNN Grad-CAM
* ROC/PR curve
* Debugging checklist

---

### **Module 13 — Deployment (Gradio/Flask/TFLite)**

* Build Gradio inference UI
* Flask endpoint example
* Convert Keras → TFLite

---

### **Module 14 — Mini Projects**

* Flower classification project
* Sugarcane disease detection
* Audio classification
* NDVI forecasting
* Rubrics & submission guidelines

---

### **Module 15 — Instructor Utilities**

* Set seeds
* Download helpers
* Lesson templates
* Pre-run checklist

---

### **Module 16 — FAQ, Troubleshooting, Next Steps**

* Common Colab issues
* Performance optimization
* Extending curriculum (GANs, RL, SSL)
* Useful resources & links

---

## 🎯 Recommended Usage

**For Teachers:**

* Run notebooks in Colab during lectures
* Use demo datasets for fast execution
* Provide full datasets for assignments
* Use Module 14 for project evaluation

**For Students:**

* Explore, modify, and extend
* Train models on full datasets after class
* Use the deployment module to build mini-apps

---

## 📁 Folder Structure

```
deep-learning-tutorial-series/
├── notebooks/
│   ├── 01_Image_Processing.ipynb
│   ├── ...
│   └── 16_FAQ_Troubleshooting.ipynb
│
├── datasets/
│   └── README.md
│
├── utils/
│   └── …
│
├── environment/
│   ├── requirements_colab.txt
│   └── requirements_local.txt
│
├── .gitignore
└── README.md
```

---

## ▶️ How to Run in Google Colab

1. Open any `.ipynb` file
2. Click **“Open in Colab”**
3. Enable **GPU Runtime**
   *Runtime → Change Runtime Type → GPU*
4. Run the setup cell
5. Follow instructions inside each notebook

---

## 🖥 Local Installation

```bash
git clone https://github.com/<your-username>/<repo>.git
cd <repo>

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r environment/requirements_local.txt
```

---

## 🧑‍🏫 Instructor Notes & Teaching Strategy

* Use small datasets during class to keep runtime short
* Switch to full datasets for assignments
* For slow internet classrooms, pre-download all data
* Pin package versions to avoid Colab breakage
* Flip between theory → code → visualization


