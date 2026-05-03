# Cloud-Enabled AI System for Skin Cancer Detection

This project is a web-based, AI-powered system that classifies dermatoscopic skin images into 7 different diagnostic categories. It uses an **EfficientNet-B3** deep learning model trained on the HAM10000 dataset, combined with **Explainable AI (Grad-CAM)** to highlight the specific regions of the skin lesion that the model focused on to make its prediction.

## 🚀 How to Run the Project Locally

### Prerequisites
- Python 3.8 or higher installed on your system.
- The trained model file `best_b3_model.pth` must be placed in the root folder of this project.

### Step 1: Install Dependencies
Open your terminal or command prompt in the project directory and run:
```bash
pip install -r requirements.txt
```
*(This will install Flask, PyTorch, TorchVision, Pillow, and other required libraries).*

### Step 2: Start the Server
Run the startup script:
```bash
python run.py
```

### Step 3: Open the Application
Once the server starts, open your web browser and go to:
```
http://localhost:5000
```
You can now drag and drop a skin lesion image into the application to see the AI's prediction and the Grad-CAM analysis!

---
## 🧠 Diagnostic Categories Supported (HAM10000)
1. Actinic keratoses and intraepithelial carcinoma (akiec)
2. Basal cell carcinoma (bcc)
3. Benign keratosis-like lesions (bkl)
4. Dermatofibroma (df)
5. Melanoma (mel)
6. Melanocytic nevi (nv)
7. Vascular lesions (vasc)
