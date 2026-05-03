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

## 📁 Important Files (Repository Structure)

Here is a breakdown of the key files you should upload to your GitHub repository:

### Backend & AI Logic
- **`app.py`**: The main Flask server application. It handles routing, receives the uploaded images via the `/api/predict` endpoint, and communicates with the AI model.
- **`model_utils.py`**: The core AI logic. This file handles loading the EfficientNet-B3 model, preprocessing the images, executing the inference, and generating the Explainable AI (Grad-CAM) heatmap overlays.
- **`run.py`**: A simple entry-point script to start the Flask server easily.
- **`requirements.txt`**: Lists all the necessary Python packages required to run the project.

### Frontend UI (Inside the `static/` folder)
- **`static/index.html`**: The structure of the web page, including the upload zone, the results panel, and the Grad-CAM visualization section.
- **`static/style.css`**: The premium, glassmorphism-styled dark theme that makes the application look modern and professional.
- **`static/script.js`**: The client-side logic that handles file drag-and-drop, communicates with the backend API, and smoothly animates the results (like the confidence ring and probability bars) onto the screen.

### ⚠️ A Note on the Model File
The **`best_b3_model.pth`** file is your trained AI model. 
- GitHub has a strict **100 MB file size limit**. Since your B3 model is around 43 MB, you *can* push it directly to GitHub without issues.
- However, if the file ever exceeds 100 MB in the future, you will need to use [Git Large File Storage (LFS)](https://git-lfs.com/) to upload it, or host the model file externally (like on Google Drive or AWS S3) and add instructions on where to download it.

---

## 🧠 Diagnostic Categories Supported (HAM10000)
1. Actinic keratoses and intraepithelial carcinoma (akiec)
2. Basal cell carcinoma (bcc)
3. Benign keratosis-like lesions (bkl)
4. Dermatofibroma (df)
5. Melanoma (mel)
6. Melanocytic nevi (nv)
7. Vascular lesions (vasc)
