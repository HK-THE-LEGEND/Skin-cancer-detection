"""
Cloud-Enabled AI System for Skin Cancer Detection — Flask Server
Serves the premium skin cancer detection UI and handles prediction requests.
Using EfficientNet and Explainable Federated Learning.
"""

import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from model_utils import load_model, predict_image

# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODEL_PATH = os.path.join(BASE_DIR, 'best_b3_model.pth')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB max upload

# ============================================================
# App Initialization
# ============================================================

app = Flask(__name__, static_folder=STATIC_DIR)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Load model at startup
model = None
device = None

def init_model():
    """Load the trained model. Called once at startup."""
    global model, device
    if os.path.exists(MODEL_PATH):
        model, device = load_model(MODEL_PATH)
        print(f"[SkinAI] Model ready for inference!")
    else:
        print(f"[SkinAI] WARNING: Model file not found at {MODEL_PATH}")
        print(f"[SkinAI] Please place 'best_efficientnet_ham10000.pth' in: {BASE_DIR}")

# ============================================================
# Routes
# ============================================================

@app.route('/')
def index():
    """Serve the main application page."""
    return send_from_directory(STATIC_DIR, 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """Serve static assets (CSS, JS, images)."""
    return send_from_directory(STATIC_DIR, filename)

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Accept an image upload and return skin lesion classification results.
    
    Expects: multipart/form-data with field 'image'
    Returns: JSON with prediction, confidence, risk level, and all class probabilities
    """
    # Check model availability
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please ensure the model file exists.',
            'details': f'Expected at: {MODEL_PATH}'
        }), 503
    
    # Validate file presence
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided. Please upload an image.'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400
    
    # Validate file extension
    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            'error': f'Invalid file type: .{ext}',
            'details': f'Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    try:
        # Run prediction
        result = predict_image(file, model, device)
        return jsonify(result), 200
    
    except Exception as e:
        print(f"[SkinAI] Prediction error: {e}")
        return jsonify({
            'error': 'Failed to process the image.',
            'details': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    }), 200

# ============================================================
# Entry Point
# ============================================================

if __name__ == '__main__':
    init_model()
    print("\n" + "=" * 60)
    print("  Cloud-Enabled AI System for Skin Cancer Detection")
    print("  EfficientNet + Explainable Federated Learning")
    print("  Open: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
