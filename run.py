"""
Cloud-Enabled AI System for Skin Cancer Detection — Quick Start
Run this file to start the application: python run.py
"""

from app import app, init_model

if __name__ == '__main__':
    init_model()
    print("\n" + "=" * 60)
    print("  Cloud-Enabled AI System for Skin Cancer Detection")
    print("  EfficientNet + Explainable Federated Learning")
    print("  Open: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
