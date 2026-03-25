from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import secrets

app = Flask(__name__)
app.secret_key = 'alexnet_cifar10_kartik_pune_2026_v1.0'  # Secure secret key
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your trained CIFAR-10 AlexNet model
print("Loading AlexNet CIFAR-10 model...")
# Line 19 - CHANGE TO:
model = tf.keras.models.load_model('models/alexnet_cifar10_trained.h5', compile=False)
# Test model on CIFAR-10 test data (ADD THIS after model = ...)
print("🧪 Testing model accuracy...")
(X_test, y_test), _ = tf.keras.datasets.cifar10.load_data()
test_predictions = model.predict(X_test[:1000])  # Test 1000 images
test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == y_test[:1000].flatten())
print(f"Model Test Accuracy: {test_accuracy*100:.1f}%")

print("Model loaded successfully!")

# CIFAR-10 class names (exactly matching your 10 classes)
CIFAR_CLASSES = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Secure filename and save
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and preprocess image for CIFAR-10 (32x32x3)
            img = Image.open(filepath).convert('RGB')
            img = img.resize((32, 32))  # CIFAR-10 input size
            img_array = np.array(img, dtype='float32') / 255.0  # Normalize 0-1
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1,32,32,3)
            
            # Make prediction
            predictions = model.predict(img_array, verbose=0)[0]  # Get probabilities
            top_indices = np.argsort(predictions)[::-1][:5]  # Top 5 predictions
            
            # Format results with class names
            results = []
            for i, idx in enumerate(top_indices):
                class_name = CIFAR_CLASSES[idx]
                confidence = float(predictions[idx])
                results.append({
                    'rank': i + 1,
                    'class': class_name,
                    'confidence': confidence,
                    'confidence_percent': round(confidence * 100, 1)
                })
            
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify({
                'success': True,
                'predictions': results,
                'top_prediction': results[0]
            })
        else:
            return jsonify({'error': 'Invalid file type. Use JPG, PNG, GIF, or BMP.'}), 400
            
    except Exception as e:
        # Clean up file on error
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    print("Starting AlexNet CIFAR-10 Flask app...")
    print("Open: http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
