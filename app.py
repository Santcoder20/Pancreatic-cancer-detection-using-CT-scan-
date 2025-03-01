import os
import numpy as np
import tensorflow as tf
import xgboost as xgb
import joblib
import base64
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)

# Load models
densenet_model = load_model("DenseNet_model.h5")
lstm_model = load_model("LSTM_model.h5")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("xgboost_model.json")
stacking_model = joblib.load("stacking_model.joblib")

# Load the segmentation model
segmentation_model = load_model("Segmentation_model.h5")  # Ensure you have this model

# Function to preprocess image for DenseNet
def preprocess_image(image_path, image_size=(256, 256)):
    try:
        img = load_img(image_path, target_size=image_size, color_mode='grayscale')
        img_array = img_to_array(img) / 255.0
        img_array = img_array.reshape(1, *image_size, 1)  # Reshape for DenseNet
        return img_array
    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {e}")

# Function to segment the CT scan image
def segment_image(image_path):
    try:
        # Load and preprocess the image for segmentation
        img = load_img(image_path, target_size=(256, 256), color_mode='grayscale')
        img_array = img_to_array(img) / 255.0
        img_array = img_array.reshape(1, 256, 256, 1)
        
        # Predict the segmentation mask
        segmentation_mask = segmentation_model.predict(img_array)[0]  # Output shape: (256, 256, 1)
        segmentation_mask = (segmentation_mask > 0.5).astype(np.uint8)  # Binary mask
        
        # Convert the mask to an RGB image for better visualization
        segmentation_mask_rgb = np.stack([segmentation_mask] * 3, axis=-1) * 255
        
        # Overlay the segmentation mask on the original image
        original_image = np.array(img).astype(np.uint8)
        original_image_resized = cv2.resize(original_image, (256, 256))
        overlay = cv2.addWeighted(original_image_resized, 0.7, segmentation_mask_rgb, 0.3, 0)
        return overlay
    except Exception as e:
        raise ValueError(f"Error segmenting image {image_path}: {e}")

# Function to predict cancer and return segmented image
def predict_cancer(image_path):
    img = preprocess_image(image_path)

    # Get DenseNet feature map
    feature_map = densenet_model.predict(img)

    # Reshape feature map for LSTM
    feature_map_reshaped = feature_map.reshape(1, -1, 1)

    # Get LSTM feature
    lstm_feature = lstm_model.predict(feature_map_reshaped)

    # Combine DenseNet and LSTM features
    combined_features = np.hstack((feature_map.reshape(1, -1), lstm_feature))

    # Make final prediction using the stacking model
    prediction = stacking_model.predict(combined_features)[0]  # Binary output: 0 or 1

    # Generate segmented image
    segmented_image = segment_image(image_path)

    return ("Cancer detected" if prediction == 1 else "No cancer detected"), segmented_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'result': 'error', 'message': 'No image uploaded'})

    file = request.files['image']
    
    # Save the uploaded file to a temporary location
    temp_file = os.path.join("temp_image.jpg")
    file.save(temp_file)

    # Make prediction and generate segmented image
    result, segmented_image = predict_cancer(temp_file)

    # Convert the segmented image to base64 for displaying in the UI
    _, buffer = cv2.imencode('.jpg', segmented_image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    # Clean up the temporary file
    os.remove(temp_file)

    return jsonify({'result': result, 'segmented_image': encoded_image})

if __name__ == '__main__':
    app.run(debug=True)
