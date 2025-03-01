import os
import numpy as np
import tensorflow as tf
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load models
densenet_model = tf.keras.models.load_model("DenseNet_model.h5")
lstm_model = tf.keras.models.load_model("LSTM_model.h5")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("xgboost_model.json")
stacking_model = joblib.load("stacking_model.joblib")


# Function to load and preprocess a single CT scan image
def preprocess_image(image_path, image_size=(256, 256)):
    try:
        img = load_img(image_path, target_size=image_size, color_mode='grayscale')
        img_array = img_to_array(img) / 255.0
        img_array = img_array.reshape(1, *image_size, 1)  # Reshape for DenseNet
        return img_array
    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {e}")


# Function to predict cancer for a single image
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

    return "Cancer detected" if prediction == 1 else "No cancer detected"


# Function to load and preprocess test data
def load_ct_scan_images(image_dir, image_size=(256, 256)):
    images, labels = [], []
    for label_dir in os.listdir(image_dir):
        label_path = os.path.join(image_dir, label_dir)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                try:
                    img = tf.keras.preprocessing.image.load_img(
                        image_path, target_size=image_size, color_mode='grayscale'
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    images.append(img_array)
                    labels.append(1 if label_dir == "cancer" else 0)
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
    return np.array(images), np.array(labels)


# Set path to test data
test_image_dir = r"C:\Users\santh\Downloads\BATCH 18\Pancreatic Cancer Detection\final year project\pancreatic_cancer_data"
X_test, y_test = load_ct_scan_images(test_image_dir)

# Normalize and reshape images
X_test = X_test / 255.0
X_test = X_test.reshape(-1, 256, 256, 1)

# Generate feature maps using DenseNet
feature_maps_test = densenet_model.predict(X_test)

# Reshape feature maps for LSTM input
feature_maps_test_reshaped = feature_maps_test.reshape(feature_maps_test.shape[0], -1, 1)

# Generate LSTM features
lstm_features_test = lstm_model.predict(feature_maps_test_reshaped)

# Combine features for stacking
X_test_combined = np.hstack((feature_maps_test.reshape(feature_maps_test.shape[0], -1), lstm_features_test))

# Predict using the stacking model
y_pred = stacking_model.predict(X_test_combined)

# Evaluate the stacking model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Calculate ROC curve
y_score = stacking_model.predict_proba(X_test_combined)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

print(f'ROC AUC Score: {roc_auc:.2f}')

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Test cancer prediction for individual images
test_image_path = r"C:\Users\santh\Downloads\BATCH 18\Pancreatic Cancer Detection\final year project\pancreatic_cancer_data\cancer\22(12).jpg"
result = predict_cancer(test_image_path)
print(f"Prediction for the given image: {result}")
