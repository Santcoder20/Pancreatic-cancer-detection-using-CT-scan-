import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LSTM, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt

# Load and preprocess CT scan images
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

# Set paths
image_dir = r"C:\Users\ADMIN\Desktop\sanjay final year project\pancreatic_cancer_data"
images, labels = load_ct_scan_images(image_dir)

if len(images) == 0 or len(labels) == 0:
    raise ValueError("No images or labels loaded. Check the dataset path and structure.")

# Normalize images
images = images / 255.0

# Reshape images for CNN input
images = images.reshape(-1, 256, 256, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Check class distribution
print("Class distribution in training data:", np.bincount(y_train))

# Data augmentation
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2)
datagen.fit(X_train)

# DenseNet-121 model (modified for grayscale images)
def get_densenet_model(input_shape=(256, 256, 1)):
    inputs = Input(shape=input_shape)
    x = Conv2D(3, (1, 1), activation='linear')(inputs)  # Convert to 3 channels
    base_model = tf.keras.applications.DenseNet121(include_top=False, input_shape=(256, 256, 3))
    x = base_model(x)  # Pass through DenseNet
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)  # Binary classification
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Compile and train DenseNet-121
densenet_model = get_densenet_model()
densenet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
densenet_model.fit(datagen.flow(X_train, y_train, batch_size=16), validation_data=(X_test, y_test), epochs=50)


# Save DenseNet model
densenet_model.save("DenseNet_model.h5")
print("DenseNet model saved as DenseNet_model.h5")

# Generate feature maps
feature_maps_train = densenet_model.predict(X_train)
feature_maps_test = densenet_model.predict(X_test)

# Define Quad LSTM
def create_quad_lstm(input_shape):
    inputs = Input(shape=input_shape)
    lstm1 = LSTM(64, return_sequences=True)(inputs)
    lstm2 = LSTM(64, return_sequences=True)(lstm1)
    lstm3 = LSTM(64, return_sequences=True)(lstm2)
    lstm4 = LSTM(64)(lstm3)
    dropout = Dropout(0.5)(lstm4)
    return Model(inputs, dropout)

# Reshape feature maps for LSTM input
feature_maps_train_reshaped = feature_maps_train.reshape(feature_maps_train.shape[0], -1, 1)
feature_maps_test_reshaped = feature_maps_test.reshape(feature_maps_test.shape[0], -1, 1)

# Train Quad LSTM
lstm_model = create_quad_lstm((feature_maps_train_reshaped.shape[1], 1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(feature_maps_train_reshaped, feature_maps_train_reshaped, epochs=50, batch_size=32)

# Save LSTM model
lstm_model.save("LSTM_model.h5")
print("LSTM model saved as LSTM_model.h5")

# Generate LSTM features
lstm_features_train = lstm_model.predict(feature_maps_train_reshaped)
lstm_features_test = lstm_model.predict(feature_maps_test_reshaped)

# Train XGBoost on LSTM features
xgb_model = xgb.XGBClassifier()
xgb_model.fit(lstm_features_train, y_train)

# Save XGBoost model
xgb_model.save_model("xgboost_model.json")
print("XGBoost model saved as xgboost_model.json")

# Combine features for stacking
X_train_combined = np.hstack((feature_maps_train.reshape(feature_maps_train.shape[0], -1), lstm_features_train))
X_test_combined = np.hstack((feature_maps_test.reshape(feature_maps_test.shape[0], -1), lstm_features_test))

# Train final classifier
stacking_model = LogisticRegression()
stacking_model.fit(X_train_combined, y_train)

# Save stacking model
joblib.dump(stacking_model, "stacking_model.joblib")
print("Stacking model saved as stacking_model.joblib")

# Test the ensemble model
y_pred = stacking_model.predict(X_test_combined)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
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
