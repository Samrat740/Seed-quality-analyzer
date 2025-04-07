import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Color histogram (HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Texture (using LBP or Haralick-like descriptor)
    texture = cv2.Laplacian(gray, cv2.CV_64F).var()  # Variance as texture

    # Shape (contour features)
    contours, _ = cv2.findContours(cv2.Canny(gray, 100, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_area = sum([cv2.contourArea(c) for c in contours])

    return np.hstack([hist, texture, contour_area])

# Load dataset
X = []
y = []
data_dir = "dataset"

for label in os.listdir(data_dir):
    label_path = os.path.join(data_dir, label)
    for img_file in os.listdir(label_path):
        img_path = os.path.join(label_path, img_file)
        try:
            features = extract_features(img_path)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)
with open("model/seed_quality_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model training complete and saved to model/seed_quality_model.pkl")
