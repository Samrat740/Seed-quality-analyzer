import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from feature_extractor import extract_features

# 👇 Folder structure:
# dataset/
# ├── good/
# ├── average/
# └── poor/

data_dir = "dataset"
labels = []
features = []

for label in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, label)
    if not os.path.isdir(class_dir):
        continue
    for file in os.listdir(class_dir):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(class_dir, file)
            try:
                feat = extract_features(img_path)
                features.append(feat)
                labels.append(label)
            except Exception as e:
                print(f"❌ Error with {img_path}: {e}")

X = np.array(features)
y = np.array(labels)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Save the model
os.makedirs("model", exist_ok=True)
with open("model/seed_quality_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ Model trained and saved successfully!")
