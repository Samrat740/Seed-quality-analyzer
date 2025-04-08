import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from feature_extractor import extract_features

# 📁 Set your dataset directory
DATASET_DIR = "dataset/"

X = []
y = []

# 💡 Assumes folder names are class labels (e.g., Good/, Bad/, Average/)
for label in os.listdir(DATASET_DIR):
    class_dir = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(class_dir):
        continue

    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        try:
            features = extract_features(img_path)
            if features is not None:
                X.append(features)
                y.append(label)
        except Exception as e:
            print(f"❌ Error processing {img_file}: {e}")

X = np.array(X)
y = np.array(y)

print(f"✅ Dataset shape: {X.shape}, Labels: {set(y)}")

# 📊 Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# 💾 Save the model
joblib.dump(clf, "seed_quality_model.pkl")
print("🎉 Model saved as 'seed_quality_model.pkl'")
