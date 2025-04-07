import pickle
from feature_extractor import extract_features

def predict_seed_quality(image_path):
    with open('model/seed_quality_model.pkl', 'rb') as f:
        model = pickle.load(f)

    features = extract_features(image_path).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

# Example usage
if __name__ == '__main__':
    image_path = 'test_seed.jpg'
    result = predict_seed_quality(image_path)
    print(f"Predicted seed quality: {result}")
