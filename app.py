import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pickle
from feature_extractor import extract_features

# Load model
with open('model/seed_quality_model.pkl', 'rb') as f:
    model = pickle.load(f)


st.set_page_config(page_title="Seed Quality Analyzer", layout="centered")
st.title("🌱 Automated Seed Quality Analyzer")
st.write("Upload an image of a seed to check its quality (Good / Average / Bad)")

uploaded_file = st.file_uploader("Choose a seed image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Seed Image', use_column_width=True)

    # Save temp image for OpenCV to read
    temp_path = "temp_seed.jpg"
    image.save(temp_path)

    if st.button("🔍 Analyze Quality"):
        try:
            features = extract_features(temp_path).reshape(1, -1)
            prediction = model.predict(features)[0]
            st.success(f"🌾 Predicted Quality: **{prediction}**")
        except Exception as e:
            st.error("Error during processing. Make sure the image is clear and well-lit.")
            st.exception(e)
