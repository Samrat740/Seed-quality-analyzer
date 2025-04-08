import streamlit as st
import pickle
import os
from feature_extractor import extract_features

# Load trained model
with open("model/seed_quality_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Seed Quality Analyzer", layout="centered")
st.title("🌾 Seed Quality Analyzer")
st.markdown("Upload a clear image of a **single seed** to assess its quality.")

# Track state across interactions
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

# File upload section
uploaded_file = st.file_uploader("📁 Upload Seed Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_path = "temp_image.png"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(temp_path, caption="📷 Uploaded Image", use_container_width=True)

    if st.button("🔍 Analyze Quality"):
        try:
            features = extract_features(temp_path).reshape(1, -1)
            prediction = model.predict(features)[0]

            st.success(f"🌾 Predicted Quality: **{prediction}**")

            # --- Show detailed feature insights ---
            st.markdown("### 🔍 Feature Insights")

            feature_labels = {
                0: ("Red Mean", "Color intensity in red channel"),
                1: ("Green Mean", "Color intensity in green channel"),
                2: ("Blue Mean", "Color intensity in blue channel"),
                3: ("Aspect Ratio", "Shape balance (w/h) — lower = more round"),
                4: ("Contour Area", "Size of the seed"),
                5: ("Extent", "Compactness in bounding box — closer to 1 = tighter fit"),
                6: ("Solidity", "Shrivelled or broken seeds = lower value"),
                7: ("Circularity", "Roundness — 1 = perfect circle")
            }

            for idx, (label, description) in feature_labels.items():
                st.write(f"**{label}**: {features[0][idx]:.2f} → {description}")


        except Exception as e:
            st.error("❌ Error during processing. Make sure the image is clear and well-lit.")
            st.exception(e)