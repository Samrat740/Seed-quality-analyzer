import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from feature_extractor import extract_features
from fpdf import FPDF
from datetime import datetime
from PIL import Image

# Load trained model
MODEL_PATH = "model/seed_quality_model.pkl"
model = joblib.load(MODEL_PATH)

# App UI config
st.set_page_config(page_title="Seed Quality Analyzer", layout="centered")
st.title("🌾 Seed Quality Analyzer")
st.markdown("Upload a clear image of a **single seed** to assess its quality.")

# Initialize session state
for key in ["analyzed", "features", "prediction"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "analyzed" else False

# Upload UI
uploaded_file = st.file_uploader("📁 Upload Seed Image", type=["jpg", "jpeg", "png"])

# Clear previous analysis if image is removed
if uploaded_file is None and st.session_state.analyzed:
    st.session_state.analyzed = False
    st.session_state.features = None
    st.session_state.prediction = None

# Track the last uploaded file to detect changes
if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None

# If image is removed or a new one is uploaded, reset analysis state
if (uploaded_file is None and st.session_state.analyzed) or \
   (uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded_filename):
    st.session_state.analyzed = False
    st.session_state.features = None
    st.session_state.prediction = None

# Save the new filename to track future changes
if uploaded_file is not None:
    st.session_state.last_uploaded_filename = uploaded_file.name


if uploaded_file is not None:
    temp_path = "temp_image.png"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(temp_path, caption="📷 Uploaded Image", use_container_width=True)

    # --- Analyze button ---
    if st.button("🔍 Analyze Quality"):
        try:
            features = extract_features(temp_path).reshape(1, -1)

            if features.shape[1] != model.n_features_in_:
                st.error(f"❌ Feature mismatch: Expected {model.n_features_in_} features, but got {features.shape[1]}")
            else:
                prediction = model.predict(features)[0]

                # Save to session state
                st.session_state.features = features
                st.session_state.prediction = prediction
                st.session_state.analyzed = True

        except Exception as e:
            st.error("❌ Error during processing. Make sure the image is clear and well-lit.")
            st.exception(e)

# --- Show results if analyzed ---
if st.session_state.analyzed:

    features = st.session_state.features
    prediction = st.session_state.prediction

    st.success(f"🌾 Predicted Quality: **{prediction}**")

    laplacian_value = features[0][-6]
    contour_area = features[0][-5]
    aspect_ratio = features[0][-4]
    extent = features[0][-3]
    solidity = features[0][-2]
    circularity = features[0][-1]

    feature_data = {
        "Laplacian Variance (Texture)": laplacian_value,
        "Contour Area": contour_area,
        "Aspect Ratio": aspect_ratio,
        "Extent": extent,
        "Solidity": solidity,
        "Circularity": circularity,
        "Predicted Quality": prediction
    }

    df = pd.DataFrame([feature_data])

    # --- Toggle display ---
    show_features = st.toggle("📊 Show Feature Details")

    if show_features:
        st.markdown("### 🔍 Extracted Feature Values")
        for key, value in feature_data.items():
            if key != "Predicted Quality":
                st.markdown(f"- **{key}:** `{value:.2f}`")

        st.markdown("### 📉 Visual Summary")
        st.bar_chart(df.drop(columns=["Predicted Quality"]).T.rename(columns={0: "Value"}))

        #Generate PDF
        def generate_pdf(image_path, prediction, features_dict):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            # Title
            pdf.set_text_color(34, 139, 34)
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, "Seed Quality Report", ln=True, align='C')
            
            # Timestamp
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            
            # Layout constants
            margin_top = pdf.get_y() + 5
            left_col_x = 10
            right_col_x = 110  # For image
            line_height = 8

            # --- Left Column: Features and Prediction ---
            pdf.set_xy(left_col_x, margin_top)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(90, line_height, f"Predicted Quality: {prediction}", ln=True)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(90, line_height, "Extracted Features:", ln=True)
            pdf.set_font("Arial", size=12)

            for key, val in features_dict.items():
                if key != "Predicted Quality":
                    pdf.cell(90, line_height, f"- {key}: {val:.2f}", ln=True)

            # --- Right Column: Resized Image ---
            temp_image = "resized_image.jpg"
            img = Image.open(image_path)
            img.thumbnail((90, 90))
            img.save(temp_image)

            pdf.image(temp_image, x=right_col_x, y=margin_top, w=90)

            # --- Chart Below ---
            chart_path = "feature_chart.png"
            plt.figure(figsize=(6, 2.5))  # Smaller chart
            filtered_features = {k: v for k, v in features_dict.items() if k != "Predicted Quality"}
            plt.barh(list(filtered_features.keys()), list(filtered_features.values()), color='green')
            plt.xlabel("Value")
            plt.title("Feature Summary")
            plt.tight_layout()
            plt.savefig(chart_path)
            plt.close()

            # Position chart below the lowest Y
            max_y = max(pdf.get_y(), margin_top + 90) + 10
            pdf.set_y(max_y)
            pdf.image(chart_path, x=15, y=None, w=180)

            output_path = "seed_quality_report.pdf"
            pdf.output(output_path)
            return output_path
        
        # --- PDF Report Download ---
        pdf_path = generate_pdf(temp_path, prediction, feature_data)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📝 Download PDF Report",
                data=f,
                file_name="seed_quality_report.pdf",
                mime="application/pdf"
            )
