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
st.markdown("Upload a clear image of a **seed** to assess its quality.")

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

# Function to add qualitative labels
def label_feature_values(features):
    labels = {}

    # Texture
    if features["Laplacian Variance (Texture)"] > 200:
        labels["Laplacian Variance (Texture)"] = "High Texture (Good)"
    elif features["Laplacian Variance (Texture)"] > 100:
        labels["Laplacian Variance (Texture)"] = "Moderate Texture"
    else:
        labels["Laplacian Variance (Texture)"] = "Low Texture (May be Poor)"

    # Contour Area
    if features["Contour Area"] > 2500:
        labels["Contour Area"] = "Large Size (Good)"
    elif features["Contour Area"] > 1500:
        labels["Contour Area"] = "Medium Size"
    else:
        labels["Contour Area"] = "Small Size (May be Weak)"

    # Aspect Ratio
    ar = features["Aspect Ratio"]
    if 0.8 <= ar <= 1.2:
        labels["Aspect Ratio"] = "Rounded (Ideal)"
    elif 0.6 <= ar <= 1.5:
        labels["Aspect Ratio"] = "Acceptable"
    else:
        labels["Aspect Ratio"] = "Elongated/Irregular (May be Poor)"

    # Extent
    if features["Extent"] > 0.8:
        labels["Extent"] = "Compact (Good)"
    else:
        labels["Extent"] = "Loose Shape"

    # Solidity
    if features["Solidity"] > 0.95:
        labels["Solidity"] = "Very Solid (Healthy)"
    elif features["Solidity"] > 0.85:
        labels["Solidity"] = "Solid"
    else:
        labels["Solidity"] = "Irregular (May be Hollow)"

    # Circularity
    if features["Circularity"] > 0.85:
        labels["Circularity"] = "Highly Circular (Good)"
    elif features["Circularity"] > 0.6:
        labels["Circularity"] = "Moderate"
    else:
        labels["Circularity"] = "Low Circularity (Irregular)"

    return labels

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

    label_data = label_feature_values(feature_data)

    show_features = st.toggle("📊 Show Feature Details")

    if show_features:
        st.markdown("### 🔍 Extracted Feature Values")
        for key, value in feature_data.items():
            if key != "Predicted Quality":
                label = label_data.get(key, "")
                st.markdown(f"- **{key}:** `{value:.2f}` — {label}")

        st.markdown("### 📉 Visual Summary")
        df = pd.DataFrame([feature_data])
        st.bar_chart(df.drop(columns=["Predicted Quality"]).T.rename(columns={0: "Value"}))

        def generate_pdf(image_path, prediction, features_dict, labels_dict):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            pdf.set_text_color(34, 139, 34)
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, "Seed Quality Report", ln=True, align='C')

            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

            margin_top = pdf.get_y() + 5
            left_col_x = 10
            right_col_x = 110
            line_height = 8

            pdf.set_xy(left_col_x, margin_top)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(90, line_height, f"Predicted Quality: {prediction}", ln=True)
            pdf.cell(90, line_height, "Extracted Features:", ln=True)
            pdf.set_font("Arial", size=12)

            for key, val in features_dict.items():
                if key != "Predicted Quality":
                    label = labels_dict.get(key, "")
                    line = f"- {key}: {val:.2f} ({label})"
                    pdf.multi_cell(90, line_height, line.encode("latin-1", "replace").decode("latin-1"))

            temp_image = "resized_image.jpg"
            img = Image.open(image_path)
            img.thumbnail((90, 90))
            img.save(temp_image)

            pdf.image(temp_image, x=right_col_x, y=margin_top, w=90)

            chart_path = "feature_chart.png"
            plt.figure(figsize=(6, 2.5))
            filtered_features = {k: v for k, v in features_dict.items() if k != "Predicted Quality"}
            plt.barh(list(filtered_features.keys()), list(filtered_features.values()), color='green')
            plt.xlabel("Value")
            plt.title("Feature Summary")
            plt.tight_layout()
            plt.savefig(chart_path)
            plt.close()

            max_y = max(pdf.get_y(), margin_top + 90) + 10
            pdf.set_y(max_y)
            pdf.image(chart_path, x=15, y=None, w=180)

            output_path = "seed_quality_report.pdf"
            pdf.output(output_path)

            if os.path.exists(temp_image):
                os.remove(temp_image)
            if os.path.exists(chart_path):
                os.remove(chart_path)

            return output_path

        pdf_path = generate_pdf(temp_path, prediction, feature_data, label_data)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📝 Download PDF Report",
                data=f,
                file_name="seed_quality_report.pdf",
                mime="application/pdf"
            )
