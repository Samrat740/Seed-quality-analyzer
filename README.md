# 🌱 Seed Quality Analyzer

An intelligent system for analyzing seed quality using **image processing** and a **machine learning model** — built with Python, OpenCV, and Streamlit.

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/Streamlit-App-red?logo=streamlit" alt="Streamlit">
</div>

---

## 🚀 Overview

This project provides a **full pipeline** to:
- Extract features from seed images (color, texture, shape).
- Predict the seed quality using a **trained ML model**.
- Generate a detailed **PDF report** for easy documentation.

All wrapped inside a simple, beautiful **Streamlit** web app! 🖥️✨

---

## ✨ Features

- 🔍 **Feature Extraction**:  
  - Color Histograms (HSV)
  - Texture Analysis (Laplacian Variance)
  - Shape Descriptors (Contour Area, Aspect Ratio, Extent, Solidity, Circularity)

- 🤖 **Machine Learning Prediction**:  
  Trained with a `RandomForestClassifier` on labeled seed datasets.

- 📄 **PDF Report Generation**:  
  Generates professional analysis reports ready for sharing.

- 🖼️ **Streamlit Web App**:  
  Upload images, predict, and download results — all from your browser.

---

## 🛠️ Project Structure
