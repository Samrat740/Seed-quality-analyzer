import cv2
import numpy as np

def extract_features(image_path):
    image = cv2.imread(image_path)
    resized = cv2.resize(image, (200, 200))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / h
    contour_area = cv2.contourArea(c)
    rect_area = w * h
    extent = float(contour_area) / rect_area

    # Add: Solidity
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(contour_area) / hull_area if hull_area != 0 else 0

    # Add: Circularity
    perimeter = cv2.arcLength(c, True)
    circularity = (4 * np.pi * contour_area) / (perimeter ** 2) if perimeter != 0 else 0

    # Color features
    b_mean = np.mean(resized[:, :, 0])
    g_mean = np.mean(resized[:, :, 1])
    r_mean = np.mean(resized[:, :, 2])

    return np.array([r_mean, g_mean, b_mean, aspect_ratio, contour_area, extent, solidity, circularity])
