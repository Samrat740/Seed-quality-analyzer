import cv2
import numpy as np

def extract_features(image_path):
    image = cv2.imread(image_path)
    resized = cv2.resize(image, (200, 200))

    # --- Color: HSV Histogram (512D) ---
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hsv_hist = cv2.normalize(hist, hist).flatten()

    # --- Texture: Laplacian Variance (1D) ---
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # --- Edge detection for contour ---
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Default values for shape features
    contour_area = aspect_ratio = extent = solidity = circularity = 0

    if contours:
        c = max(contours, key=cv2.contourArea)

        # Basic shape descriptors
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h if h != 0 else 0
        contour_area = cv2.contourArea(c)
        rect_area = w * h
        extent = float(contour_area) / rect_area if rect_area != 0 else 0

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(contour_area) / hull_area if hull_area != 0 else 0

        perimeter = cv2.arcLength(c, True)
        circularity = (4 * np.pi * contour_area) / (perimeter ** 2) if perimeter != 0 else 0

    # Final feature vector
    feature_vector = np.hstack([
        hsv_hist,            # 512
        laplacian_var,       # 1
        contour_area,        # 1
        aspect_ratio,        # 1
        extent,              # 1
        solidity,            # 1
        circularity          # 1
    ])

    return feature_vector