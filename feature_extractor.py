import cv2
import numpy as np

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Color histogram
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Texture (Laplacian variance)
    texture = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Shape (contour area)
    contours, _ = cv2.findContours(cv2.Canny(gray, 100, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_area = sum([cv2.contourArea(c) for c in contours])

    return np.hstack([hist, texture, contour_area]).reshape(1, -1)
