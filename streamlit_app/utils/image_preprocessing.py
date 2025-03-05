import cv2
import numpy as np

def preprocess_image(image):
    """
    Preprocesses an OpenCV image by converting it to grayscale,
    applying thresholding, and denoising.
    """
    # Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Adaptive Thresholding
    processed_image = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return processed_image
