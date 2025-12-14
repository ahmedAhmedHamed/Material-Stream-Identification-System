"""
classical_feature_extraction.py

Classical feature extraction pipeline for material recognition.
Outputs fixed-size feature vectors suitable for SVM and k-NN.

Features:
- HSV color histogram
- Local Binary Patterns (LBP)
- Gabor texture filters
- Edge density

Author: (your name)
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Configuration
# -----------------------------

IMAGE_SIZE = (256, 256)

HSV_BINS = (8, 8, 8)       # 512 dims
LBP_POINTS = 8
LBP_RADIUS = 1             # 59 dims (uniform)
GABOR_ORIENTATIONS = 4     # 8 dims (mean + std per orientation)

TOTAL_FEATURE_DIM = 512 + 59 + (2 * GABOR_ORIENTATIONS) + 1


# -----------------------------
# Preprocessing
# -----------------------------

def preprocess_image(image):
    """
    Resize image to a fixed size.
    Input: RGB uint8 image
    Output: RGB uint8 image
    """
    return cv2.resize(image, IMAGE_SIZE)


# -----------------------------
# Feature extractors
# -----------------------------

def extract_hsv_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2],
        None,
        HSV_BINS,
        [0, 180, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def extract_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(
        gray,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method="uniform"
    )
    hist, _ = np.histogram(lbp, bins=59, range=(0, 59))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist


def extract_gabor(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = []

    for theta in np.linspace(0, np.pi, GABOR_ORIENTATIONS, endpoint=False):
        kernel = cv2.getGaborKernel(
            ksize=(31, 31),
            sigma=4.0,
            theta=theta,
            lambd=10.0,
            gamma=0.5,
            psi=0
        )
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        features.append(filtered.mean())
        features.append(filtered.std())

    return np.array(features, dtype=np.float32)


def extract_edge_density(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.array([edges.mean()], dtype=np.float32)


# -----------------------------
# Full feature vector
# -----------------------------

def extract_features(image):
    """
    Input: RGB uint8 image
    Output: 1D NumPy array of fixed length
    """
    image = preprocess_image(image)

    features = np.concatenate([
        extract_hsv_histogram(image),
        extract_lbp(image),
        extract_gabor(image),
        extract_edge_density(image)
    ])

    assert features.shape[0] == TOTAL_FEATURE_DIM
    return features


# -----------------------------
# Dataset-level extraction
# -----------------------------

def build_feature_matrix(image_paths, labels):
    """
    image_paths: list of image file paths
    labels: list or array of integer labels
    """
    X = []
    y = []

    for path, label in zip(image_paths, labels):
        image = cv2.imread(path)
        if image is None:
            raise IOError(f"Could not read image: {path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        features = extract_features(image)

        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)


# -----------------------------
# Feature scaling
# -----------------------------

def scale_features(X_train, X_val=None):
    """
    Standardize features (required for SVM / k-NN).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_val_scaled, scaler

    return X_train_scaled, scaler


# -----------------------------
# Example usage
# -----------------------------

if __name__ == "__main__":
    print(f"Classical feature extractor ready.")
    print(f"Feature dimensionality: {TOTAL_FEATURE_DIM}")
