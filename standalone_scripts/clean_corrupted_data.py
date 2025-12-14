import cv2
import os
from pathlib import Path


def clean_images(root):
    exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    for path in Path(root).rglob("*"):
        if path.suffix.lower() in exts:
            try:
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError("Failed to read image")
            except Exception as e:
                print(f"Error with {path}: {e}")
                os.remove(path)


# Example usage
clean_images("../dataset")
