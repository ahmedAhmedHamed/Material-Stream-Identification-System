#!/usr/bin/env python3
import shutil
import os
from pathlib import Path

def setup_models():
    repo_root = Path(__file__).parent.parent
    android_project = Path(__file__).parent
    assets_dir = android_project / "app" / "src" / "main" / "assets"
    
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    knn_best = repo_root / "classifiers" / "KNN_best"
    svm_best = repo_root / "classifiers" / "SVM_16"
    
    model_files = [
        (knn_best / "knn_classifier.joblib", assets_dir / "knn_classifier.joblib"),
        (knn_best / "knn_scaler.joblib", assets_dir / "knn_scaler.joblib"),
        (svm_best / "svm_classifier.joblib", assets_dir / "svm_classifier.joblib"),
        (svm_best / "svm_scaler.joblib", assets_dir / "svm_scaler.joblib"),
    ]
    
    for src, dst in model_files:
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied {src.name} to {dst}")
        else:
            print(f"Warning: {src} not found")

if __name__ == "__main__":
    setup_models()

