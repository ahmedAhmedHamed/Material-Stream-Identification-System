# Material Classification Android App

Android application for real-time material classification using camera frames.

## Setup

1. Copy model files to assets:
```bash
python setup_models.py
```

2. Copy Python modules:
```bash
python setup_python_modules.py
```

3. Build and run the app using Android Studio or Gradle.

## Requirements

- Android SDK 24+
- Chaquopy Python runtime
- Camera permission

## Model Files

The app requires the following model files in `app/src/main/assets/`:
- `knn_classifier.joblib`
- `knn_scaler.joblib`
- `svm_classifier.joblib`
- `svm_scaler.joblib`

These are copied from `classifiers/KNN_best/` and `classifiers/SVM_16/` by the setup script.

