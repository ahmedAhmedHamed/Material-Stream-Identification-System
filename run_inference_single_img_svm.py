# Import custom classifier class for proper unpickling (if needed)
# Note: SVM uses standard sklearn SVC, so no custom class import needed

from inference import predict_image, predict_with_confidence, predict_with_probabilities

# Basic prediction
image_path = r"C:\gitcloned\Material-Stream-Identification-System\dataset\metal\9ff42c2b-bb0a-4d1b-ad48-38fb4886616d.jpg"
classifier_path = "classifiers/KNN_best/knn_classifier.joblib"

prediction, probabilities = predict_with_probabilities(image_path, classifier_path)
print(f"Prediction: {prediction}")
for class_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
    print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")



