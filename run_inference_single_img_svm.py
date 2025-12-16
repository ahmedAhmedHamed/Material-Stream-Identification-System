# Import custom classifier class for proper unpickling (if needed)
# Note: SVM uses standard sklearn SVC, so no custom class import needed

from inference import predict_image, predict_with_confidence, predict_with_probabilities

# Basic prediction
image_path = r"C:\Users\Stan\Downloads\pipe.jpg"
classifier_path = "classifiers/SVM_17/svm_classifier.joblib"

prediction, probabilities = predict_with_probabilities(image_path, classifier_path)
print(f"Prediction: {prediction}")
for class_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
    print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")

