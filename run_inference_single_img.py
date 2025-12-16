# Import custom classifier class for proper unpickling
from classifiers.KNN import WeightedKNeighborsClassifier

from inference import predict_image, predict_with_confidence

# Basic prediction
image_path = r"C:\Users\Stan\Downloads\cat.jpg"
classifier_path = "classifiers/KNN_best/knn_classifier.joblib"

prediction = predict_image(image_path, classifier_path)
print(f"Predicted class: {prediction}")

# With confidence score
prediction, confidence = predict_with_confidence(image_path, classifier_path)
print(f"Predicted class: {prediction} (confidence: {confidence:.2%})")
