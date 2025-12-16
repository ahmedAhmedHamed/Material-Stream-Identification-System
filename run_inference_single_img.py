# Import custom classifier class for proper unpickling
from classifiers.KNN import WeightedKNeighborsClassifier

from inference import predict_image, predict_with_confidence

# Basic prediction
image_path = r"C:\Users\Stan\Downloads\cat.jpg"
classifier_path = "classifiers/KNN_best/knn_classifier.joblib"
from inference import predict_with_probabilities

prediction, probabilities = predict_with_probabilities(image_path, classifier_path)
print(f"Prediction: {prediction}")
for class_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
    print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")