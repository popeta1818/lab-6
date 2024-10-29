import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris


data = load_iris()
X, y = data.data, data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

class MinimumDistanceClassifier:
    def fit(self, X, y):
        self.class_means = {}
        for class_label in np.unique(y):
            self.class_means[class_label] = X[y == class_label].mean(axis=0)
    
    def predict(self, X):
        predictions = []
        for sample in X:
            distances = {label: np.linalg.norm(sample - mean) for label, mean in self.class_means.items()}
            predictions.append(min(distances, key=distances.get))
        return np.array(predictions)

mdc = MinimumDistanceClassifier()
knn = KNeighborsClassifier(n_neighbors=1)

def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    return acc, conf_matrix

def validate_classifier(classifier, classifier_name):
    print(f"\n--- {classifier_name} ---\n")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc, conf_matrix = evaluate_model(y_test, y_pred)
    print("Stratified Hold-Out 70/30")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", conf_matrix)

    skf = StratifiedKFold(n_splits=10)
    acc_scores = []
    for train_index, test_index in skf.split(X, y):
        classifier.fit(X[train_index], y[train_index])
        y_pred = classifier.predict(X[test_index])
        acc, _ = evaluate_model(y[test_index], y_pred)
        acc_scores.append(acc)
    print("\nStratified 10-Fold Cross-Validation")
    print("Average Accuracy:", np.mean(acc_scores))

    loo = LeaveOneOut()
    acc_scores = []
    for train_index, test_index in loo.split(X):
        classifier.fit(X[train_index], y[train_index])
        y_pred = classifier.predict(X[test_index])
        acc = accuracy_score(y[test_index], y_pred)
        acc_scores.append(acc)
    print("\nLeave-One-Out")
    print("Average Accuracy:", np.mean(acc_scores))

validate_classifier(mdc, "Minimum Distance Classifier")
validate_classifier(knn, "1-Nearest Neighbor Classifier")
