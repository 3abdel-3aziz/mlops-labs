import numpy as np
import json
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class ClassificationEvaluator:

    def __init__(self):
        pass

    def accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def precision(self, y_true, y_pred):
        return precision_score(y_true, y_pred, average="weighted", zero_division=0)

    def recall(self, y_true, y_pred):
        return recall_score(y_true, y_pred, average="weighted", zero_division=0)

    def f1(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average="weighted", zero_division=0)

    def confusion_matrix(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)

    def full_report(self, y_true, y_pred):
        return classification_report(y_true, y_pred)

    def evaluate_all(self, y_true, y_pred):
        return {
            "accuracy": self.accuracy(y_true, y_pred),
            "precision": self.precision(y_true, y_pred),
            "recall": self.recall(y_true, y_pred),
            "f1_score": self.f1(y_true, y_pred)
        }
    def save_metrics(self, metrics, output_path):
     
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
            
        print(f"Metrics saved successfully to: {output_path}")