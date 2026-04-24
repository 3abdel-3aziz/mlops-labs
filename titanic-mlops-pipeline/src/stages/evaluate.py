import pandas as pd
import yaml
import joblib
from sklearn.metrics import accuracy_score, classification_report

def evaluate():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    test_df = pd.read_csv("data/processed/test_processed.csv")
    model = joblib.load("models/model.joblib")

    TARGET = config['params']['target']
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    print("Evaluating model...")
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    print(f"\n Model Accuracy: {acc:.4f}")
    print("\nDetailed Classification Report:")
    print(report)

    with open("reports/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(report)

if __name__ == "__main__":
    evaluate()