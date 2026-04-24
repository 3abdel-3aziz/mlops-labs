import pandas as pd
import yaml
from xgboost import XGBClassifier
import joblib
import os

def train():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_df = pd.read_csv("data/processed/train_processed.csv")
    
    TARGET = config['params']['target']
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    
    model = XGBClassifier(
        n_estimators=config['model_params']['n_estimators'],
        learning_rate=config['model_params']['learning_rate'],
        max_depth=config['model_params']['max_depth'],
        subsample=config['model_params']['subsample'],
        colsample_bytree=config['model_params']['colsample_bytree'],
        random_state=config['params']['random_state'],
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    print("Training the model...")
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    model_path = "models/model.joblib"
    joblib.dump(model, model_path)
    
    print(f" Model trained and saved to: {model_path}")

if __name__ == "__main__":
    train()