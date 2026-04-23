import pandas as pd
import hydra
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import os 

from src.features import Handeller, NumericalHandeller, CategoricalHandeller, ImputeStrategy, Encoder, EncodingStrategy

@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # =========================
    # 1. Load Data (Using YOUR YAML: paths.train_data)
    # =========================
    df = pd.read_csv(cfg.paths.train_data)

    cols_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
    df = df.drop(columns=cols_to_drop)

    # 3. Target and Splitting (Using YOUR YAML: splitting.target)
    TARGET = cfg.splitting.target

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=cfg.splitting.test_size,    
        random_state=cfg.splitting.random_state 
    )

    # 4. Define handlers
    age_handler = NumericalHandeller(column="Age", impute_strategy=ImputeStrategy.MEAN)
    fare_handler = NumericalHandeller(column="Fare", impute_strategy=ImputeStrategy.MEDIAN)
    sex_handler = CategoricalHandeller(column="Sex")
    embarked_handler = CategoricalHandeller(column="Embarked")

    # 5 & 6. Fit and Transform
    for handler in [age_handler, fare_handler, sex_handler, embarked_handler]:
        handler.fit(X_train)
        X_train = handler.transform(X_train)
        X_test = handler.transform(X_test)

    # 8. Encoding
    sex_encoder = Encoder("Sex", EncodingStrategy.LABEL)
    embarked_encoder = Encoder("Embarked", EncodingStrategy.ONE_HOT)

    sex_encoder.fit(X_train)
    embarked_encoder.fit(X_train)

    X_train = sex_encoder.transform(X_train)
    X_train = embarked_encoder.transform(X_train)
    X_test = sex_encoder.transform(X_test)
    X_test = embarked_encoder.transform(X_test)

    # 9. Model (Using YOUR YAML model params)
    model = XGBClassifier(
        n_estimators=cfg.model.n_estimators,
        learning_rate=cfg.model.learning_rate,
        max_depth=cfg.model.max_depth,
        subsample=cfg.model.subsample,
        colsample_bytree=cfg.model.colsample_bytree
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"\n Accuracy: {accuracy_score(y_test, preds)}")

    os.makedirs(os.path.dirname(cfg.paths.model_save), exist_ok=True)
    joblib.dump(model, cfg.paths.model_save)
    
    artifacts = {
        "age_handler": age_handler,
        "fare_handler": fare_handler,
        "sex_handler": sex_handler,
        "embarked_handler": embarked_handler,
        "sex_encoder": sex_encoder,
        "embarked_encoder": embarked_encoder,
        "cols_to_drop": cols_to_drop
    }
    joblib.dump(artifacts, cfg.paths.preprocessor_save)
    print(f" Artifacts saved to: {cfg.paths.model_save}")

if __name__ == "__main__":
    main()