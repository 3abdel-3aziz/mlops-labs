import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os


class ModelTrainer:

    def __init__(self, model_params=None):
        self.model = None
        self.model_params = model_params if model_params else {}

    def load_data(self, data_path):
        return pd.read_csv(data_path)

    def split_data(self, df, target_col, test_size, random_state):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

    def build_model(self):
        self.model = LinearRegression(**self.model_params)
        return self.model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def save_model(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)

    def run(self, data_path, target_col, test_size, random_state, model_path):
        # العملية كلها بقت بتعتمد على القيم اللي مبعوتة لها
        df = self.load_data(data_path)
        X_train, X_test, y_train, y_test = self.split_data(df, target_col, test_size, random_state)

        self.build_model()
        self.train(X_train, y_train)
        self.save_model(model_path)

        print("Linear Regression model trained and saved successfully")