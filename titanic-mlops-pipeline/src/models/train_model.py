import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from config import TRAIN_DATA_PATH, TEST_SIZE, RANDOM_STATE, TARGET_COL, MODEL_PATH


class ModelTrainer:

    def __init__(self):
        self.model = None

    def load_data(self):
        df = pd.read_csv(TRAIN_DATA_PATH)
        return df

    def split_data(self, df):
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        return train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

    def build_model(self):
        self.model = LinearRegression()
        return self.model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def save_model(self):
        joblib.dump(self.model, MODEL_PATH)

    def run(self):
        df = self.load_data()

        X_train, X_test, y_train, y_test = self.split_data(df)

        self.build_model()
        self.train(X_train, y_train)

        self.save_model()

        print("Linear Regression model trained and saved successfully")