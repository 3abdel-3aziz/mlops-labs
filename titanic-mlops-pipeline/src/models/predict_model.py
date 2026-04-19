import pandas as pd
import joblib
from config import MODEL_PATH


class ModelPredictor:

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def predict(self, df):
        return self.model.predict(df)