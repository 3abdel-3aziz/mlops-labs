import joblib

class ModelPredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, df):
        return self.model.predict(df)