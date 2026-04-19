TRAIN_DATA_PATH = r"D:\ITI\iti.mlops\data_0\train.csv"
TEST_DATA_PATH = r"D:\ITI\iti.mlops\data_0\test.csv"
#>>>>>>>>>>>>>>>>>>>>>>>>>>>> SPLITTING_DATA ===========================
TRAIN_SIZE = 0.8
VALIDATION_SIZE = 0.2
TARGET_COL = "Survived"
RANDOM_STATE = 42

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE
}

CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "loss_function": "Logloss",
    "verbose": False
}

MODEL_SAVE_PATH = "models/saved_model.pkl"