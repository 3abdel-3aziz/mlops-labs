TRAIN_DATA_PATH = r"D:\ITI\iti.mlops\data_0\train.csv"
TEST_DATA_PATH = r"D:\ITI\iti.mlops\data_0\test.csv"
#>>>>>>>>>>>>>>>>>>>>>>>>>>>> SPLITTING_DATA ===========================

TARGET_COL = "Survived"
TRAIN_SIZE = 0.8
TEST_SIZE = 0.2  
RANDOM_STATE = 42

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
MODEL_PATH = "models/model.pkl"


XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE
}



MODEL_SAVE_PATH = "models/saved_model.pkl"