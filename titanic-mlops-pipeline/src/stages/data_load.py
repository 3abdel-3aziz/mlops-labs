import yaml
import pandas as pd
from data.make_dataset import CSVDataIngestion, DataSplitter

def data_load():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Ingestion
    ingestor = CSVDataIngestion(config['paths']['raw_data'])
    df = ingestor.load_data()

    cols_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
    df = df.drop(columns=cols_to_drop)

    splitter = DataSplitter()
    X_train, X_test, y_train, y_test = splitter.split_data(
    df, 
    target_col=config['params']['target'], 
    test_size=config['params']['test_size'], 
    random_state=config['params']['random_state']
)
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    splitter.save_data(train_df, test_df, "data/interim")

if __name__ == "__main__":
    data_load()