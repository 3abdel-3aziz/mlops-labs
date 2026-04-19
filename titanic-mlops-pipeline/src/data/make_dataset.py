import pandas as pd
import numpy as np 
from abc import ABC, abstractmethod
from config import TRAIN_DATA_PATH , TEST_DATA_PATH , TARGET_COL, TRAIN_SIZE, TEST_SIZE , RANDOM_STATE
from sklearn import train_test_split
from enum import    Enum
class DataIngestion(ABC):

    @abstractmethod
    def load_data(self):
        pass    

class CSVDataIngestion(DataIngestion):

    def __init__(self, file_path) :
        self.file_path = file_path

    def load_data(self) :
        data = pd.read_csv(self.file_path) 
        return data   
        

class DataSplitter:
    def split_data(Self, df):
        x= df.drop(TARGET_COL, axis=1)
        y= df[TARGET_COL]
        return train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


