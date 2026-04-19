import pandas as pd
import numpy as np 
from abc import ABC, abstractmethod
from config import TRAIN_DATA_PATH , TEST_DATA_PATH , TARGET_COL, TRAIN_SIZE, TEST_SIZE , RANDOM_STATE
from sklearn import train_test_split
from enum import    Enum



class ImputeStrategy(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"


class Handeller(ABC):

    @abstractmethod
    def fit_impute(self, df):
        pass

    def log_scale(self, df , column):
        self.df[self.column] = np.log(self.df[self.column].min()+1)
    def drop_column(self, df , column) :
        self.df = self.df.drop(column=column, inplace=True)


class NumericalHandeller(Handeller) :

    def __init__(self, column , group_by_column=None ) :
        self.column = column 
        self.group_by_column = group_by_column

        self.learned_number = None

    def fit_transform_impute(self, df):

        try :
            if self.impute_strategy == ImputeStrategy.MEAN :
                    self.df[self.column].fillna(df[self.column].mean(), inplace=True)
            else :
                    self.impute_strategy == ImputeStrategy.MEDIAN
                    df[self.column].fillna(df[self.column].median(), inplace=True)
        except :
            raise Exception("error unvalid imputing strategy was inserted")
    def fit_impute (self, df) :

        if self.group_by_column :
            if self.impute_strategy == ImputeStrategy.MEAN:
                self.learned_number = df.groupby(self.group_by_column)[self.column].mean()
            elif self.impute_strategy == ImputeStrategy.MEDIAN:
                self.learned_number = df.groupby(self.group_by_column)[self.column].median()
        else:
            if self.impute_strategy == ImputeStrategy.MEAN :
                self.learned_number = df[self.column].mean()
                return df
            elif self.impute_strategy == ImputeStrategy.MEDAIN :
                self.learned_number = df[self.column].median()
        return df

          
    def transform(self, df) :
        df_copy = df.copy()
        df_copy[self.column] = df_copy[self.column].fillna(self.learned_number)
        return df_copy


class CategoricalHandeller(Handeller):

    def __init__(self, column, group_by_column=None):
        self.column = column 
        self.group_by_column = group_by_column
        self.learned_number = None

    def fit_impute(self, df):
        if self.group_by_column is not None:
            self.learned_number = df.groupby(self.group_by_column)[self.column].apply(lambda x: x.mode()[0] if not x.mode().empty else None)
        else:
            self.learned_number = df[self.column].mode()[0]
        return df

    def transform(self, df):
        df_copy = df.copy()
        if self.group_by_column is not None:
            df_copy[self.column] = df_copy[self.column].fillna(df_copy[self.group_by_column].map(self.learned_number))
        else:
            df_copy[self.column] = df_copy[self.column].fillna(self.learned_number)
        return df_copy

    def fit_transform_impute(self, df):
        self.fit_impute(df)
        return self.transform(df)
    