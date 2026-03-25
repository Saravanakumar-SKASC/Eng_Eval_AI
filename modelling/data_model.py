import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import Config
import random

seed = 0
random.seed(seed)
np.random.seed(seed)


class Data():
    def __init__(self, X: np.ndarray, df: pd.DataFrame) -> None:
        self.embeddings = X
        self.y = df[Config.CLASS_COL]

        valid_mask = self.y.notna() & (self.y.astype(str).str.strip() != '')
        X_clean  = X[valid_mask]
        df_clean = df[valid_mask].reset_index(drop=True)
        y_clean  = self.y[valid_mask].reset_index(drop=True)

        indices = np.arange(len(X_clean))
        idx_train, idx_test = train_test_split(
            indices, test_size=0.2, random_state=seed, stratify=y_clean
        )

        self.X_train  = X_clean[idx_train]
        self.X_test   = X_clean[idx_test]
        self.y_train  = y_clean.iloc[idx_train].values
        self.y_test   = y_clean.iloc[idx_test].values
        self.train_df = df_clean.iloc[idx_train].reset_index(drop=True)
        self.test_df  = df_clean.iloc[idx_test].reset_index(drop=True)

    def get_type(self):         return self.y
    def get_X_train(self):      return self.X_train
    def get_X_test(self):       return self.X_test
    def get_type_y_train(self): return self.y_train
    def get_type_y_test(self):  return self.y_test
    def get_train_df(self):     return self.train_df
    def get_embeddings(self):   return self.embeddings
    def get_type_test_df(self): return self.test_df
