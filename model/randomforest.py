import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from numpy import *
import random

num_folds = 0
seed = 0
np.random.seed(seed)
random.seed(seed)


class RandomForest(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super(RandomForest, self).__init__()
        self.model_name  = model_name
        self.embeddings  = embeddings
        self.y           = y
        self.predictions = None

        # ── Improved hyperparameters ──────────────────────────────────────
        self.mdl = RandomForestClassifier(
            n_estimators=500,        # 500 trees — good balance speed vs accuracy
            max_depth=None,          # grow full trees
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',     # sqrt(n_features) — standard for classification
            class_weight='balanced', # handles class imbalance (Others has very few rows)
            random_state=seed,
            n_jobs=-1                # use all CPU cores
        )
        self.data_transform()

    def train(self, data) -> None:
        print(f"[RandomForest] Training '{self.model_name}' on {data.X_train.shape[0]} samples...")
        self.mdl = self.mdl.fit(data.X_train, data.y_train)
        print(f"[RandomForest] Training complete.")

    def predict(self, X_test: pd.Series):
        self.predictions = self.mdl.predict(X_test)
        return self.predictions

    def print_results(self, data):
        acc = accuracy_score(data.y_test, self.predictions)
        print(f"\n{'='*60}")
        print(f"  Model : {self.model_name}")
        print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"{'='*60}")
        print(classification_report(data.y_test, self.predictions, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(data.y_test, self.predictions))

    def data_transform(self) -> None:
        ...
