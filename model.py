from pathlib import Path

import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATASET_PATH = BASE_DIR / "dataset.csv"


def train_model():
    data = pd.read_csv(DATASET_PATH)

    X = data.drop("disease", axis=1).values
    y = data["disease"]

    n_neighbors = min(7, len(X))
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    return model


def load_model():
    if not MODEL_PATH.exists():
        return train_model()

    return joblib.load(MODEL_PATH)
