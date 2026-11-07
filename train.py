import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
import os

PROCESSED_PATH = "data/processed"
MODEL_PATH = "models/isolation_forest.pkl"

def train_model():

    train = pd.read_csv(os.path.join(PROCESSED_PATH, "train.csv"))

    X_train = train.drop('label', axis=1)

    print("Training Isolation Forest...")

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train)

    joblib.dump(model, MODEL_PATH)

    print("Model training complete.")