import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"

def run_preprocessing():

    os.makedirs(PROCESSED_PATH, exist_ok=True)

    files = [
        "UNSW-NB15_1.csv",
        "UNSW-NB15_2.csv",
        "UNSW-NB15_3.csv",
        "UNSW-NB15_4.csv"
    ]

    dfs = []

    for file in files:
        file_path = os.path.join(RAW_PATH, file)
        print(f"Reading {file_path}")
        df = pd.read_csv(file_path, low_memory=False)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    print("\nMerged Shape:", df.shape)

    df = df[df['label'].notna()]
    df['label'] = df['label'].astype(int)

    print("Shape after dropping NaN labels:", df.shape)

    print("\nLabel Distribution:")
    print(df['label'].value_counts())

    drop_cols = []
    if 'id' in df.columns:
        drop_cols.append('id')
    if 'attack_cat' in df.columns:
        drop_cols.append('attack_cat')

    df.drop(columns=drop_cols, inplace=True)

    # Fill only numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop('label', axis=1)
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    X_scaled['label'] = y.values

    majority_label = y.value_counts().idxmax()

    normal_data = X_scaled[X_scaled['label'] == majority_label]
    attack_data = X_scaled[X_scaled['label'] != majority_label]

    train = normal_data.sample(frac=0.8, random_state=42)
    test_normal = normal_data.drop(train.index)
    test = pd.concat([test_normal, attack_data])

    train.to_csv(os.path.join(PROCESSED_PATH, "train.csv"), index=False)
    test.to_csv(os.path.join(PROCESSED_PATH, "test.csv"), index=False)

    print("\nTrain Shape:", train.shape)
    print("Test Shape:", test.shape)
    print("Preprocessing completed successfully.")