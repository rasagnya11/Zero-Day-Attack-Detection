import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    accuracy_score,
    precision_recall_curve
)

PROCESSED_PATH = "data/processed"
MODEL_PATH = "data/processed/isolation_forest.pkl"
OUTPUT_PATH = "outputs"

def evaluate_model():

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    test = pd.read_csv(os.path.join(PROCESSED_PATH, "test.csv"))
    test = test[test['label'].notna()]
    test['label'] = test['label'].astype(int)

    X_test = test.drop('label', axis=1)
    y_true = test['label']

    model = joblib.load(MODEL_PATH)

    print("Evaluating model...")

    scores = model.decision_function(X_test)
    preds_raw = model.predict(X_test)
    y_pred = [1 if p == -1 else 0 for p in preds_raw]

    # Basic Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, -scores)

    print("\n--- Final Metrics ---")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("ROC-AUC  :", roc_auc)

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(OUTPUT_PATH, "confusion_matrix.png"))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, -scores)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(os.path.join(OUTPUT_PATH, "roc_curve.png"))
    plt.close()

    # Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, -scores)
    plt.figure()
    plt.plot(recall_curve, precision_curve)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(os.path.join(OUTPUT_PATH, "pr_curve.png"))
    plt.close()

    # Anomaly Score Distribution
    plt.figure()
    plt.hist(scores, bins=50)
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(OUTPUT_PATH, "anomaly_distribution.png"))
    plt.close()

    print("\nAll plots saved in 'outputs/' folder.")