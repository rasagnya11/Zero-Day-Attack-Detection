from src.preprocess import run_preprocessing
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    print("Starting preprocessing...")
    run_preprocessing()

    print("\nTraining model...")
    train_model()

    print("\nEvaluating model...")
    evaluate_model()

if __name__ == "__main__":
    main()