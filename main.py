import pandas as pd
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train import train_and_evaluate

def main():
    """
    Main function to run the thyroid classification pipeline.
    """
    # Load data
    df = load_data('data/thyroidDF.csv')

    # Preprocess data
    data, X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df)

    # Train and evaluate models
    train_and_evaluate(X_train, y_train, X_val, y_val)

if __name__ == '__main__':
    main()
