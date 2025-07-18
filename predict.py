import pandas as pd
import argparse
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import get_models
from sklearn.externals import joblib

def predict(data_path):
    """
    Makes predictions on new data.
    """
   
    
    # from src.train import train_and_evaluate
    # _, X_train, _, _, y_train, _, _ = preprocess_data(load_data('data/thyroidDF.csv'))
    # model = get_models()['Random Forest']
    # model.fit(X_train, y_train)
    # joblib.dump(model, 'random_forest_model.pkl')

    # Load the trained model
    # model = joblib.load('random_forest_model.pkl')
    # df = load_data(data_path)
    # _, X_new, _, _, _, _, _ = preprocess_data(df)
    # predictions = model.predict(X_new)
    # print(predictions)
    print("Prediction script is a placeholder. You need to train and save a model first.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the new data CSV file.')
    args = parser.parse_args()
    predict(args.data_path)
