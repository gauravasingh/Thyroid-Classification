# Place this file in the 'src/' directory

import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score
from data_loader import load_data
from preprocessing import preprocess_data
from model import get_models

def train_and_save_models():
    """
    Main function to run the model training and evaluation pipeline.
    It saves the trained models to the 'models/' directory.
    """
    print("🚀 Starting the training pipeline...")

    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # --- 1. Load and Preprocess Data ---
    print("💾 Loading and preprocessing data...")
    df = load_data('data/thyroidDF.csv')
    _, X_train, X_val, _, y_train, y_val, _ = preprocess_data(df)
    print("✅ Data preprocessing complete.")

    # --- 2. Train, Evaluate, and Save Models ---
    classifiers = get_models()
    evaluation_results = []

    print("\n🏋️ Training and evaluating models...")
    for name, model in classifiers.items():
        print(f"--> Training {name}...")
        model.fit(X_train, y_train)

        # Evaluate on validation set
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        evaluation_results.append({'Model': name, 'Accuracy': accuracy, 'F1 Score': f1})
        print(f"    {name} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

        # Save the trained model
        model_filename = f"models/{name.replace(' ', '_')}.pkl"
        joblib.dump(model, model_filename)
        print(f"    ✅ Model saved to {model_filename}")

    # --- 3. Display Results ---
    metrics_df = pd.DataFrame(evaluation_results)
    print("\n📊 Model Evaluation Summary:")
    print(metrics_df.sort_values(by='F1 Score', ascending=False))

    print("\n🎉 Training pipeline finished successfully!")

if __name__ == '__main__':
    train_and_save_models()
