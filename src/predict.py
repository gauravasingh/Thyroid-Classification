import pandas as pd
import joblib
import argparse
from data_loader import load_data
from preprocessing import preprocess_data

def predict(model_path, data_path):
    """
    Loads a trained model and makes predictions on new data.

    Args:
        model_path (str): The path to the saved .pkl model file.
        data_path (str): The path to the new data in CSV format.
    """
    try:
        # --- 1. Load Model and Data ---
        print(f"🔄 Loading model from {model_path}...")
        model = joblib.load(model_path)
        print(f"🔄 Loading data from {data_path}...")
        new_df = load_data(data_path)

        # --- 2. Preprocess the New Data ---
        # Note: The 'y' values will be dummy placeholders as they are unknown.
        # The preprocessing function is reused to ensure consistency.
        print("⚙️  Preprocessing new data...")
        _, _, _, X_new, _, _, _ = preprocess_data(new_df)


        # --- 3. Make Predictions ---
        print("🧠 Making predictions...")
        predictions = model.predict(X_new)

        # Map numeric predictions back to class names
        class_mapping = {
            0: 'No Condition',
            1: 'Hyperthyroid',
            2: 'Hypothyroid',
            3: 'Binding Protein',
            4: 'General Health',
            5: 'Replacement Therapy',
            6: 'Miscellaneous'
        }
        predicted_classes = [class_mapping[p] for p in predictions]

        # --- 4. Display Results ---
        result_df = pd.DataFrame({
            'Patient_ID': new_df['patient_id'],
            'Predicted_Class': predicted_classes
        })
        print("\n✅ Prediction Complete!")
        print("\nSample of Predictions:")
        print(result_df.head(10))
        
        # Print class distribution
        print("\nPrediction Distribution:")
        distribution = pd.Series(predicted_classes).value_counts()
        for class_name, count in distribution.items():
            percentage = (count / len(predicted_classes)) * 100
            print(f"{class_name}: {count} ({percentage:.2f}%)")

    except FileNotFoundError:
        print(f"❌ ERROR: Model file not found at '{model_path}'.")
        print("👉 Please run the training script first: `python src/train.py`")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Thyroid Disease Prediction Script")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pkl file (e.g., models/Random_Forest.pkl).')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the new data CSV file for prediction.')
    args = parser.parse_args()

    predict(args.model_path, args.data_path)