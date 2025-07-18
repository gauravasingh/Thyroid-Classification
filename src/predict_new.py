import pandas as pd
import joblib
import argparse
from data_loader import load_data

def preprocess_for_prediction(df):
    """
    Performs preprocessing on new data for prediction.
    """
    # Drop unnecessary columns
    df = df.drop(columns=['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured'], axis=1)

    # Age filtering
    df = df[df['age'] <= 100]

    # Handle missing values in 'sex'
    df['sex'].fillna(df['sex'].mode()[0], inplace=True)

    # TBG imputation
    replacement_values = {
        (1, 9, 'M'): 3.75, (10, 19, 'M'): 3.35,
        (1, 9, 'F'): 3.75, (10, 19, 'F'): 3.35,
        (20, 100, 'M'): 1.85, (20, 100, 'F'): 2.2
    }

    def replace_tbg_null(row):
        age = row['age']
        gender = row['sex']
        if pd.isnull(row['TBG']):
            if 1 <= age <= 9:
                return replacement_values[(1, 9, gender)]
            elif 10 <= age <= 19:
                return replacement_values[(10, 19, gender)]
            else:
                return replacement_values[(20, 100, gender)]
        else:
            return row['TBG']

    df['TBG'] = df.apply(replace_tbg_null, axis=1)

    # Impute missing values with mean
    numeric_columns = ['T3', 'TT4', 'TSH', 'T4U', 'FTI']
    for col in numeric_columns:
        df[col].fillna(df[col].mean(), inplace=True)

    # One-hot encode categorical features
    categorical_columns = [
        'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds',
        'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment',
        'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre',
        'tumor', 'hypopituitary', 'psych', 'referral_source'
    ]
    return pd.get_dummies(df, columns=categorical_columns, drop_first=True)

def predict(model_path, data_path):
    """
    Loads a trained model and makes predictions on new data.
    """
    try:
        # --- 1. Load Model and Data ---
        print(f"🔄 Loading model from {model_path}...")
        model = joblib.load(model_path)
        
        print(f"🔄 Loading data from {data_path}...")
        new_df = load_data(data_path)
        patient_ids = new_df['patient_id'].copy()

        # --- 2. Preprocess the New Data ---
        print("⚙️  Preprocessing new data...")
        processed_df = preprocess_for_prediction(new_df)
        
        # Create a mapping between patient_ids and processed data
        processed_df['patient_id'] = new_df['patient_id']
        
        # Drop non-feature columns for prediction
        X_new = processed_df.drop(['patient_id', 'target'], axis=1, errors='ignore')

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
            6: 'Miscellaneous',
            7: 'Other'
        }
        predicted_classes = [class_mapping[p] for p in predictions]

        # --- 4. Display Results ---
        result_df = pd.DataFrame({
            'Patient_ID': processed_df['patient_id'],
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

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Thyroid Disease Prediction Script")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pkl file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the new data CSV file for prediction')
    args = parser.parse_args()

    predict(args.model_path, args.data_path)
