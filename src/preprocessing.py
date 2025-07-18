import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN

def preprocess_data(df):
    """
    Performs preprocessing on the thyroid dataset.

    Args:
        df (pandas.DataFrame): The input dataframe.

    Returns:
        tuple: A tuple containing the preprocessed data and the split datasets.
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

    # Target class mapping
    mapping = {
        'A': 'Hyperthyroid', 'B': 'Hyperthyroid', 'C': 'Hyperthyroid', 'D': 'Hyperthyroid', 'AK': 'Hyperthyroid',
        'E': 'Hypothyroid', 'F': 'Hypothyroid', 'G': 'Hypothyroid', 'H': 'Hypothyroid', 'GK': 'Hypothyroid',
        'GI': 'Hypothyroid', 'FK': 'Hypothyroid', 'GKJ': 'Hypothyroid',
        'I': 'Binding Protein', 'J': 'Binding Protein', 'C I': 'Binding Protein',
        'K': 'General Health', 'KJ': 'General Health', 'H|K': 'General Health',
        'M': 'Replacement Therapy', 'L': 'Replacement Therapy', 'N': 'Replacement Therapy',
        'MK': 'Replacement Therapy', 'MI': 'Replacement Therapy', 'LJ': 'Replacement Therapy',
        'P': 'Miscellaneous', 'Q': 'Miscellaneous', 'OI': 'Miscellaneous',
        'R': 'Miscellaneous', 'S': 'Miscellaneous', 'T': 'Miscellaneous', 'DIR': 'Miscellaneous',
        '-': 'No Condition'
    }
    df['class'] = df['target'].map(mapping)

    # Drop rows with NaN values for specific features if class is 'No Condition'
    rows_to_drop = df[(df['TSH'].isnull() | df['T3'].isnull() | df['TT4'].isnull() | df['T4U'].isnull() | df['FTI'].isnull()) & (df['class'] == 'No Condition')]
    df.drop(rows_to_drop.index, inplace=True)

    # Impute missing values with the mean of age-binned groups
    df['age_group'] = pd.cut(df['age'], bins=10)
    for col in ['T3', 'TT4', 'TSH']:
        df[col] = df.groupby('age_group')[col].transform(lambda x: x.fillna(x.mean()))
    df.drop('age_group', axis=1, inplace=True)

    df['T4U'].fillna(df['T4U'].mean(), inplace=True)
    df['FTI'].fillna(df['FTI'].mean(), inplace=True)

    # One-hot encode categorical features
    data = pd.get_dummies(df, columns=['sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds', 'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'referral_source'], drop_first=True)

    # Splitting data
    X = data.drop(['class', 'target', 'patient_id'], axis=1)
    y = pd.Categorical(data['class'])
    y = pd.Categorical(y).codes  # Convert class labels to numeric codes
    y = y + 1  # Shift the codes to start from 0 instead of -1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Balancing classes with SMOTEENN
    smoteenn = SMOTEENN(random_state=42)
    X_train, y_train = smoteenn.fit_resample(X_train, y_train)

    # Further splitting the train set into train & validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    return data, X_train, X_val, X_test, y_train, y_val, y_test
