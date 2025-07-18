import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, fbeta_score
from src.model import get_models

def train_and_evaluate(X_train, y_train, X_val, y_val):
    """
    Trains and evaluates multiple classifiers.

    Args:
        X_train (pandas.DataFrame): Training features.
        y_train (pandas.Series): Training labels.
        X_val (pandas.DataFrame): Validation features.
        y_val (pandas.Series): Validation labels.
    """
    classifiers = get_models()
    evaluation_metrics = {
        'Model': [], 'Accuracy': [], 'F1 Score': [], 'Precision': [], 'Recall': [], 'F2 Score': []
    }

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred_val = clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred_val)
        f1 = f1_score(y_val, y_pred_val, average='weighted')
        precision = precision_score(y_val, y_pred_val, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred_val, average='weighted')
        f2 = fbeta_score(y_val, y_pred_val, beta=2, average='weighted')

        evaluation_metrics['Model'].append(name)
        evaluation_metrics['Accuracy'].append(accuracy)
        evaluation_metrics['F1 Score'].append(f1)
        evaluation_metrics['Precision'].append(precision)
        evaluation_metrics['Recall'].append(recall)
        evaluation_metrics['F2 Score'].append(f2)

    metrics_df = pd.DataFrame(evaluation_metrics)
    print("Model Evaluation Metrics:")
    print(metrics_df)
