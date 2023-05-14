# Libraries:
import os
from typing import Tuple
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from config import DATA
from src.global_models.paper_modeling_unbalanced import adjust_prediction

# Constants:
RANDOM_SEED = 42  # For reproducibility


# Functions:
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function loads the data and returns the features and target as pandas DataFrames.
    """
    X = pd.read_csv(os.path.join(DATA, 'features.csv'))
    y = pd.read_csv(os.path.join(DATA, 'target.csv'))
    return X, y


def train_val_test_split_ratio(X: pd.DataFrame, y: pd.DataFrame,
                               train_ratio: float = 5 / 8,
                               val_ratio: float = 1 / 8) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    This function splits the data into training, validation, and test sets, ensuring no patient
    is in more than one set. We use a train:val:test ratio.
    """
    patient_ids = X['patient_id'].unique()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(patient_ids)

    # Calculate the cut-off indices for the train and validation sets
    train_idx = int(len(patient_ids) * train_ratio)
    val_idx = train_idx + int(len(patient_ids) * val_ratio)

    # Split the patients into train, val and test sets and create the masks
    train_patients = patient_ids[:train_idx]
    val_patients = patient_ids[train_idx:val_idx]
    test_patients = patient_ids[val_idx:]
    train_mask = X['patient_id'].isin(train_patients)
    val_mask = X['patient_id'].isin(val_patients)
    test_mask = X['patient_id'].isin(test_patients)

    # Return the splits
    return (X[train_mask], y[train_mask]['spindle'],
            X[val_mask], y[val_mask]['spindle'],
            X[test_mask], y[test_mask]['spindle'])


def objective(trial, X, y) -> float:
    # Define the hyperparameter search space
    svc_c = trial.suggest_float('svc_c', 1e-10, 1e10, log=True)
    svc_gamma = trial.suggest_float('svc_gamma', 1e-10, 1e10, log=True)
    svc_kernel = trial.suggest_categorical('svc_kernel', ['linear'])  # linear is always better

    # Create the SVC model with the current hyperparameters
    model = SVC(C=svc_c, gamma=svc_gamma, kernel=svc_kernel, random_state=42)

    # Create the pipeline with RandomUnderSampler, StandardScaler and SVC
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rus', RandomUnderSampler(random_state=42)),
        ('model', model),
    ])

    # Return the mean cross-validated score
    return cross_val_score(pipeline, X, y, cv=5, scoring='f1').mean()  # change scoring here ('f1_macro', 'roc_auc')


if __name__ == '__main__':
    # Load the dataset
    X, y = load_data()
    # Split the data into training, validation and test sets
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_ratio(X, y)
    # Combine the training and validation sets
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    # Drop the patient_id column
    X_train = X_train.drop(columns=['patient_id'])
    X_train_val = X_train_val.drop(columns=['patient_id'])
    X_val = X_val.drop(columns=['patient_id'])
    X_test = X_test.drop(columns=['patient_id'])

    # ##################################################################################################3
    # # Create the study
    # study = optuna.create_study(direction='maximize', study_name='svc_hyperparameter_optimization')
    #
    # # Optimize the study
    # study.optimize(lambda trial: objective(trial, X_train_val, y_train_val), n_trials=10, n_jobs=1,
    #               timeout=300)

    # get the best hyperparameters
    # best_params = study.best_params

    # Create the SVC model with the best hyperparameters
    # model = SVC(C=best_params['svc_c'], gamma=best_params['svc_gamma'], kernel=best_params['svc_kernel'],
    #            random_state=42)

    # Best hyperparameters after a few studies
    model = SVC(C=0.0006090634636441789, gamma=0.00046809544352621393, kernel='linear', random_state=42)

    # ##############################################################################################
    # Test on the test set
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rus', RandomUnderSampler(random_state=42)),
        ('model', model),
    ])

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Test the pipeline
    y_pred = pipeline.predict(X_test)

    # Adjust the predictions
    y_pred = adjust_prediction(y_pred)

    # F1 score
    # Binary
    f1 = f1_score(y_test, y_pred)
    # Macro
    f1_m = f1_score(y_test, y_pred, average='macro')
    # Weighted
    f1_w = f1_score(y_test, y_pred, average='weighted')

    print(f'Global Optimized F1 score (binary) on SVC: {f1: .3f}')
    print(f'Global Optimized F1 score (macro) on SVC: {f1_m: .3f}')
    print(f'Global Optimized F1 score (weighted) on SVC: {f1_w: .3f}')

