# Libraries
from typing import Tuple
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
import pickle
from config import DATA

np.random.seed(42)  # For reproducibility


# Functions
def adjust_prediction(y_pred: np.ndarray) -> np.ndarray:
    """
    This function post-processes the predictions of the model to ensure that there are no isolated spindle window
    segments, following the approach of the paper.
    """
    for idx in range(1, len(y_pred) - 1):
        if y_pred[idx] == 0:
            continue
        if y_pred[idx - 1] + y_pred[idx + 1] <= 0:
            y_pred[idx] = 0
    y_pred[0] = y_pred[1]
    y_pred[-1] = y_pred[-2]
    return y_pred


# Classes
class DataHandler:
    """
    This class handles the data for the global model.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y


    def train_val_test_split_ratio(self, train_ratio=0.6, val_ratio=0.2) \
            -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
        """
        This function splits the data into training, validation and test sets, ensuring no patient is in more than one set.
        """
        patient_ids = self.X['patient_id'].unique()
        patient_size = len(patient_ids)
        np.random.shuffle(patient_ids)
        train_patients = patient_ids[:int(patient_size * train_ratio)]
        val_cap = int(patient_size * train_ratio + patient_size * val_ratio)
        val_patients = patient_ids[int(patient_size * train_ratio):val_cap]
        test_patients = patient_ids[val_cap:]
        train_mask = self.X['patient_id'].isin(train_patients)
        val_mask = self.X['patient_id'].isin(val_patients)
        test_mask = self.X['patient_id'].isin(test_patients)

        # Split the data into training and test sets using the masks
        X_train = self.X[train_mask]
        y_train = self.y[train_mask]['spindle'].values
        X_val = self.X[val_mask]
        y_val = self.y[val_mask]['spindle'].values
        X_test = self.X[test_mask]
        y_test = self.y[test_mask]['spindle'].values
        return X_train, y_train, X_val, y_val, X_test, y_test


class GlobalModel:
    """
    This class implements the global models for our spindle detection system.
    """
    def __init__(self, model, X_train, y_train, X_test, y_test) -> None:
        """
        Constructor for the GlobalModel class.
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.scaler = StandardScaler()

    def train_eval_model(self, print_confusion_matrix=False):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        # Post-process the predictions
        y_pred = adjust_prediction(y_pred)
        f_1 = f1_score(self.y_test, y_pred, average='weighted')
        print(classification_report(self.y_test, y_pred))
        print(f"Weighted f-1 score: {f_1:.3f}")
        if print_confusion_matrix:
            cm = confusion_matrix(self.y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Blues)
            # save the confusion matrix
            plt.savefig(os.path.join(DATA, 'global_raw_data_svc.png'))
            plt.show()

    def save_model(self, filename):
        # Save the model to disk in the models folder
        pickle.dump(self.model, open(os.path.join(DATA, filename), 'wb'))


# Driver code:
if __name__ == '__main__':
    # Load the data
    X = pd.read_csv(os.path.join(DATA, 'features.csv'))
    y = pd.read_csv(os.path.join(DATA, 'target.csv'))

    # Preprocess the data
    data_handler = DataHandler(X, y)
    X_train, y_train, X_val, y_val, X_test, y_test = data_handler.train_val_test_split_ratio()

    # Train and evaluate the support vector machine model on the raw dataset and save it
    svc_model = GlobalModel(SVC(kernel='linear'), X_train, y_train, X_test, y_test)
    svc_model.train_eval_model(print_confusion_matrix=True)
    svc_model.save_model('global_Raw Dataset_SVC.pkl')
