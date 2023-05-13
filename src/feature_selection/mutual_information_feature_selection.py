# Libraries:
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import os
from config import DATA

# Constants:
THRESHOLD = 70
NUMBER_OF_PATIENTS = 8


# Classes:
class MutualInformationFeatureSelector(BaseEstimator, TransformerMixin):
    """
    This class applies Mutual Information based feature selection to a dataset.
    """

    def __init__(self, percentile: int):
        self.percentile = percentile
        self.selector_ = SelectPercentile(mutual_info_classif, percentile=self.percentile)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.selector_.fit(X, y)
        return self

    def transform(self, X: np.ndarray):
        check_is_fitted(self)
        return self.selector_.transform(X)


# Function to load and preprocess data:
def load_and_preprocess_data(patient_id=None):
    X = pd.read_csv(os.path.join(DATA, 'features.csv'))
    y = pd.read_csv(os.path.join(DATA, 'target.csv'), index_col=0)

    if patient_id is not None:
        X = X[X['patient_id'] == patient_id]
        y = y[y['patient_id'] == patient_id]

    y = y['spindle'].to_numpy()

    X = X.drop(columns=['patient_id'])
    return X.to_numpy(), y, X.columns


# Driver code:
if __name__ == '__main__':
    # Global feature selection
    X, y, features = load_and_preprocess_data()

    selector = MutualInformationFeatureSelector(percentile=THRESHOLD)
    selector.fit(X, y)

    selected_features = features[selector.selector_.get_support()]
    print(f"Globally selected features using Mutual Information with a threshold of {THRESHOLD}%:")
    [print(feature) for feature in selected_features]
    print('-'*50)

    # Personalized feature selection for each patient
    for i in range(NUMBER_OF_PATIENTS):
        X, y, features = load_and_preprocess_data(patient_id=i)
        selector.fit(X, y)
        selected_features = features[selector.selector_.get_support()]
        print(f"Personalized features using Mutual Information with a threshold of {THRESHOLD}% for patient {i+1}:")
        [print(feature) for feature in selected_features]
        print('-' * 50)

