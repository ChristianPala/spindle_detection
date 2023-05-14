# Libraries:
import os

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from skfeature.function.information_theoretical_based import MRMR

from config import RESULTS
from mutual_information_feature_selection import load_and_preprocess_data


# Classes:
class MRMRFeatureSelector(BaseEstimator, TransformerMixin):
    """
    This class applies MRMR feature selection to a dataset.
    """

    def __init__(self, n_features_to_select: int):
        """
        The constructor of the MRMRFeatureSelector class.
        :param n_features_to_select: The number of features to select.
        """
        self.n_features_to_select = n_features_to_select
        self.selected_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This method applies MRMR feature selection to the dataset.
        :param X: The feature matrix.
        :param y: The target vector.
        :return: self
        """
        # Ensure y is an integer type
        y = y.astype(int)

        # Apply MRMR feature selection
        selected_feature_indices = MRMR.mrmr(X, y, n_selected_features=X.shape[1])

        # Save the top n_features_to_select
        self.selected_features_ = selected_feature_indices[:self.n_features_to_select]

        return self

    def transform(self, X: np.ndarray):
        """
        This method returns a new dataset containing only the selected features.
        :param X: The feature matrix.
        :return: The new feature matrix.
        """
        # Check if fit has been called
        check_is_fitted(self)

        # Return only the selected features
        return X[:, self.selected_features_]

    def fit_transform(self, X, y=None, **fit_params):
        """
        This method applies fit and transform in one step.
        :param X: The feature matrix.
        :param y: The target vector.
        :return: The new feature matrix.
        """
        return self.fit(X, y).transform(X)


# Driver code:
if __name__ == '__main__':

    # Load the dataset
    X, y, features = load_and_preprocess_data()

    selector = MRMRFeatureSelector(X.shape[1])  # Rank all features with the same method as the paper.

    # Apply feature selection
    X_new = selector.fit_transform(X, y)

    sorted_indices = selector.selected_features_
    sorted_features = features[sorted_indices]

    with open(os.path.join(RESULTS, "global_feature_rankings_MRMRF.txt"), 'w') as f:
        f.write("Ranking of the global features using MRMRF:\n")
        for rank, feature in enumerate(sorted_features, start=1):
            f.write(f"{feature}: {rank}\n")