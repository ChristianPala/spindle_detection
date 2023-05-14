# Libraries
from sklearn.base import BaseEstimator


class ModelExplainability:
    """
    This class implements methods to explain the predictions of various types of models.
    """
    def __init__(self, model: BaseEstimator, feature_names: list) -> None:
        """
        Constructor for the ModelExplainability class.
        """
        self.model = model
        self.feature_names = feature_names
