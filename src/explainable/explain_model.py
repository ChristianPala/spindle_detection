# Libraries
from eli5 import explain_weights, explain_prediction
from treeinterpreter import treeinterpreter as ti
from sklearn.base import BaseEstimator
import numpy as np


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

    def explain_weights(self) -> None:
        """
        This function explains the feature importances of the model.
        """
        print(explain_weights(self.model, feature_names=self.feature_names))

    def explain_prediction(self, instance: np.ndarray) -> None:
        """
        This function explains the prediction for a specific instance.
        """
        print(explain_prediction(self.model, instance, feature_names=self.feature_names))

    def interpret_tree(self, instance: np.ndarray) -> None:
        """
        This function explains the prediction for a specific instance using treeinterpreter for tree-based models.
        """
        if not hasattr(self.model, 'tree_'):
            print("Model is not tree-based.")
            return
        prediction, bias, contributions = ti.predict(self.model, instance)
        for i in range(len(instance)):
            print("Instance", i)
            print("Bias (trainset mean)", bias[i])
            print("Feature contributions:")
            for c, feature in sorted(zip(contributions[i], self.feature_names), key=lambda x: -abs(x[0])):
                print(feature, round(c, 2))
            print("-"*20)


if __name__ == '__main__':


    # Initialize the ModelExplainability class with the trained model and feature names
    explainer = ModelExplainability(svc_model, feature_names)

    # Explain the weights of the model
    explainer.explain_weights()

    # Explain the prediction for a specific instance
    explainer.explain_prediction(X_test[0])

    # Interpret the prediction for a specific instance (only for tree-based models)
    explainer.interpret_tree(X_test[0])