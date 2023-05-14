# Libraries
import os
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
from paper_modeling_unbalanced import GlobalModel, DataHandler
from typing import Union, Dict
from config import DATA


# Classes
class BalancedGlobalModel(GlobalModel):
    """
    This class extends the GlobalModel to handle imbalanced datasets using undersampling and oversampling techniques.
    """

    def __init__(self, model, X_train, y_train, X_test, y_test, sampling_strategy=None) -> None:
        """
        Constructor for the BalancedGlobalModel class.
        """
        super().__init__(model, X_train, y_train, X_test, y_test)
        self.sampling_strategy = sampling_strategy

    def balance_data(self) -> None:
        """
        Balance the data using the specified sampling strategy.
        """
        if self.sampling_strategy == 'undersample':
            undersampler = RandomUnderSampler(random_state=42)
            self.X_train, self.y_train = undersampler.fit_resample(self.X_train, self.y_train)
        elif self.sampling_strategy == 'oversample':
            oversampler = RandomOverSampler(random_state=42)
            self.X_train, self.y_train = oversampler.fit_resample(self.X_train, self.y_train)

    def train_eval_model(self, filename: str, print_confusion_matrix=False) -> Union[str, Dict[str, Dict[str, float]]]:
        """
        This function balances the data before training and evaluating the model.
        """
        self.balance_data()
        return super().train_eval_model(filename, print_confusion_matrix)


# Driver code:
if __name__ == '__main__':
    # Load the data
    X = pd.read_csv(os.path.join(DATA, 'features.csv'))
    y = pd.read_csv(os.path.join(DATA, 'target.csv'))

    # Preprocess the data
    data_handler = DataHandler(X, y)
    X_train, y_train, X_val, y_val, X_test, y_test = data_handler.train_val_test_split_ratio()

    # Train and evaluate the support vector machine model on the raw dataset with undersampling
    print("Global model with undersampling:")
    svc_model_undersampled = BalancedGlobalModel(SVC(kernel='linear'), X_train, y_train, X_test, y_test, 'undersample')
    report_undersampled = svc_model_undersampled.train_eval_model \
        (filename='Global_Random Under Sampler_SVC_confusion_matrix', print_confusion_matrix=True)
    print("\nMetrics for the majority class with undersampling:")
    print(f"Precision: {report_undersampled['0']['precision']: .3f}")
    print(f"Recall: {report_undersampled['0']['recall']: .3f}")
    print(f"F1-score: {report_undersampled['0']['f1-score']: .3f}")

    # Save the model
    svc_model_undersampled.save_model('Global_Random Under Sampler_SVC')

    print("\nGlobal model with oversampling:")
    # Train and evaluate the support vector machine model on the raw dataset with oversampling
    svc_model_oversampled = BalancedGlobalModel(SVC(kernel='linear'), X_train, y_train, X_test, y_test, 'oversample')
    report_oversampled = svc_model_oversampled.train_eval_model \
        (filename='Global_Random Over Sampler_SVC_confusion_matrix', print_confusion_matrix=True)
    print("\nMetrics for the majority class with oversampling:")
    print(f"Precision: {report_oversampled['0']['precision']: .3f}")
    print(f"Recall: {report_oversampled['0']['recall']: .3f}")
    print(f"F1-score: {report_oversampled['0']['f1-score']: .3f}")

    # Save the model
    svc_model_undersampled.save_model('Global_Random Over Sampler_SVC')
