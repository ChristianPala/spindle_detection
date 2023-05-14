# Libraries:
import copy
from sklearn.preprocessing import StandardScaler
from config import DATA, MODELS
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, SVMSMOTE, ADASYN
import pickle
from src.global_models.paper_modeling_unbalanced import adjust_prediction
from tqdm import tqdm
from typing import Dict, Tuple, Any

# Constants:
seed = 42  # For reproducibility


# Classes:
class NoSampler:
    """
    Dummy sampler to conform to the other samplers in the pipeline.
    """

    def fit_resample(self, X, y) -> tuple:
        """
        Dummy fit_resample method to conform to the other samplers in the pipeline.
        """
        return X, y


class PatientModelTrainer:

    """
    This class trains a model for each patient in the dataset with a different sampler and model.
    """
    def __init__(self, patient_id, X, y) -> None:
        """
        The constructor of the PatientModelTrainer class.
        """
        self.patient_id = patient_id
        self.X = X[X.patient_id == patient_id].drop(columns=['patient_id'])
        self.y = y.spindle[X.patient_id == patient_id]
        self.seed = seed

        self.samplers = {'Raw Dataset': NoSampler(),
                         'Random Under Sampler': RandomUnderSampler(random_state=seed),
                         'Random Over Sampler': RandomOverSampler(random_state=seed),
                         'SMOTE': SMOTE(random_state=seed),
                         'SVM SMOTE': SVMSMOTE(random_state=seed),
                         'ADASYN': ADASYN(random_state=seed)}

        self.models = {'SVC': SVC(kernel='linear', random_state=seed),  # replicate the paper
                       'K-Nearest Neighbors': KNeighborsClassifier(),  # baseline
                       'Random Forest': RandomForestClassifier(random_state=seed),
                       'Gradient Boosting': GradientBoostingClassifier(random_state=seed)}

        self.require_standardization = [SVC, KNeighborsClassifier]
        self.models_dict = {}
        self.f1_score_dict = {}

    def train(self) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Any]]]:
        """
        This method trains a model for each patient in the dataset with a different sampler and model.
        """
        for sampler_name, sampler in self.samplers.items():
            self.models_dict[sampler_name] = {}
            self.f1_score_dict[sampler_name] = {}

            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=0.8, random_state=self.seed)

            # Standardize the data before resampling to preserve the distribution
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Resample the data
            X_train_res, y_train_res = copy.deepcopy(sampler).fit_resample(X_train, y_train)

            for model_name, model in self.models.items():
                model.fit(X_train_res, y_train_res)
                y_pred = model.predict(X_test)

                # Adjust predictions to match the paper by removing outlier positive predictions
                adj_y_pred = adjust_prediction(y_pred)
                adj_f_1 = f1_score(y_test, adj_y_pred)
                self.f1_score_dict[sampler_name][model_name] = adj_f_1
                self.models_dict[sampler_name][model_name] = model

                with open(os.path.join(MODELS, f'{self.patient_id}_{sampler_name}_{model_name}.pkl'),
                          'wb') as model_serialized:
                    pickle.dump(model, model_serialized)

        return self.models_dict, self.f1_score_dict


# Driver Code:
if __name__ == '__main__':
    # Load data
    features_file_name = os.path.join(DATA, 'features.csv')
    target_file_name = os.path.join(DATA, 'target.csv')

    X = pd.read_csv(features_file_name)
    y = pd.read_csv(target_file_name, index_col=0)

    # Train models
    models_dict = {}
    f1_score_dict = {}
    for patient_id in tqdm(X.patient_id.unique(), desc='Training Models', total=len(X.patient_id.unique())):
        trainer = PatientModelTrainer(patient_id, X, y)
        models, f1_scores = trainer.train()
        models_dict[patient_id] = models
        f1_score_dict[patient_id] = f1_scores

    # Save models and f1 scores
    with open(os.path.join(MODELS, 'all_models.pkl'), 'wb') as model_file:
        pickle.dump(models_dict, model_file)

    with open(os.path.join(MODELS, 'models_f1_eval.pkl'), 'wb') as f1_file:
        pickle.dump(f1_score_dict, f1_file)
