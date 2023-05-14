import copy
import os
import pickle
import warnings
from typing import Any, Tuple
import joblib
import optuna
from imblearn.over_sampling import ADASYN, SMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from config import DATA, MODELS
from src.personalized_models.personalized_models import PatientModelTrainer, NoSampler
from src.global_models.paper_modeling_unbalanced import adjust_prediction

np.random.seed(42)  # For reproducibility
seed = 42  # For reproducibility


class ModelTuning:
    def __init__(self, X, y):
        self.study_name = f'global'
        self.seed = seed
        self.X = X
        if isinstance(y, pd.DataFrame):
            self.y = y.spindle
        else:
            self.y = y
        self.model_classes = {'SVC': SVC,  # replicate the paper
                              'K-Nearest Neighbors': KNeighborsClassifier,  # baseline
                              'Random Forest': RandomForestClassifier,
                              'Gradient Boosting': GradientBoostingClassifier}

    def define_configuration(self, trial: optuna.trial.Trial, model_name: str) -> dict:
        if model_name == 'SVC':
            return {'C': trial.suggest_categorical('C', [0.1, 1, 10, 100]),
                    'gamma': trial.suggest_float('gamma', 1e-1, 1, log=True),
                    'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
                    'random_state': self.seed}

        if model_name == 'K-Nearest Neighbors':
            return {'n_neighbors': trial.suggest_int('n_neighbors', 3, 15, 2),
                    'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                    'metric': trial.suggest_categorical('metric', ['minkowski', 'euclidean', 'manhattan']),
                    'leaf_size': trial.suggest_int('leaf_size', 15, 50)}

        if model_name == 'Random Forest':
            return {'n_estimators': trial.suggest_int('n_estimators', 50, 1000, 50),
                    'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
                    'max_depth': trial.suggest_int('max_depth', 10, 100, 5),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'min_samples_split': trial.suggest_int('min_samples_split', 10, 200, 10),
                    'random_state': self.seed}

        if model_name == 'Gradient Boosting':
            return {'n_estimators': trial.suggest_int('n_estimators', 50, 1000, 50),
                    'loss': trial.suggest_categorical('loss', ['log_loss', 'exponential']),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1, log=True),
                    'max_depth': trial.suggest_int('max_depth', 10, 100, 5),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 200),
                    'min_samples_split': trial.suggest_int('min_samples_split', 5, 200, 5),
                    'random_state': self.seed}

        return {}

    def tune_model(self, trial: optuna.trial.Trial, X_train: pd.DataFrame, y_train: np.array, model_name: str) -> float:
        # extract proper configuration for the trial based on the model
        cfg = self.define_configuration(trial, model_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt_model: Any = self.model_classes[model_name](**cfg)
            kfold = StratifiedKFold(n_splits=5, shuffle=True)
            kf_cv_scores = cross_val_score(opt_model, X_train, y_train, cv=kfold, scoring='f1', n_jobs=-1)
        score_mean = kf_cv_scores.mean()
        return score_mean

    def optimize(self, sampler, model_name) -> Tuple[Any, float, float]:
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=0.8, random_state=self.seed)
        # Standardize the data before resampling to preserve the distribution
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Resample the data
        X_train_res, y_train_res = copy.deepcopy(sampler).fit_resample(X_train, y_train)

        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=self.seed), direction='maximize')
        study.optimize(func=lambda trial: self.tune_model(trial, X_train_res, y_train_res, model_name), n_trials=25)
        joblib.dump(study, f'studies/{self.study_name}_best_analysis.pkl')
        best_model = self.model_classes[model_name](**study.best_params)
        best_model.fit(X_test, y_test)
        adj_y_pred = adjust_prediction(best_model.predict(X_test))
        f1_minority = f1_score(y_test, adj_y_pred, pos_label=1)  # assuming minority class is labeled as '1'
        f1_macro = f1_score(y_test, adj_y_pred, average='macro')
        return best_model, f1_minority, f1_macro


class PersonalizedModelTuning(PatientModelTrainer, ModelTuning):
    def __init__(self, patient_id, X, y):
        PatientModelTrainer.__init__(self, patient_id, X, y)
        ModelTuning.__init__(self, self.X, self.y)
        self.study_name = f'patient_{self.patient_id}'


def personalized_hyperparameter_tuning(X, y):
    minority_config = {1: {'sampler': ADASYN(), 'model_name': 'SVC', 'f1_minority': 0.415},
                       2: {'sampler': ADASYN(), 'model_name': 'Gradient Boosting', 'f1_minority': 0.317},
                       3: {'sampler': SMOTE(), 'model_name': 'SVC', 'f1_minority': 0.317},
                       4: {'sampler': SMOTE(), 'model_name': 'SVC', 'f1_minority': 0.292},
                       5: {'sampler': RandomUnderSampler(), 'model_name': 'Random Forest', 'f1_minority': 0.361},
                       6: {'sampler': ADASYN(), 'model_name': 'K-Nearest Neighbors', 'f1_minority': 0.301},
                       7: {'sampler': SVMSMOTE(), 'model_name': 'K-Nearest Neighbors', 'f1_minority': 0.187},
                       8: {'sampler': RandomUnderSampler(), 'model_name': 'Gradient Boosting', 'f1_minority': 0.330}}
    # Train models
    models_dict = {}
    f1_score_dict = {}
    for patient_id in tqdm(X.patient_id.unique(), desc='Training Models', total=len(X.patient_id.unique())):
        f1_score_dict[patient_id] = {}
        trainer = PersonalizedModelTuning(patient_id, X, y)
        models_dict[patient_id], f1_score_dict[patient_id]['fine_tuned_f1_minority'], f1_score_dict[patient_id][
            'fine_tuned_f1_macro'] = \
            trainer.optimize(minority_config[patient_id]['sampler'], minority_config[patient_id]['model_name'])
        f1_score_dict[patient_id]['base_f1_minority'] = minority_config[patient_id]['f1_minority']
    # Save models and f1 scores
    with open(os.path.join(MODELS, 'all_best_models.pkl'), 'wb') as model_file:
        pickle.dump(models_dict, model_file)

    with open(os.path.join(MODELS, 'best_models_eval.pkl'), 'wb') as f1_file:
        pickle.dump(f1_score_dict, f1_file)


# Driver Code:
if __name__ == '__main__':
    # Load data
    features_file_name = os.path.join(DATA, 'features.csv')
    target_file_name = os.path.join(DATA, 'target.csv')

    X = pd.read_csv(features_file_name)
    y = pd.read_csv(target_file_name, index_col=0)

    # Personalized hyperparameter tuning
    personalized_hyperparameter_tuning(X, y)
