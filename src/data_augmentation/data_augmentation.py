# Libraries:
from collections import Counter
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
from src.modeling.modeling import adjust_prediction
from tqdm import tqdm

# Constants:
seed = 42  # For reproducibility


# Classes:
class NoSampler:
    """
    Dummy sampler to conform to the other samplers in the pipeline.
    """
    def fit_resample(self, X, y):
        return X, y


class PatientModelTrainer:
    def __init__(self, patient_id, X, y, seed=42):
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
        self.models = {'SVC': SVC(kernel='linear', random_state=seed),
                       'K-Nearest Neighbors': KNeighborsClassifier(),
                       'Random Forest': RandomForestClassifier(random_state=seed),
                       'Gradient Boosting': GradientBoostingClassifier(random_state=seed)}
        self.models_dict = {}
        self.f1_score_dict = {}

    def train(self):
        for sampler_name, sampler in self.samplers.items():
            self.models_dict[sampler_name] = {}
            self.f1_score_dict[sampler_name] = {}
            X_sampled, y_sampled = sampler.fit_resample(self.X, self.y)
            for model_name, model in self.models.items():
                X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, train_size=0.8,
                                                                    random_state=self.seed)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                adj_y_pred = adjust_prediction(y_pred)
                adj_f_1 = f1_score(y_test, adj_y_pred, average='weighted')
                self.f1_score_dict[sampler_name][model_name] = adj_f_1
                self.models_dict[sampler_name][model_name] = model
                with open(os.path.join(MODELS, f'{self.patient_id}_{sampler_name}_{model_name}.pkl'), 'wb') \
                        as model_file:
                    pickle.dump(model, model_file)

        return self.models_dict, self.f1_score_dict


if __name__ == '__main__':
    features_file_name = os.path.join(DATA, 'features.csv')
    target_file_name = os.path.join(DATA, 'target.csv')

    X = pd.read_csv(features_file_name)
    y = pd.read_csv(target_file_name, index_col=0)

    models_dict = {}
    f1_score_dict = {}
    for patient_id in tqdm(X.patient_id.unique(), desc='Training Models', total=len(X.patient_id.unique())):
        trainer = PatientModelTrainer(patient_id, X, y, seed)
        models, f1_scores = trainer.train()
        models_dict[patient_id] = models
        f1_score_dict[patient_id] = f1_scores

    with open(os.path.join(MODELS, 'all_models.pkl'), 'wb') as model_file:
        pickle.dump(models_dict, model_file)

    with open(os.path.join(MODELS, 'models_f1_eval.pkl'), 'wb') as f1_file:
        pickle.dump(f1_score_dict, f1_file)
