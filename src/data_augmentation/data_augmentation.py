from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, SVMSMOTE, ADASYN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
from xgboost import XGBClassifier

from config import DATA, MODELS
import os
from collections import Counter
import numpy as np
import pickle

from src.modeling.modeling import adjust_prediction


class NoSampler:
    def __init__(self):
        pass
    def fit_resample(self, X,y):
        return X, y
    
features_file_name = os.path.join(DATA, 'features.csv')
target_file_name = os.path.join(DATA, 'target.csv')

X = pd.read_csv(features_file_name)
y = pd.read_csv(target_file_name, index_col=0)
seed = 42

samplers = {'Raw Dataset':NoSampler(),'Random Under Sampler':RandomUnderSampler(random_state=seed), 'Random Over Sampler': RandomOverSampler(random_state=seed),'SMOTE': SMOTE(random_state=seed), 'SVM SMOTE': SVMSMOTE(random_state=seed),'ADASYN': ADASYN(random_state=seed)}
models = {'SVC': SVC(kernel='linear', random_state=seed), 'K-Nearest Neighbors': KNeighborsClassifier(), 'Random Forest': RandomForestClassifier(random_state=seed), 'Gradient Boosting':GradientBoostingClassifier(random_state=seed)}

models_dict = {}
f1_score_dict = {}
for patient_id in X.patient_id.unique():
    models_dict[patient_id] = {}
    f1_score_dict[patient_id] = {}
    pers_mask = X.patient_id == patient_id
    X_pers = X[pers_mask].drop(columns=['patient_id'])
    y_pers = y.spindle[pers_mask]
    print('-'*50)
    for sampler_name, sampler in samplers.copy().items(): # Each data has its samplers
        models_dict[patient_id][sampler_name]={}
        f1_score_dict[patient_id][sampler_name]={}
        X_sampled, y_sampled = sampler.fit_resample(X_pers, y_pers)
        print('-'*30)
        print(f'Patient {patient_id}, Distribution using {sampler_name}: {Counter(y_sampled)}')
        for model_name, model in models.copy().items():
            X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, train_size=0.8, random_state=seed)
            model.fit( X_train, y_train)
            y_pred=model.predict( X_test )
            adj_y_pred=adjust_prediction(y_pred)
            f_1=f1_score( y_test, y_pred, average='weighted' )
            adj_f_1=f1_score( y_test, adj_y_pred, average='weighted' )
            print('-'*10)
            print(f'{model_name} evaluation:')
            f1_score_dict[patient_id][sampler_name][model_name] = adj_f_1
            print(f'F1-Score: {f_1:.3f}')
            print( f'Adjusted F1-Score: {adj_f_1:.3f}' )
            print( classification_report( y_test, y_pred ) )
            models_dict[patient_id][sampler_name][model_name] = model
            with open( os.path.join(MODELS, f'{patient_id}_{sampler_name}_{model_name}.pkl'), 'wb' ) as model_file:
                pickle.dump( model, model_file )

with open( os.path.join(MODELS, 'all_models.pkl'), 'wb') as model_file:
    pickle.dump(models_dict, model_file)

with open( os.path.join(MODELS, 'models_f1_eval.pkl'), 'wb') as f1_file:
    pickle.dump(f1_score_dict, f1_file)
