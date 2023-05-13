from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd

from config import DATA

np.random.seed(42)
import os

def adjust_prediction(y_pred):
    for idx in range(1,len(y_pred)-1):
        if y_pred[idx] == 0:
            continue
        if y_pred[idx-1] + y_pred[idx + 1] <= 0:
            y_pred[idx] = 0
    y_pred[0] = y_pred[1]
    y_pred[-1]=y_pred[-2]
    return y_pred

def train_val_test_split_ratio(X, y, train_ratio=0.6, val_ratio=0.2):
    patient_ids= X['patient_id'].unique()
    patient_size = len(patient_ids)
    np.random.shuffle( patient_ids )
    train_patients=patient_ids[:int(patient_size*train_ratio)]
    val_cap = int(patient_size*train_ratio+patient_size*val_ratio)
    val_patients=patient_ids[int(patient_size*train_ratio):val_cap]
    test_patients=patient_ids[val_cap:]
    train_mask=X['patient_id'].isin( train_patients )
    val_mask=X['patient_id'].isin( val_patients )
    test_mask=X['patient_id'].isin( test_patients )

    # Split the data into training and test sets using the masks
    X_train=X[train_mask]
    y_train=y[train_mask]['Target'].values
    X_val=X[val_mask]
    y_val=y[val_mask]['Target'].values
    X_test=X[test_mask]
    y_test=y[test_mask]['Target'].values
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_eval_model(model, X_train, y_train, X_test, y_test, print_confusion_matrix=False):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f_1 = f1_score(y_test, y_pred, average='weighted')
    print(classification_report(y_test, y_pred))
    print(f"Weighted f-1 score: {f_1:.3f}")
    if print_confusion_matrix:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()

if __name__ == '__main__':

    X = pd.read_csv( os.path.join( DATA, '../data/features.csv' ) ).drop(columns=['inst_freq'])
    y = pd.read_csv( os.path.join( DATA, '../data/target.csv' ) )
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_ratio(X,y)
    svc = SVC(kernel='linear')
    svc.fit( X_train, y_train )
    y_pred=svc.predict( X_test )
    f_1=f1_score( y_test, y_pred, average='weighted' )
    print( classification_report( y_test, y_pred ) )


    ############################
    train_eval_model(svc, X_train, y_train, X_test, y_test)