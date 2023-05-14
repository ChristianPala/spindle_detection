# Driver code:
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from config import DATA
from src.global_models.paper_modeling_unbalanced import DataHandler, GlobalModel

if __name__ == '__main__':
    # Load the data
    X = pd.read_csv(os.path.join(DATA, 'features.csv'))
    y = pd.read_csv(os.path.join(DATA, 'target.csv'))

    # Preprocess the data
    data_handler = DataHandler(X, y)
    X_train, y_train, X_val, y_val, X_test, y_test = data_handler.train_val_test_split_ratio()

    models = {'KNN': KNeighborsClassifier(n_neighbors=5),
              'RF': RandomForestClassifier(n_estimators=100, random_state=42),
              'GB': GradientBoostingClassifier(n_estimators=100, random_state=42)}

    for name, model in models.items():
        print(f"Global model with {name}:")
        model = GlobalModel(model, X_train, y_train, X_test, y_test)
        report = model.train_eval_model(filename=f'Global_Raw Dataset_{name}_confusion_matrix',
                                        print_confusion_matrix=False)
        print(f"\nMetrics for the majority class with {name}:")
        print(f"Precision: {report['0']['precision']: .3f}")
        print(f"Recall: {report['0']['recall']: .3f}")
        print(f"F1-score: {report['0']['f1-score']: .3f}")

