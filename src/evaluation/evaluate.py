# Libraries:
import pickle
from config import MODELS
import os

# Constants:
NR_PATIENTS = 8


# Functions:
def process_model_f1_scores(file_name: pickle) -> None:
    """
    This function prints the best sampler and model combination for each patient.
    """
    # read f1 scores from the pickle file
    with open(os.path.join(MODELS, file_name), 'rb') as f1_file:
        f1_score_dict = pickle.load(f1_file)

    # for each patient, print the best sampler and model combination and the corresponding f1 score achieved.
    for patient_id, f1_scores in f1_score_dict.items():
        print(f'Patient {patient_id} best sampler and model combination:')

        flat_scores = {(sampler, model): score for sampler, models in f1_scores.items()
                       for model, score in models.items()}

        (best_sampler, best_model), best_score = max(flat_scores.items(), key=lambda item: item[1])

        print(f'Best sampler: {best_sampler}')
        print(f'Best model: {best_model}')
        print(f'Best score: {best_score:.3f}\n')


if __name__ == '__main__':
    process_model_f1_scores('models_f1_eval.pkl')
