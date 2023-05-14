# Libraries:
import pickle
from config import MODELS
import os

# Constants:
NR_PATIENTS = 8


# Functions:
def process_model_scores(file_name: str, score_name: str) -> None:
    """
    This function prints the best sampler and model combination for each patient.
    """
    # read scores from the pickle file
    with open(os.path.join(MODELS, file_name), 'rb') as score_file:
        score_dict = pickle.load(score_file)

    # for each patient, print the best sampler and model combination and the corresponding score achieved.
    for patient_id, scores in score_dict.items():
        print(f'Patient {patient_id} best sampler and model combination based on {score_name}:')

        flat_scores = {(sampler, model): metrics[score_name] for sampler, models in scores.items()
                       for model, metrics in models.items()}

        (best_sampler, best_model), best_score = max(flat_scores.items(), key=lambda item: item[1])

        print(f'Best sampler: {best_sampler}')
        print(f'Best model: {best_model}')
        print(f'Best {score_name}: {best_score:.3f}\n')


if __name__ == '__main__':
    process_model_scores('models_eval.pkl', 'F1 (minority)')
    process_model_scores('models_eval.pkl', 'F1 (average)')
    process_model_scores('models_eval.pkl', 'AUC-ROC')

