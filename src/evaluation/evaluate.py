# Libraries:
import pickle
from config import MODELS
import os

# Constants:
NR_PATIENTS = 8


# Functions:
def process_tuning_model_scores(file_name: str) -> None:
    """
    This function prints the best sampler and model combination for each patient.
    """
    # read scores from the pickle file
    with open(os.path.join(MODELS, file_name), 'rb') as score_file:
        score_dict = pickle.load(score_file)

    print(f'Base F1 Minority: {score_dict["base_f1_minority"]:.3f}')
    print( f'Fine-Tuned F1 Minority: {score_dict["fine_tuned_f1_minority"]:.3f}' )
    print(f'Improvement Minority: {(((score_dict["fine_tuned_f1_minority"] - score_dict["base_f1_minority"])/score_dict["base_f1_minority"]) * 100):.1f}%')
    print( f'Fine-Tuned F1 Macro: {score_dict["fine_tuned_f1_macro"]:.3f}' )
    print('-'*10)


def process_tuning_patient_model_scores(file_name: str) -> None:
    """
    This function prints the best sampler and model combination for each patient.
    """
    # read scores from the pickle file
    with open(os.path.join(MODELS, file_name), 'rb') as score_file:
        score_dict = pickle.load(score_file)

    # for each patient,
    for patient_id, scores in score_dict.items():
        print(f'Patient {patient_id}:')
        print(f'Base F1 Minority: {scores["base_f1_minority"]:.3f}')
        print( f'Fine-Tuned F1 Minority: {scores["fine_tuned_f1_minority"]:.3f}' )
        print(f'Improvement Minority: {(((scores["fine_tuned_f1_minority"] - scores["base_f1_minority"])/scores["base_f1_minority"]) * 100):.1f}%')
        print( f'Fine-Tuned F1 Macro: {scores["fine_tuned_f1_macro"]:.3f}' )
        print('-'*10)


if __name__ == '__main__':
    process_tuning_patient_model_scores('best_models_eval.pkl')
