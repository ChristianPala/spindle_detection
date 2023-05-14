# project root and directories:
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
MODELS = os.path.join(ROOT, 'models')
EXCERPTS = os.path.join(DATA, 'excerpts')
HYPNOGRAMS = os.path.join(DATA, 'hypnograms')
# scoring is in 2 different folders
SCORING = [os.path.join(DATA, 'scoring_1'), os.path.join(DATA, 'scoring_2')]
FEATURES = os.path.join(DATA, 'features')
RESULTS = os.path.join(ROOT, 'results')