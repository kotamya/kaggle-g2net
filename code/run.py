# import numpy as np

# from feat import CreateFeatures
from model_rnn import ModelRNN
from runner import Runner
from util import Util, Submission


if __name__ == '__main__':

    params_rnn = {
                'use_multiprocessing': True,
                'wokers': 4,
                'epochs': 10,
                }

    # learning/prediction by lightGBM and create submission file
    run_name = 'rnn'
    runner = Runner(run_name, ModelRNN, params_rnn)
    runner.run_train_cv()
    runner.run_predict_cv()
    Submission.create_submission(run_name)
