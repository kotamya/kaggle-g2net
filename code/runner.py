import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from typing import Callable, Tuple, Union, Optional

from model import Model
from util import Logger, Util, CustomDataset

logger = Logger()


class Runner:

    def __init__(self, run_name: str, model_cls: Callable[[str, dict], Model],
                 params: dict):
        ''' Constructor

        :param run_name: name of run
        :param model_cls: class of model
        :param feature_names: list of feature names
        :param params: hyper parameters
        '''
        self.run_name = run_name
        self.model_cls = model_cls
        self.params = params
        self.n_fold = 4


    def __train_fold(self, i_fold: Union[int, str]) -> Tuple[
                    Model, Optional[np.array],
                    Optional[np.array], Optional[float]]:
        ''' specifies number of fold for cv then learns & evaluates

        :param i_fold: number of fold
        :return: a tuple of instance of model, index of record,
                 predicted value, and evaluation score
        '''
        # load train data
        y_train_all = self.__load_y_train()

        # split data into training and validation
        idx_train, idx_valid = self.__load_index_fold(i_fold)
        y_train = y_train_all.iloc[idx_train]
        y_valid = y_train_all.iloc[idx_valid]

        train_dataset = CustomDataset(y_train, '../input/g2net-n-mels-128-train-images', batch_size=64)
        valid_dataset = CustomDataset(y_valid, '../input/g2net-n-mels-128-train-images', batch_size=64, shuffle=False)

        # execute learning
        model = self.__build_model(i_fold)
        model.train(train_dataset, valid_dataset)

        # prediction and evaluation with validation data
        pred_valid = model.predict(valid_dataset)
        _, score, _ = roc_auc_score(y_true=y_valid, y_pred=pred_valid)

        # return model, index, prediction, and score
        return model, idx_valid, pred_valid, score


    def run_train_cv(self) -> None:
        ''' learns and evaluates by CV

        learns, evaluates, and saves models and scores of each fold
        '''
        scores = []
        idxes_valid = []
        preds = []

        # learning for each fold
        for i_fold in range(self.n_fold):
            model, idx_valid, pred_valid, score = self.__train_fold(i_fold)

            # save model
            model.save_model()

            # hold result
            idxes_valid.append(idx_valid)
            scores.append(score)
            preds.append(pred_valid)

            logger.info(f'{self.run_name} - end training cv - score {np.mean(scores)}')

            # save prediction
            Util.dump(preds, f'../model/pred/{self.run_name}-train.pkl')

            # save scores
            logger.result_scores(self.run_name, scores)


    def run_predict_cv(self) -> None:
        ''' predicts for test data with the mean of
            each fold's model learned through CV

        needs to run run_train_cv beforehand
        '''
        logger.info(f'{self.run_name} - start prediction cv')

        y_test = self.__load_y_test()
        test_dataset = CustomDataset(y_test, "../input/g2net-n-mels-128-test-images",batch_size=64, target=False, shuffle=False)

        preds = []
        # prediction for each fold's model
        for i_fold in range(self.n_fold):
            logger.info(f'{self.run_name} - start prediction fold:{i_fold}')
            model = self.__build_model(i_fold)
            model.load_model()
            pred = model.predict(test_dataset)
            preds.append(pred)
            logger.info(f'{self.run_name} - end prediction fold:{i_fold}')

        # mean of the prediction values
        pred_mean = np.mean(preds, axis=0)

        # save the prediction result
        Util.dump(pred_mean, f'../model/pred/{self.run_name}-test.pkl')

        logger.info(f'{self.run_name} - end prediction cv')


    def __build_model(self, i_fold: Union[int, str]) -> Model:
        ''' builds a model with a specified fold for cv

        :param i_fold: number of fold
        :return: instance of model
        '''
        # build a model with run name, fold, and class of model
        run_fold_name = f'{self.run_name}-{i_fold}'
        return self.model_cls(run_fold_name, self.params)


    def __load_y_train(self) -> pd.DataFrame:
        ''' loads target of train data

        :return: target of train data
        '''
        return pd.read_csv('../input/g2net-gravitational-wave-detection/training_labels.csv')


    def __load_y_test(self) -> pd.DataFrame:
        ''' loads sample target of test data

        :return: sample target of test data
        '''
        return pd.read_csv('../input/g2net-gravitational-wave-detection/sample_submission.csv')


    def __load_index_fold(self, i_fold: int) -> np.array:
        ''' returns the record index in response to the fold specified for cv

        :param i_fold: number of the fold
        :return: record index for the fold
        '''
        y_train = self.__load_y_train()
        x_dummy = np.zeros(len(y_train))
        skf = KFold(n_splits=self.n_fold, shuffle=True, random_state=31)
        return list(skf.split(x_dummy, y_train))[i_fold]
