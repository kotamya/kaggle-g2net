import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):

    def __init__(self, run_fold_name: str, params: dict) -> None:
        ''' Constructor

        :param params: hyper parameters
        '''
        self.run_fold_name = run_fold_name
        self.params = params
        self.model = None

    @abstractmethod
    def train(self, train_dataset, valid_dataset) -> None:
        ''' trains a model

        :param X_train: features of training data
        :param y_train: targets of training data
        :param X_valid: features of validation data
        :param y_valid: targets of validation data
        '''
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.array:
        ''' returns prediction output from a learned model

        :param X: features of test data or validation data
        :return: predicted value
        '''
        pass

    @abstractmethod
    def save_model(self) -> None:
        ''' saves a model '''
        pass

    @abstractmethod
    def load_model(self) -> None:
        ''' loads a model '''
        pass
