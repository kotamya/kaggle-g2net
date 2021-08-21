import datetime
import joblib
import logging
import numpy as np
import os
import pandas as pd
import tensorflow as tf


class Util:

    @classmethod
    def dump(cls, value, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path):
        return joblib.load(path)


class Logger:

    def __init__(self):
        self.general_logger = logging.getLogger('general')
        self.result_logger = logging.getLogger('result')
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler('../model/general.log')
        file_result_handler = logging.FileHandler('../model/result.log')
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)
            self.result_logger.addHandler(stream_handler)
            self.result_logger.addHandler(file_result_handler)
            self.result_logger.setLevel(logging.INFO)

    def info(self, message):
        # output time to console and log file
        self.general_logger.info('[{}] - {}'.format(self.__now_string(), message))

    def result(self, message):
        self.result_logger.info(message)

    def result_scores(self, run_name, scores):
        # output result to console and log file
        dic = dict()
        dic['name'] = run_name
        dic['score'] = np.mean(scores)
        for i, score in enumerate(scores):
            dic[f'score{i}'] = score
        self.result(self.__to_lstv(dic))

    def __now_string(self):
        return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def __to_lstv(self, dic):
        return '\t'.join(['{}:{}'.format(key, value) for key, value in dic.items()])


class Submission:

    @classmethod
    def create_submission(cls, run_name):
        submission = pd.read_csv('../input/sample_submission.csv')
        pred = Util.load(f'../model/pred/{run_name}-test.pkl')
        print('\npred: ', pred, '\n')
        for i in range(pred.shape[1]):
            submission[f'Class_{i + 1}'] = pred[:, i]
        submission.to_csv(f'../submission/{run_name}.csv', index=False)


class CustomDataset(tf.keras.utils.Sequence):
    
    def __init__(self, df, directory, batch_size=32,
                 random_state=42, shuffle=True, target=True, ext='.npy'):
        np.random.seed(random_state)
        
        self.directory = directory
        self.df = df
        self.shuffle = shuffle
        self.target = target
        self.batch_size = batch_size
        self.ext = ext
        
        self.on_epoch_end()
    

    def __len__(self):
        return np.ceil(self.df.shape[0] / self.batch_size).astype(int)
    

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        batch = self.df[start_idx: start_idx + self.batch_size]
        
        signals = []

        for fname in batch.id:
            path = os.path.join(self.directory, fname + self.ext)
            data = np.load(path)
            signals.append(data)
        
        signals = np.stack(signals).astype('float32')
        
        if self.target:
            return signals, batch.target.values
        else:
            return signals
    

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
