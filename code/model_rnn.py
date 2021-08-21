import os
import tensorflow as tf
from tensorflow.keras import layers

from model import Model
from util import Util


class ModelRNN(Model):

    def train(self, train_dataset, valid_dataset=None):
        params = self.params

        # build model
        inputs = layers.Input(shape=(27, 128))
        
        gru1 = layers.Bidirectional(layers.GRU(128, return_sequences=True), name='gru_1')
        gru2 = layers.Bidirectional(layers.GRU(128, return_sequences=True), name='gru_2')
        pool1 = layers.GlobalAveragePooling1D(name='avg_pool')
        pool2 = layers.GlobalMaxPooling1D(name='max_pool')

        x = gru1(inputs)
        x = gru2(x)
        x = tf.keras.layers.Concatenate()([pool1(x), pool2(x)])

        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(1, activation="sigmoid", name="sigmoid")(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)
        model.compile("adam", loss="binary_crossentropy",
                           metrics=[tf.keras.metrics.AUC()])

        ckpt = tf.keras.callbacks.ModelCheckpoint(
                "model_weights.h5", save_best_only=True, save_weights_only=True,
                )

        if valid_dataset is not None:
            train_history = model.fit(
                                  train_dataset, 
                                  use_multiprocessing=params['use_multiprocessing'], 
                                  workers=params['wokers'], 
                                  epochs=params['epochs'],
                                  validation_data=valid_dataset,
                                  callbacks=[ckpt],
                                  )
        else:
            train_history = model.fit(
                                  train_dataset, 
                                  use_multiprocessing=params.use_multiprocessing,
                                  workers=params.wokers,
                                  epochs=params.epochs,
                                  callbacks=[ckpt],
                                  )
        model.load_weights('model_weights.h5')
        self.model = model


    def save_model(self):
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.h5')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
    

    def load_model(self):
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.h5')
        self.model = Util.load(model_path)


    def predict(self, test_dataset):
        return self.model.predict(test_dataset)
