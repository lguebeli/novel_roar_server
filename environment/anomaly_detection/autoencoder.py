from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import set_random_seed
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

from environment.state_handling import get_instance_number


class ThresholdConfig(Enum):
    MSE_IRQ = "mse_irq",
    MSE_PERCENTAGE = "mse_perc"


class ActivationConfig(str, Enum):
    RELU = "relu",
    SILU = "silu"


VERBOSE_OUTPUT = False
THRESHOLD = ThresholdConfig.MSE_PERCENTAGE
ACTIVATION = ActivationConfig.SILU


class AutoEncoder(object):
    def __init__(self, encoding_dim, random_state):
        self.encoding_dim = encoding_dim
        set_random_seed(random_state)

    def fit(self, training_set):
        # --------------------
        # Prepare AutoEncoder
        # --------------------

        input_dim = training_set.shape[1]  # the number of features

        input_layer = Input(shape=(input_dim,))
        hidden = Dense(self.encoding_dim[0], activation=ACTIVATION.value)(input_layer)
        for n_neurons in self.encoding_dim[1:]:
            hidden = Dense(n_neurons, activation=ACTIVATION.value)(hidden)
        hidden = Dense(input_dim)(hidden)

        self.autoencoder = Model(inputs=input_layer, outputs=hidden)
        opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        self.autoencoder.compile(loss='mean_squared_error', optimizer=opt, metrics=['acc'])

        # --------------------
        # Setup Threshold
        # --------------------

        if THRESHOLD == ThresholdConfig.MSE_IRQ:  # threshold mse irq, else percentage
            self.threshold = self.__get_threshold_mse_iqr(training_set)
        else:
            self.threshold = self.__get_threshold_mse_percentage(training_set, outlier_percentage=0.1)

        # --------------------
        # Train AutoEncoder
        # --------------------

        instance = get_instance_number()
        es = EarlyStopping(monitor='val_loss', mode='min', patience=20)
        mc = ModelCheckpoint('storage/best_model-{}.h5'.format(instance), monitor='val_loss', mode='min', save_best_only=True)
        history = self.autoencoder.fit(training_set, training_set, epochs=2000,
                                       batch_size=16,
                                       shuffle=True,
                                       validation_split=0.2,
                                       verbose=1 if VERBOSE_OUTPUT else 0,
                                       callbacks=[es, mc]).history
        self.autoencoder = load_model('storage/best_model-{}.h5'.format(instance))

        if VERBOSE_OUTPUT:
            self.__plot_autoencoder_training_history(history)

    def __get_threshold_mse_iqr(self, train_data):
        train_predicted = self.autoencoder.predict(train_data, verbose=VERBOSE_OUTPUT)
        mse = np.mean(np.power(train_data - train_predicted, 2), axis=1)
        iqr = np.quantile(mse, 0.75) - np.quantile(mse, 0.25)
        up_bound = np.quantile(mse, 0.75) + 1.5 * iqr
        bottom_bound = np.quantile(mse, 0.25) - 1.5 * iqr
        threshold = [up_bound, bottom_bound]
        return threshold

    def __get_threshold_mse_percentage(self, train_data, outlier_percentage):
        train_predicted = self.autoencoder.predict(train_data, verbose=VERBOSE_OUTPUT)
        mse = np.mean(np.power(train_data - train_predicted, 2), axis=1)
        thresh = np.quantile(mse, 1 - outlier_percentage)
        return thresh

    def __plot_autoencoder_training_history(self, history):
        plt.figure(figsize=(9, 6))
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')
        plt.show()

    def predict(self, eval_set):
        if THRESHOLD == ThresholdConfig.MSE_IRQ:
            return self.__detect_outliers_range(eval_set)
        else:
            return self.__detect_outliers(eval_set)

    def __detect_outliers(self, df):
        pred = self.autoencoder.predict(df, verbose=VERBOSE_OUTPUT)
        mse = np.mean(np.power(df - pred, 2), axis=1)
        outliers = np.asarray([np.array(mse) < self.threshold])
        return outliers

    def __detect_outliers_range(self, df):
        pred = self.autoencoder.predict(df, verbose=VERBOSE_OUTPUT)
        mse = np.mean(np.power(df - pred, 2), axis=1)
        up_bound = self.threshold[0]
        bottom_bound = self.threshold[1]
        outliers = np.asarray([(np.array(mse) < up_bound) & (np.array(mse) > bottom_bound)])
        return outliers
