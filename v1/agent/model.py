from tensorflow import keras

from environment.state_handling import get_num_configs

FP_DIMS = 86


class Model(object):
    def __init__(self):
        num_actions = get_num_configs()
        # Describe model structure
        self.model = keras.Sequential(
            [
                keras.Input(shape=(FP_DIMS,)),
                keras.layers.Dense(FP_DIMS, activation="relu", name="layer1"),
                keras.layers.Dense(num_actions, activation="softmax", name="layer2")
            ]
        )

        # Display model summary (requires stating Input shape)
        self.model.summary()

    def forward(self, inputs):
        return self.model(inputs)
