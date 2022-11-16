from tensorflow import keras

from environment.state_handling import get_num_configs

FP_DIMS = 86


class Model(object):
    def __init__(self):
        # Describe model structure
        self.model = keras.Sequential(
            [
                keras.Input(shape=(FP_DIMS,)),
                keras.layers.Dense(FP_DIMS, activation="relu", name="layer1"),
                keras.layers.Dense(FP_DIMS, activation="relu", name="layer2"),
                keras.layers.Dense(get_num_configs(), activation="softmax", name="layer3")
            ]
        )

        # Display model summary (requires stating Input shape)
        self.model.summary()

    def forward(self, inputs):
        # TODO: https://keras.io/guides/writing_a_training_loop_from_scratch/
        #  implement backward prop (either in agent during episode loop or here)
        #  https://keras.io/examples/rl/deep_q_network_breakout/
        return self.model(inputs)
