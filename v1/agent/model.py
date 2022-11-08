from tensorflow import keras

FP_DIMS = 86


class Model(object):
    def __init__(self):
        # Describe model structure
        self.model = keras.Sequential(
            [
                keras.Input(shape=(FP_DIMS,)),
                keras.layers.Dense(FP_DIMS, activation="relu", name="layer1"),
                keras.layers.Dense(FP_DIMS, activation="softmax", name="layer2")
            ]
        )

        # Display model summary (requires stating Input shape)
        self.model.summary()

    def forward(self, inputs):
        return self.model(inputs)
