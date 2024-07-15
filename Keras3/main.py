import keras

num_features = 4
num_classes = 2
model = keras.Sequential(
    [
        keras.layers.Input(shape=(num_features,)),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()
