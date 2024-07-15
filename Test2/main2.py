import tensorflow as tf
import keras

(x_train_data, y_train_data), (x_val_data, y_val_data) = (
    keras.datasets.fashion_mnist.load_data()
)
