import tensorflow as tf
import numpy as np


def build_vgg_custom(bigger_class_n=False):
    """

    """
    inp = tf.keras.layers.InputLayer((4096,), name='Input Layer')
    dense = tf.keras.layers.Dense(143 if bigger_class_n else 10, activation='softmax', name='Simplex')

    model = tf.keras.Sequential([
        inp,
        dense
    ],
        name='VGG143' if bigger_class_n else 'VGG10')
    model.summary()

    return model


def build_of_custom(bigger_class_n=False):
    """
    
    """
    inp = tf.keras.layers.InputLayer((128,), name='Sphere Input')
    dense_1 = tf.keras.layers.Dense(286 if bigger_class_n else 12, name='Fully_Connected', activation='tanh')
    dense_2 = tf.keras.layers.Dense(143 if bigger_class_n else 10, name='Simplex', activation='softmax')

    model = tf.keras.Sequential([
        inp,
        dense_1,
        dense_2
    ],
        name='OF143' if bigger_class_n else 'OF10')
    model.summary()

    return model


if __name__ == '__main__':
    build_of_custom(True)
