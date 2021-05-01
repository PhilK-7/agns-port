import tensorflow as tf
import numpy as np


def get_xavier_initialization(mat_shape):
    """
    Does Xavier initialization for one (weight) matrix, given a shape.
    :param mat_shape: the numpy shape of the matrix to be initialized
    """
    init = tf.keras.initializers.GlorotNormal(seed=42)  # get initializer with random seed
    initialized_matrix = init(shape=mat_shape)

    return initialized_matrix
