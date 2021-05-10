import tensorflow as tf
import numpy as np
import math

label_real = 0.9  # soft label (see paper)
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # binary cross-entropy


def get_xavier_initialization(mat_shape):
    """
    Does Xavier initialization for one (weight) matrix, given a shape.
    :param mat_shape: the numpy shape of the matrix to be initialized
    """
    init = tf.keras.initializers.GlorotNormal(seed=42)  # get initializer with random seed
    initialized_matrix = init(shape=mat_shape)

    return initialized_matrix


def get_gen_loss(fake_image_out):
    return bce(np.zeros(fake_image_out.shape), fake_image_out)


def get_discrim_loss(fake_image_out, real_image_out):
    """

    """
    fake_loss = bce(np.full(fake_image_out.shape, label_real), fake_image_out)  # log(1 - D(G(z)))
    real_loss = bce(np.ones(real_image_out.shape), real_image_out)  # log(D(x))

    return fake_loss + real_loss


def produce_training_batches(data, bs=260):
    """
    Produces equi-sized batches of training data from a given dataset.

    :param data: the dataset, as a numpy array (4D)
    :param bs: the batch size (the last batch might be smaller)
    """
    n_samples = data.shape[0]
    n_batches = math.ceil(n_samples / bs)
    current_batch_index = 0

    while current_batch_index < n_batches:
        if current_batch_index < n_batches - 1:
            yield data[current_batch_index * bs: ((current_batch_index + 1) * bs)]
        else:
            yield data[current_batch_index * (n_batches - 1):]
        current_batch_index += 1


def display_custom_loading_bar(msg, current_prog, max):
    """

    """
    progress = (current_prog / max) * 100  # progress in %
    progress_saved = progress
    bars_left = 50  # one char for every 2%
    display_str = msg + ':   '

    while progress > 0:
        display_str += '='
        progress -= 2
        bars_left -= 1

    while bars_left > 0:
        display_str += '_'
        bars_left -= 1

    display_str += f'>   {round(progress_saved)}%'

    return display_str
