import tensorflow as tf
import numpy as np
import math

LABEL_REAL = 0.9  # soft label (see paper)
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
    """
    Computes the DCGAN generator loss, based on the whole model´s output with a generated fake image.
    """

    return bce(np.zeros(fake_image_out.shape), fake_image_out)


def get_discrim_loss(fake_image_out, real_image_out):
    """
    Computes the DCGAN discriminator loss, based on the whole model´s output when given a pair of fake and real images.
    """
    fake_loss = bce(np.full(fake_image_out.shape, LABEL_REAL), fake_image_out)  # log(1 - D(G(z)))
    real_loss = bce(np.ones(real_image_out.shape), real_image_out)  # log(D(x))

    return fake_loss + real_loss


@DeprecationWarning
def produce_training_batches(data, bs=32):
    """
    Produces random, equi-sized mini-batches of training data from a given dataset.

    :param data: the dataset, as a numpy array (4D)
    :param bs: the batch size (the last batch might be smaller)
    """
    np.random.shuffle(data)  # shuffle randomly to get randomized mini batches
    n_samples = data.shape[0]
    n_batches = math.ceil(n_samples / bs)
    current_batch_index = 0

    while current_batch_index < n_batches:
        if current_batch_index < n_batches - 1:
            yield data[current_batch_index * bs: ((current_batch_index + 1) * bs)]
        else:
            yield data[current_batch_index * (n_batches - 1):]
        current_batch_index += 1


@DeprecationWarning
def display_custom_loading_bar(msg, current_prog, max):
    """
    A help function to generate a loading bar when loading and processing the training data.

    :param msg: the start of the loading bar string
    :param current_prog: the current progress, as an absolute number
    :param max: the maximum progress as absolute number
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
