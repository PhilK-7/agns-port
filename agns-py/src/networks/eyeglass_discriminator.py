import tensorflow as tf
import model_importer
import numpy as np
from PIL import Image
import dcgan_utils


def build_model():
    """
    Builds the discriminator model for the eyeglass-generating DCGAN.

    :return: a tf.keras Sequential model object
    """

    inp = tf.keras.layers.InputLayer((64, 176, 3))
    conv1 = tf.keras.layers.Conv2D(20, (5, 5), strides=(2, 2), padding='same')
    conv2 = tf.keras.layers.Conv2D(40, (5, 5), strides=(2, 2), padding='same')
    conv3 = tf.keras.layers.Conv2D(80, (5, 5), strides=(2, 2), padding='same')
    conv4 = tf.keras.layers.Conv2D(160, (5, 5), strides=(2, 2), padding='same')
    reshape = tf.keras.layers.Reshape((7040, 1))
    dense = tf.keras.layers.Dense(1, activation='sigmoid')

    dmodel = tf.keras.models.Sequential(
        [
            inp,
            conv1,
            tf.keras.layers.LeakyReLU(),
            conv2,
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            conv3,
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            conv4,
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            reshape,
            tf.keras.layers.Flatten(),
            MiniBatchDiscrimination(),
            dense
        ],
        name='Discriminator'
    )

    dmodel.summary()

    return dmodel


class MiniBatchDiscrimination(tf.keras.layers.Layer):
    """
    A mini-batch discrimination layer, which helps in DCGAN training to generate more diverse samples.
    """
    def __init__(self, **kwargs):
        super(MiniBatchDiscrimination, self).__init__()
        self.dense_help = tf.keras.layers.Dense(160 * 3)

    def get_config(self):
        conf = super().get_config().copy()
        return conf

    def call(self, inputs, **kwargs):
        # as taken similarly from https://github.com/AYLIEN/gan-intro
        x = self.dense_help(inputs)
        activation = tf.reshape(x, (-1, 160, 3))
        diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
        return tf.concat([inputs, minibatch_features], 1)


# NOTE: start fresh training instead
@DeprecationWarning
def load_discrim_weights(dmodel):
    npas = model_importer.load_dcgan_mat_model_weights('../matlab-models/discrim.mat')
    dmodel.layers[0].set_weights([np.reshape(npas[0], (5, 5, 3, 20)), dcgan_utils.get_xavier_initialization((20,))])
    dmodel.layers[2].set_weights([np.reshape(npas[1], (5, 5, 20, 40)), dcgan_utils.get_xavier_initialization((40,))])
    dmodel.layers[5].set_weights([np.reshape(npas[4], (5, 5, 40, 80)), dcgan_utils.get_xavier_initialization((80,))])
    dmodel.layers[8].set_weights([np.reshape(npas[7], (5, 5, 80, 160)), dcgan_utils.get_xavier_initialization((160,))])
    dmodel.layers[13].set_weights([npas[10], dcgan_utils.get_xavier_initialization((1,))])

    return dmodel


@DeprecationWarning
def convert_image_to_matrix(impath):
    """
    Converts an image to a numpy matrix of the correct form to provide as input to the discriminator model.

    :param impath: the path to the input image
    :return: a numpy array of shape (1, 64, 176, 3) with values in range [-1, 1] computed from the given image
    """

    img = Image.open(impath)  # open real glasses image
    img = img.crop((25, 53, 201, 117))  # crop to actual size 64x176 (also see matlab code)
    img_matrix = np.asarray(img)  # get as numpy array
    mapped_matrix = img_matrix / 127.5 - 1  # map (0, 255) range to (-1, 1)
    mapped_matrix = np.reshape(mapped_matrix, (1, img_matrix.shape[0], img_matrix.shape[1], img_matrix.shape[2]))
    print(f'Shape of image: {np.shape(mapped_matrix)}')
    img.show()

    return mapped_matrix
