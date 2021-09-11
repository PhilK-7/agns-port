import numpy as np
import tensorflow as tf
import dcgan_utils
import os

from PIL import Image


def build_model():
    """
    Builds the generator part of the eye-glass generating DCGAN model.

    :return: the built generator model as tf.keras Sequential object (not compiled yet)
    """
    inp = tf.keras.layers.InputLayer((25,))  # input layer
    fc = tf.keras.layers.Dense(7040)  # fully connected layer
    reshape = tf.keras.layers.Reshape(target_shape=(4, 11, 160))  # reshape tensor
    # "deconvolutional" layers
    deconv1 = tf.keras.layers.Conv2DTranspose(80, (5, 5), strides=(2, 2), padding='same')
    deconv2 = tf.keras.layers.Conv2DTranspose(40, (5, 5), strides=(2, 2), padding='same')
    deconv3 = tf.keras.layers.Conv2DTranspose(20, (5, 5), strides=(2, 2), padding='same')
    deconv4 = tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='tanh')

    model = tf.keras.models.Sequential(
        [
            inp,
            fc,
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            reshape,
            deconv1,
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            deconv2,
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            deconv3,
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            deconv4
        ],
        name='Generator'
    )

    model.summary()

    return model


# NOTE: start fresh training instead
@DeprecationWarning
def load_gen_weights(gmodel):
    from deprecated import model_importer
    npas = model_importer.load_dcgan_mat_model_weights('../matlab-models/gen.mat')
    gmodel.layers[0].set_weights([npas[0], dcgan_utils.get_xavier_initialization((7040,))])
    gmodel.layers[4].set_weights([np.reshape(npas[3], (5, 5, 80, 160)), dcgan_utils.get_xavier_initialization((80,))])
    gmodel.layers[7].set_weights([np.reshape(npas[6], (5, 5, 40, 80)), dcgan_utils.get_xavier_initialization((40,))])
    gmodel.layers[10].set_weights([np.reshape(npas[9], (5, 5, 20, 40)), dcgan_utils.get_xavier_initialization((20,))])
    gmodel.layers[13].set_weights([np.reshape(npas[12], (5, 5, 3, 20)), dcgan_utils.get_xavier_initialization((3,))])

    return gmodel


def scale_gen_output(prediction):
    """
    Scales the values of a NumPy array with original range [-1, 1] to range [0, 255].

    :param prediction: a numpy array with values ranging between -1 and 1
    :return: a numpy array with integer values between 0 and 255
    """
    prediction += 1  # shift to range [0, 2]
    prediction *= 127.5  # scale to range [0, 255]
    prediction = np.round(prediction, 0)
    prediction = prediction.astype(int)

    return prediction


def save_gen_output_to_file(matrix):
    print(f'Saving image matrix of size {np.shape(matrix)}')
    matrix = np.asarray(matrix, dtype=np.uint8)
    img = Image.fromarray(matrix, 'RGB')

    # img.show()
    if not os.path.exists('../../out'):
        os.makedirs('../../out')
    img.save('../../out/generated_glass.png', 'PNG')
