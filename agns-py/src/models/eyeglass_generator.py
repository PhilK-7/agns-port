import tensorflow as tf
import numpy as np
from src import model_importer
from matplotlib import pyplot as plt
from PIL import Image


def build_model():
    """
    Builds the generator part of the eye-glass generating DCGAN model.
    :return: the built generator model (not compiled yet)
    """
    inp = tf.keras.layers.InputLayer((25,))  # input layer
    fc = tf.keras.layers.Dense(7040)  # fully connected layer
    reshape = tf.keras.layers.Reshape(target_shape=(4, 11, 160))  # reshape tensor
    # deconvolutional layers
    deconv1 = tf.keras.layers.Conv2DTranspose(80, (5, 5), strides=(2, 2), padding='same')
    deconv2 = tf.keras.layers.Conv2DTranspose(40, (5, 5), strides=(2, 2), padding='same')
    deconv3 = tf.keras.layers.Conv2DTranspose(20, (5, 5), strides=(2, 2), padding='same')
    deconv4 = tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same',
                                              activation='tanh', use_bias=False)

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
        ]
    )

    model.summary()

    return model


def load_gen_weights(model):
    npas = model_importer.load_mat_model_weights('../../gen.mat')
    model.layers[0].set_weights([npas[0], np.reshape(npas[1], (7040,))])
    model.layers[4].set_weights([np.reshape(npas[3], (5, 5, 80, 160)), np.reshape(npas[4], (80,))])
    model.layers[7].set_weights([np.reshape(npas[6], (5, 5, 40, 80)), np.reshape(npas[7], (40,))])
    model.layers[10].set_weights([np.reshape(npas[9], (5, 5, 20, 40)), np.reshape(npas[10], (20,))])
    model.layers[13].set_weights([np.reshape(npas[12], (5, 5, 3, 20))])

    return model


def scale_gen_output(prediction):
    prediction += 1
    prediction *= 127.5
    prediction = np.round(prediction, 0)

    return prediction


def save_gen_output_to_file(matrix):
    print(f'Saving image matrix of size {np.shape(matrix)}')
    img = Image.fromarray(matrix, 'RGB')
    img.show()


if __name__ == '__main__':
    model = build_model()
    model = load_gen_weights(model)
    vector = np.random.uniform(-1, 1, 25)
    vector = np.reshape(vector, (1, 25))
    print(np.shape(vector))
    model.build()
    pred = scale_gen_output(np.reshape(model.predict(vector), (64, 176, 3)))
    print(pred)
    save_gen_output_to_file(pred)
