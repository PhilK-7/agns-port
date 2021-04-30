import tensorflow as tf
from src import model_importer
import numpy as np

def build_model():
    """

    :return:
    """

    inp = tf.keras.layers.InputLayer((64, 176, 3))
    conv1 = tf.keras.layers.Conv2D(20, (5, 5), strides=(2, 2), padding='same', use_bias=False)
    conv2 = tf.keras.layers.Conv2D(40, (5, 5), strides=(2, 2), padding='same')
    conv3 = tf.keras.layers.Conv2D(80, (5, 5), strides=(2, 2), padding='same')
    conv4 = tf.keras.layers.Conv2D(160, (5, 5), strides=(2, 2), padding='same')
    reshape = tf.keras.layers.Reshape((7040, 1))
    dense = tf.keras.layers.Dense(1, activation='sigmoid', use_bias=False)

    model = tf.keras.models.Sequential(
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
            dense
        ]
    )

    model.summary()

    return model


def load_discrim_weights(model):
    npas = model_importer.load_mat_model_weights('../../discrim.mat')
    model.layers[0].set_weights([np.reshape(npas[0], (5, 5, 3, 20))])
    model.layers[2].set_weights([np.reshape(npas[1], (5, 5, 20, 40)), np.reshape(npas[2], (40,))])
    model.layers[5].set_weights([np.reshape(npas[4], (5, 5, 40, 80)), np.reshape(npas[5], (80,))])
    model.layers[8].set_weights([np.reshape(npas[7], (5, 5, 80, 160)), np.reshape(npas[8], (160,))])
    model.layers[13].set_weights([npas[10]])

    return model


if __name__ == '__main__':
    model = build_model()
    model = load_discrim_weights(model)
    fake_img = np.random.randint(0, 255, (1, 64, 176, 3))
    model.build()
    pred = model.predict(fake_img)
    if pred >= 0.5:
        print('Real!')
    else:
        print('Fake!')

