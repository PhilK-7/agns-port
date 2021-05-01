import tensorflow as tf
import model_importer
import numpy as np
from PIL import Image
import net_utils
import batchnorm_custom_layer as bn


def build_model():
    """

    :return:
    """

    inp = tf.keras.layers.InputLayer((64, 176, 3))
    conv1 = tf.keras.layers.Conv2D(20, (5, 5), strides=(2, 2), padding='same')
    conv2 = tf.keras.layers.Conv2D(40, (5, 5), strides=(2, 2), padding='same')
    conv3 = tf.keras.layers.Conv2D(80, (5, 5), strides=(2, 2), padding='same')
    conv4 = tf.keras.layers.Conv2D(160, (5, 5), strides=(2, 2), padding='same')
    reshape = tf.keras.layers.Reshape((7040, 1))
    dense = tf.keras.layers.Dense(1, activation='sigmoid')

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


def load_discrim_weights(dmodel):
    npas = model_importer.load_mat_model_weights('../discrim.mat')
    dmodel.layers[0].set_weights([np.reshape(npas[0], (5, 5, 3, 20)), net_utils.get_xavier_initialization((20,))])
    dmodel.layers[2].set_weights([np.reshape(npas[1], (5, 5, 20, 40)), net_utils.get_xavier_initialization((40,))])
    dmodel.layers[5].set_weights([np.reshape(npas[4], (5, 5, 40, 80)), net_utils.get_xavier_initialization((80,))])
    dmodel.layers[8].set_weights([np.reshape(npas[7], (5, 5, 80, 160)), net_utils.get_xavier_initialization((160,))])
    dmodel.layers[13].set_weights([npas[10], net_utils.get_xavier_initialization((1,))])

    return dmodel


def convert_image_to_matrix(impath):
    img = Image.open(impath)  # open real glasses image
    img = img.crop((25, 53, 201, 117))
    img_matrix = np.asarray(img)
    img_matrix = np.reshape(img_matrix, (1, img_matrix.shape[0], img_matrix.shape[1], img_matrix.shape[2]))
    print(f'Shape of image: {np.shape(img_matrix)}')
    img.show()

    return img_matrix


if __name__ == '__main__':
    model = build_model()
    model = load_discrim_weights(model)
    fake_img = np.random.randint(0, 255, (1, 64, 176, 3))
    real_img = convert_image_to_matrix('../eyeglasses/glasses000019-2.png')
    model.build()
    pred = model.predict(real_img)
    if pred >= 0.5:
        print('Real!')
    else:
        print('Fake!')

