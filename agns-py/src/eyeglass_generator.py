import numpy as np
import tensorflow as tf
import model_importer
import net_utils


from PIL import Image


def build_model():
    """
    Builds the generator part of the eye-glass generating DCGAN model.
    :return: the built generator model (not compiled yet)
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
        ]
    )

    model.summary()

    return model


def load_gen_weights(gmodel):
    npas = model_importer.load_mat_model_weights('../gen.mat')
    gmodel.layers[0].set_weights([npas[0], net_utils.get_xavier_initialization((7040,))])
    gmodel.layers[4].set_weights([np.reshape(npas[3], (5, 5, 80, 160)), net_utils.get_xavier_initialization((80,))])
    gmodel.layers[7].set_weights([np.reshape(npas[6], (5, 5, 40, 80)), net_utils.get_xavier_initialization((40,))])
    gmodel.layers[10].set_weights([np.reshape(npas[9], (5, 5, 20, 40)), net_utils.get_xavier_initialization((20,))])
    gmodel.layers[13].set_weights([np.reshape(npas[12], (5, 5, 3, 20)), net_utils.get_xavier_initialization((3,))])

    return gmodel


def scale_gen_output(prediction):
    prediction += 1
    prediction *= 127.5
    prediction = np.round(prediction, 0)

    return prediction


def save_gen_output_to_file(matrix):
    print(f'Saving image matrix of size {np.shape(matrix)}')
    img = Image.fromarray(matrix, 'RGB')
    #img.show()
    img.save('../out/generated.png', 'PNG')


if __name__ == '__main__':
    model = build_model()
    model = load_gen_weights(model)
    vector = np.random.uniform(-1, 1, 25)
    vector = np.reshape(vector, (1, 25))
    #print(np.shape(vector))
    model.build()
    pred = scale_gen_output(np.reshape(model.predict(vector), (64, 176, 3)))
    print(pred)
    save_gen_output_to_file(pred)
