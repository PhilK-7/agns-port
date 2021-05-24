import tensorflow as tf
import numpy as np
import tensorflow.keras.applications.vgg16 as vgg


def get_original_vgg_model():
    """
    Loads the original VGG model (VGG16, type D), and cuts of the final layer (1000-class output).
    The weights are loaded from a pretrained model, trained on Imagenet data.
    The model's output is a 4096-d face descriptor.
    """
    base_model = vgg.VGG16(include_top=True)
    model = tf.keras.models.Model(base_model.input, base_model.layers[-2].output, name='VGG_Descriptor')
    model.summary()

    return model


def build_openface_model():
    """
    Builds the nn4.small2 OpenFace model, with about 3.74M parameters
    (exact model type was inferred from paper's parameter count).
    The model's output is a 128-sphere.
    """
    with tf.keras.utils.CustomObjectScope({'tf': tf}):
        model = tf.keras.models.load_model('../nn4.small2.v1.h5')
        model.summary()
        return model


def build_vgg_custom_part(bigger_class_n=False):
    """
    Builds the paper's additional model part for a VGG-based face DNN.
    The inputs are 4096-d face descriptors, gained from the standard VGG model.

    :param bigger_class_n: whether to use 143 instead of 10 classes
    """
    inp = tf.keras.layers.InputLayer((4096,), name='Descriptor_Input')
    dense = tf.keras.layers.Dense(143 if bigger_class_n else 10, activation='softmax', name='Simplex')

    model = tf.keras.Sequential([
        inp,
        dense
    ],
        name='VGG143' if bigger_class_n else 'VGG10')
    model.summary()

    return model


def build_of_custom_part(bigger_class_n=False):
    """
    Builds the paper's additional model part for an OpenFace-based face DNN.
    The inputs are 128-spheres, gained from the OpenFace model.

    :param bigger_class_n: whether to use 143 instead of 10 classes
    """
    inp = tf.keras.layers.InputLayer((128,), name='Sphere_Input')
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
    build_openface_model()
