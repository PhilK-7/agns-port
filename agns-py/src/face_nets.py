# pasted from Latex suggestion
import os
import sys
from os import path
#for usage from command line
sys.path.append(path.dirname(path.dirname(path.abspath('face_nets.py'))))

import tensorflow as tf
import numpy as np
import tensorflow.keras.applications.vgg16 as vgg
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_10_class_model_dict():
    """
    Returns the dictionary for the classes used in the 10 class face recognition models.
    """
    return {
        0: 'Alyssa Milano',
        1: 'Barack Obama',
        2: 'Clive Owen',
        3: 'Drew Barrymore',
        4: 'Eva Mendes',
        5: 'Faith Hill',
        6: 'George Clooney',
        7: 'Halle Berry',
        8: 'Gisele Bundchen',
        9: 'Jack Nicholson'
    }


def get_original_vgg_model():
    """
    Loads the original VGG model (VGG16, type D), and cuts of the final layer (1000-class output).
    The weights are loaded from a pretrained model, trained on Imagenet data.
    The model's output is a 4096-d face descriptor.
    """
    base_model = vgg.VGG16(include_top=True)
    model = tf.keras.models.Model(base_model.input, base_model.layers[-2].output, name='VGG_Descriptor')
    #model.summary()

    return model


def build_openface_model():
    """
    Builds the nn4.small2 OpenFace model, with about 3.74M parameters (3733968)
    (exact model type was inferred from paper's parameter count).
    The model's output is a 128-sphere.
    """
    '''with tf.keras.utils.CustomObjectScope({'tf': tf}):
        model = tf.keras.models.load_model('../nn4.small2.v1.h5')
        model.summary()
        return model'''

    # input part
    inp = tf.keras.layers.Input((96, 96, 3))  # input is (aligned) RBG image pf 96x96
    x = tf.keras.layers.Conv2D(64, 7, 2, 'same', dilation_rate=1, name='First_Conv2D')(inp)  # 48x48
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    #x = tf.keras.layers.Lambda(tf.nn.dilation2d(x, 64, 1, 1))(x)
    x = tf.keras.layers.MaxPool2D((3, 3), (2, 2), padding='same')(x)  # 24x24
    #x = tf.keras.layers.Lambda(tf.nn.local_response_normalization(x, 5, alpha=1e-4, beta=0.75))(x)

    # Inception 2 (output size 24x24)
    x = tf.keras.layers.Conv2D(64, 1, 1, 'same', name='Inception_2_Conv2D')(x)  # 24x24
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(192, 3, padding='same')(x)  # 24x24
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    #x = tf.keras.layers.Lambda(tf.nn.local_response_normalization())(x)
    #x = tf.keras.layers.Lambda(tf.nn.dilation2d())(x)
    x = tf.keras.layers.MaxPool2D((3, 3), (2, 2), padding='same')(x)  # 12x12

    # Inception 3a (output size 12x12)

    # Inception 3b (output size 12x12)

    # Inception 3c (output size 6x6)

    # Inception 4a (output size 6x6)

    # Inception 4e (output size 3x3)

    # Inception 5a (output size 3x3)

    # Inception 5b (output size 3x3)

    # final layers
    x = tf.keras.layers.AvgPool2D((3, 3))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128)(x)
    #x = tf.keras.layers.Lambda(tf.math.l2_normalize())(x)

    model = tf.keras.Model(inputs=[inp], outputs=[x], name='Openface NN4.Small2.v1')
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
        name='VGG143_head' if bigger_class_n else 'VGG10_head')
    #model.summary()

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
        name='OF143_head' if bigger_class_n else 'OF10_head')
    model.summary()

    return model


def train_vgg_dnn(epochs=1, bigger_class_n=True):
    """

    """
    # compose complete model
    vgg_base = get_original_vgg_model()
    for layer in vgg_base.layers:  # freeze VGG base layers (transfer learning)
        layer.trainable = False
    top_part = build_vgg_custom_part(bigger_class_n)
    class_suffix = '_143' if bigger_class_n else '_10'
    save_path = '../saved-models/vgg' + class_suffix + '.h5'

    # get saved weights, or start with new transfer learning
    try:
        model = tf.keras.models.load_model(save_path)
        print('Model state loaded. Continue training...')
    except OSError:
        model = tf.keras.Sequential([vgg_base, top_part], name='VGG' + class_suffix + '_complete')
        print('No saved weights found. Start training new model...')

    model.summary()

    # load dataset, rescale + resize images
    ds_path = '../data/pubfig/dataset_'
    remote_ds_path = '../../../../data-private/dataset_'
    ds_path = remote_ds_path
    if not bigger_class_n:
        ds_path += '10/'
    else:
        ds_path += '/'

    datagen = ImageDataGenerator(rescale=1./255)
    datagen = datagen.flow_from_directory(ds_path, (224, 224))
    '''
    WARNING: If there are mysterious duplicate files starting with '._' in the subclass directories, the program
    will crash.
    To remove the files, go to 'dataset_' and execute this command:
    find . -type f -name ._\* -exec rm {} \;
    '''

    # do training
    opt = tf.keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(opt, 'categorical_crossentropy', ['accuracy'])
    losses = model.fit(datagen, epochs=epochs, ).history

    # save model state
    model.save(save_path)


if __name__ == '__main__':
    '''os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"]='4,5'
    train_vgg_dnn(50, True)'''
    build_openface_model()
