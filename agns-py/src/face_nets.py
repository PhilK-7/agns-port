# pasted from Latex suggestion
import os
import sys
from os import path

# for usage from command line
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
    # model.summary()

    return model


class LocalResponseNormalization(tf.keras.layers.Layer):
    """
    The Local Response Normalization layer. Normalizes high frequency features with radial masks.
    Applies the parameters needed in OpenFace NN4.small2.v1.
    """
    def __init__(self):
        super(LocalResponseNormalization, self).__init__()

    def call(self, inputs, **kwargs):
        return tf.nn.local_response_normalization(inputs, alpha=1e-4, beta=0.75)


class L2Pooling(tf.keras.layers.Layer):
    """
    The L2 Pooling layer. Computes a kind of Euclidean norm using average pooling.
    Uses pooling size 3, as needed in OpenFace NN4.small2.v1.
    """
    def __init__(self):
        super(L2Pooling, self).__init__()

    def call(self, inputs, **kwargs):
        x = tf.math.square(inputs)
        x = tf.keras.layers.AvgPool2D((3, 3), (1, 1), padding='same')(x)
        x = tf.math.multiply(x, 9)
        x = tf.math.sqrt(x)

        return x


class L2Normalization(tf.keras.layers.Layer):
    """
    The L2 Normalization layer. Just computes the L2 norm of its input.
    """
    def __init__(self):
        super(L2Normalization, self).__init__()

    def call(self, inputs, **kwargs):

        return tf.nn.l2_normalize(inputs)


class InceptionModule(tf.keras.layers.Layer):
    # TODO doc
    def __init__(self, conv_output_sizes, reduce_sizes, name, use_l2_pooling=False):
        super(InceptionModule, self).__init__(name=name)

        assert len(conv_output_sizes) >= 1
        assert len(reduce_sizes) >= 3
        assert len(reduce_sizes) == len(conv_output_sizes) + 2
        self.cos = conv_output_sizes
        self.rs = reduce_sizes

        # two variants: one with both 3x3 and 5x5 convolutions, one with only 3x3 convolution
        self.shift = 0
        if len(conv_output_sizes) == 1:
            self.shift = 1  # adjust reduce + pool indices if necessary
        self.reduce_conv_out = tf.keras.layers.Conv2D(self.rs[3 - self.shift], (1, 1), padding='same')
        self.rco_bn = tf.keras.layers.BatchNormalization(axis=3)
        self.reduce_3 = tf.keras.layers.Conv2D(self.rs[0], (1, 1), padding='same')
        self.r3_bn = tf.keras.layers.BatchNormalization(axis=3)
        self.out_3 = tf.keras.layers.Conv2D(self.cos[0], (3, 3), padding='same')
        self.o3_bn = tf.keras.layers.BatchNormalization(axis=3)
        if self.shift == 0:  # only add these layers for one variant
            self.reduce_5 = tf.keras.layers.Conv2D(self.rs[1], (1, 1), padding='same')
            self.r5_bn = tf.keras.layers.BatchNormalization(axis=3)
            self.out_5 = tf.keras.layers.Conv2D(self.cos[1], (5, 5), padding='same')
            self.o5_bn = tf.keras.layers.BatchNormalization(axis=3)
        if use_l2_pooling:
            self.pool = L2Pooling()
        else:
            self.pool = tf.keras.layers.MaxPool2D((3, 3), (1, 1), 'same')
        self.pool_out = tf.keras.layers.Conv2D(self.rs[2 - self.shift], (1, 1), padding='same')
        self.po_bn = tf.keras.layers.BatchNormalization(axis=3)

    def call(self, inputs, **kwargs):
        # only reduction part
        p1 = self.reduce_conv_out(inputs)
        p1 = self.rco_bn(p1)
        p1 = tf.keras.layers.ReLU()(p1)
        # 3x3 convolution part
        p2 = self.reduce_3(inputs)
        p2 = self.r3_bn(p2)
        p2 = tf.keras.layers.ReLU()(p2)
        p2 = self.out_3(p2)
        p2 = self.o3_bn(p2)
        p2 = tf.keras.layers.ReLU()(p2)
        # 5x5 convolution part (one of two variants)
        if self.shift == 0:
            p3 = self.reduce_5(inputs)
            p3 = self.r5_bn(p3)
            p3 = tf.keras.layers.ReLU()(p3)
            p3 = self.out_5(p3)
            p3 = self.o5_bn(p3)
            p3 = tf.keras.layers.ReLU()(p3)
        else:
            p3 = None
        # pooling part
        p4 = self.pool(inputs)  # pooling without reducing filters
        p4 = self.pool_out(p4)
        p4 = self.po_bn(p4)
        p4 = tf.keras.layers.ReLU()(p4)

        concat = tf.keras.layers.Concatenate(axis=3)

        return concat([p1, p2, p3, p4]) if p3 is not None else concat([p1, p2, p4])


class InceptionModuleShrink(tf.keras.layers.Layer):
    # TODO doc
    def __init__(self, conv_output_sizes, reduce_sizes, name):
        super(InceptionModuleShrink, self).__init__(name=name)

        assert len(conv_output_sizes) == 2 == len(reduce_sizes)
        self.cos = conv_output_sizes
        self.rs = reduce_sizes

        self.reduce3 = tf.keras.layers.Conv2D(self.rs[0], (1, 1), padding='same')
        self.r3_bn = tf.keras.layers.BatchNormalization(axis=3)
        self.conv3 = tf.keras.layers.Conv2D(self.cos[0], (3, 3), (2, 2), 'same')
        self.c3_bn = tf.keras.layers.BatchNormalization(axis=3)
        self.reduce5 = tf.keras.layers.Conv2D(self.rs[1], (1, 1), padding='same')
        self.r5_bn = tf.keras.layers.BatchNormalization(axis=3)
        self.conv5 = tf.keras.layers.Conv2D(self.cos[1], (5, 5), (2, 2), 'same')
        self.c5_bn = tf.keras.layers.BatchNormalization(axis=3)
        self.pool = tf.keras.layers.MaxPool2D((3, 3), (2, 2), padding='same')

    def call(self, inputs, **kwargs):
        # 3x3 convolution part
        p1 = self.reduce3(inputs)
        p1 = self.r3_bn(p1)
        p1 = tf.keras.layers.ReLU()(p1)
        p1 = self.conv3(p1)
        p1 = self.c3_bn(p1)
        p1 = tf.keras.layers.ReLU()(p1)
        # 5x5 convolution part
        p2 = self.reduce5(inputs)
        p2 = self.r5_bn(p2)
        p2 = tf.keras.layers.ReLU()(p2)
        p2 = self.conv5(p2)
        p2 = self.c5_bn(p2)
        p2 = tf.keras.layers.ReLU()(p2)
        # pooling part
        pool = self.pool(inputs)

        return tf.keras.layers.Concatenate(axis=3)([p1, p2, pool])


def build_openface_model():
    """
    Builds the nn4.small2 OpenFace model, with about 3.74M trainable parameters (3733968)
    (exact model type was inferred from paper's parameter count).
    The model's output is a 128-sphere.
    """

    # input part
    inp = tf.keras.layers.Input((96, 96, 3))  # input is (aligned) RBG image pf 96x96
    x = tf.keras.layers.Conv2D(64, 7, 2, padding='same', name='First_Conv2D')(inp)  # 48x48 x 64
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPool2D(3, 2, padding='same')(x)  # 24x24  x 64
    x = LocalResponseNormalization()(x)

    # Inception 2 (output size 24x24)
    x = tf.keras.layers.Conv2D(64, 1, 1, 'same', name='Inception_2_Conv2D')(x)  # 24x24 x 64
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(192, 3, padding='same')(x)  # 24x24 x 192
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.ReLU()(x)

    x = LocalResponseNormalization()(x)
    x = tf.keras.layers.MaxPool2D(3, 2, padding='same')(x)  # 12x12 x 192

    # Inception 3a (output size 12x12 x 256)
    x = InceptionModule([128, 32], [96, 16, 32, 64], 'Inception_3a')(x)

    # Inception 3b (output size 12x12 x 320)
    x = InceptionModule([128, 64], [96, 32, 64, 64], 'Inception_3b', True)(x)

    # Inception 3c (output size 6x6 x 640)
    x = InceptionModuleShrink([256, 64], [128, 32], 'Inception_3c')(x)

    # Inception 4a (output size 6x6 x 640)
    x = InceptionModule([192, 64], [96, 32, 128, 256], 'Inception_4a', True)(x)

    # Inception 4e (output size 3x3 x 1024)
    x = InceptionModuleShrink([256, 128], [160, 64], 'Inception_4e')(x)

    # Inception 5a (output size 3x3 x 736)
    x = InceptionModule([384], [96, 96, 256], 'Inception_5a', True)(x)

    # Inception 5b (output size 3x3 x 736)
    x = InceptionModule([384], [96, 96, 256], 'Inception_5b')(x)

    # final layers
    x = tf.keras.layers.AvgPool2D((3, 3))(x)  # 1x1 x 736
    x = tf.keras.layers.Flatten(name='reshape')(x)  # 736
    x = tf.keras.layers.Dense(128)(x)
    x = L2Normalization()(x)

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
    # model.summary()

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

    datagen = ImageDataGenerator(rescale=1. / 255)
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
