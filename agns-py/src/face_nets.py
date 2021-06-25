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
import tensorflow_addons as tfa


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


# custom layers

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
    """
    The normal Inception Module layer.
    This inception module has a reduction convolution (1x1), 3x3 convolution, 5x5 convolution, and pooling pathway.
    The four parts outputs are concatenated to get feature maps of different kinds.
    An alternate version is also supported that has no 5x5 convolution part.
    """

    def __init__(self, conv_output_sizes, reduce_sizes, name, use_l2_pooling=False):
        """
        :param conv_output_sizes: the output sizes (filter counts) for 3x3 and 5x5 convolution;
            a list of length 2, or length 1, then the 5x5 convolution pathway is omitted
        :param reduce_sizes: the reduction sizes (filter counts) for 3x3 convolution, 5x5 convolution (reduction here
            means the first of two convolutions in their paths each), the pooling path, and the reduction path (1x1);
            either length 4, or 3 when the 5x5 convolution path is omitted
        :param name: a name to supply for this layer
        :param use_l2_pooling: whether to use L2 pooling instead of max pooling in the pooling path
        """
        super(InceptionModule, self).__init__(name=name)

        # check constraints
        assert len(conv_output_sizes) >= 1
        assert len(reduce_sizes) >= 3
        assert len(reduce_sizes) == len(conv_output_sizes) + 2
        self.cos = conv_output_sizes
        self.rs = reduce_sizes

        # two variants: one with both 3x3 and 5x5 convolutions, one with only 3x3 convolution
        self.shift = 0
        if len(conv_output_sizes) == 1:
            self.shift = 1  # adjust reduce + pool indices if necessary
        # reduction path
        self.reduce_conv_out = tf.keras.layers.Conv2D(self.rs[3 - self.shift], (1, 1), padding='same')
        self.rco_bn = tf.keras.layers.BatchNormalization(axis=3)
        # 3x3 convolution path
        self.reduce_3 = tf.keras.layers.Conv2D(self.rs[0], (1, 1), padding='same')
        self.r3_bn = tf.keras.layers.BatchNormalization(axis=3)
        self.out_3 = tf.keras.layers.Conv2D(self.cos[0], (3, 3), padding='same')
        self.o3_bn = tf.keras.layers.BatchNormalization(axis=3)
        # 5x5 convolution path (optional)
        if self.shift == 0:  # only add these layers for one variant
            self.reduce_5 = tf.keras.layers.Conv2D(self.rs[1], (1, 1), padding='same')
            self.r5_bn = tf.keras.layers.BatchNormalization(axis=3)
            self.out_5 = tf.keras.layers.Conv2D(self.cos[1], (5, 5), padding='same')
            self.o5_bn = tf.keras.layers.BatchNormalization(axis=3)
        # pooling path
        if use_l2_pooling:
            self.pool = L2Pooling()
        else:
            self.pool = tf.keras.layers.MaxPool2D((3, 3), (1, 1), 'same')
        self.pool_out = tf.keras.layers.Conv2D(self.rs[2 - self.shift], (1, 1), padding='same')
        self.po_bn = tf.keras.layers.BatchNormalization(axis=3)

    def call(self, inputs, **kwargs):
        # reduction path
        p1 = self.reduce_conv_out(inputs)
        p1 = self.rco_bn(p1)
        p1 = tf.keras.layers.ReLU()(p1)
        # 3x3 convolution path
        p2 = self.reduce_3(inputs)
        p2 = self.r3_bn(p2)
        p2 = tf.keras.layers.ReLU()(p2)
        p2 = self.out_3(p2)
        p2 = self.o3_bn(p2)
        p2 = tf.keras.layers.ReLU()(p2)
        # 5x5 convolution path (for one of two variants)
        if self.shift == 0:
            p3 = self.reduce_5(inputs)
            p3 = self.r5_bn(p3)
            p3 = tf.keras.layers.ReLU()(p3)
            p3 = self.out_5(p3)
            p3 = self.o5_bn(p3)
            p3 = tf.keras.layers.ReLU()(p3)
        else:
            p3 = None
        # pooling path
        p4 = self.pool(inputs)  # pooling without reducing filters
        p4 = self.pool_out(p4)
        p4 = self.po_bn(p4)
        p4 = tf.keras.layers.ReLU()(p4)

        concat = tf.keras.layers.Concatenate(axis=3)

        return concat([p1, p2, p3, p4]) if p3 is not None else concat([p1, p2, p4])  # combinations of all feature maps


class InceptionModuleShrink(tf.keras.layers.Layer):
    """
    The shrinking Inception Module layer.
    Shrinks the current image resolution, but increases the number of filters.
    Contains one 3x3 and 5x5 convolution path each, and also a pooling path.
    Like in the normal inception module, the paths´ output feature maps are combined to one single output.
    """

    def __init__(self, conv_output_sizes, reduce_sizes, name):
        """
        :param conv_output_sizes: the output sizes (filter counts) for the 3x3 and 5x5 convolution paths,
            meaning their second convolution output channel count each
        :param reduce_sizes: the reduction sizes for the 3x3 and 5x5 convolution paths,
            so the output channel count for their first convolutions
        :param name: a name to supply for this layer
        """
        super(InceptionModuleShrink, self).__init__(name=name)

        # check constraint
        assert len(conv_output_sizes) == 2 == len(reduce_sizes)
        self.cos = conv_output_sizes
        self.rs = reduce_sizes

        # 3x3 convolution path
        self.reduce3 = tf.keras.layers.Conv2D(self.rs[0], (1, 1), padding='same')
        self.r3_bn = tf.keras.layers.BatchNormalization(axis=3)
        self.conv3 = tf.keras.layers.Conv2D(self.cos[0], (3, 3), (2, 2), 'same')
        self.c3_bn = tf.keras.layers.BatchNormalization(axis=3)
        # 5x5 convolution path
        self.reduce5 = tf.keras.layers.Conv2D(self.rs[1], (1, 1), padding='same')
        self.r5_bn = tf.keras.layers.BatchNormalization(axis=3)
        self.conv5 = tf.keras.layers.Conv2D(self.cos[1], (5, 5), (2, 2), 'same')
        self.c5_bn = tf.keras.layers.BatchNormalization(axis=3)
        # pooling path
        self.pool = tf.keras.layers.MaxPool2D((3, 3), (2, 2), padding='same')

    def call(self, inputs, **kwargs):
        # 3x3 convolution path
        p1 = self.reduce3(inputs)
        p1 = self.r3_bn(p1)
        p1 = tf.keras.layers.ReLU()(p1)
        p1 = self.conv3(p1)
        p1 = self.c3_bn(p1)
        p1 = tf.keras.layers.ReLU()(p1)
        # 5x5 convolution path
        p2 = self.reduce5(inputs)
        p2 = self.r5_bn(p2)
        p2 = tf.keras.layers.ReLU()(p2)
        p2 = self.conv5(p2)
        p2 = self.c5_bn(p2)
        p2 = tf.keras.layers.ReLU()(p2)
        # pooling path
        pool = self.pool(inputs)

        return tf.keras.layers.Concatenate(axis=3)([p1, p2, pool])  # combine output filters


def build_openface_model():
    """
    Builds the nn4.small2 OpenFace model, with about 3.74M trainable parameters (3733968)
    (exact model type was inferred from paper's parameter count).
    The model's output is a 128-sphere, a face embedding vector.
    This model is the basis for the paper´s OF 143/10 models.

    :return the built OpenFace NN4.small2.v1 model tf.keras.Model object
    """

    # input part
    inp = tf.keras.layers.Input((96, 96, 3))  # input is (aligned) RBG image pf 96x96
    x = tf.keras.layers.Conv2D(64, 7, 2, padding='same', name='First_Conv2D')(inp)  # 48x48 x 64
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPool2D(3, 2, padding='same')(x)  # 24x24  x 64
    x = LocalResponseNormalization()(x)

    # Inception 2 (output size 24x24 x 192)
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

    # assemble model
    model = tf.keras.Model(inputs=[inp], outputs=[x], name='Openface NN4.Small2.v1')
    model.summary()

    return model


def build_vgg_custom_part(bigger_class_n=False):
    """
    Builds the paper's additional model part for a VGG-based face DNN.
    The inputs are 4096-d face descriptors, gained from the standard VGG model.

    :param bigger_class_n: whether to use 143 instead of 10 classes
    :return the additional part of VGG 143/10 as tf.keras.Sequential object
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
    :return the additional part of OF 143/10 as tf.keras.Sequential object
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
    # model.summary()

    return model


def train_vgg_dnn(epochs=1, bigger_class_n=True):
    """
    Trains the complete custom VGG 143/10 model on the given dataset.
    Either starts training / fine-tuning from scratch, or continues with a found saved model state.

    :param epochs: how many training epochs long to train for this function call
    :param bigger_class_n: whether to train the VGG 143 model, instead of the VGG 10 model
        (also deciding which subset of the PubFig data is used)
    """

    # compose complete model
    vgg_base = get_original_vgg_model()
    for layer in vgg_base.layers:  # freeze VGG base layers (transfer learning)
        layer.trainable = False
    top_part = build_vgg_custom_part(bigger_class_n)
    class_suffix = '_143' if bigger_class_n else '_10'
    save_path = '../saved-models/vgg' + class_suffix + '.h5'

    # get saved weights, or start with new transfer learning (fine-tune top layers)
    try:
        model = tf.keras.models.load_model(save_path)
        print('Model state loaded. Continue training...')
    except OSError:
        model = tf.keras.Sequential([vgg_base, top_part], name='VGG' + class_suffix + '_complete')
        print('No saved weights found. Start training new model...')

    model.summary()

    # load dataset, rescale + resize images
    ds_path = '../data/pubfig/dataset_'  # local machine path
    remote_ds_path = '../../../../data-private/dataset_'  # use this one on remote workstation
    ds_path = remote_ds_path  # comment out when using on local machine
    if not bigger_class_n:
        ds_path += '10/'
    else:
        ds_path += '/'

    # get part of PubFig dataset, separated by classes; also scale pixel values and image size
    datagen = ImageDataGenerator(rescale=1. / 255)
    datagen = datagen.flow_from_directory(ds_path, (224, 224))
    '''
    WARNING: If there are mysterious duplicate files starting with '._' in the subclass directories, the program
    will crash.
    To remove the files, go to 'dataset_' and execute this command:
    find . -type f -name ._\* -exec rm {} \;
    '''

    # do training
    opt = tf.keras.optimizers.Adam(learning_rate=5e-4)  # can be adjusted, use smaller rate the more progressed
    model.compile(opt, 'categorical_crossentropy', ['accuracy'])
    losses = model.fit(datagen, epochs=epochs, ).history

    # save model state
    model.save(save_path)


def align_dataset_for_openface():
    """
    Uses Dlib (with Python bindings) to align the images of the PubFig dataset.
    This creates aligned outputs with transformed and cropped face images.
    """
    pass


def pretrain_openface_model(epochs=1):
    """
    Trains the OpenFace NN4.small2.v1 model, as preparation for the custom OF 143/10 models.
    Uses aligned images from the PubFig dataset.

    :param epochs: the amount of epochs to train the model this function call
    """

    try:
        model = tf.keras.models.load_model('../saved-models/openface.h5')
        print('Model loaded. Continue training:')
    except (ImportError, IOError):
        print('No model save found. Start training:')
        model = build_openface_model()

    # load aligned face images
    # TODO
    x, y = 0, 0

    # train model
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    pretain_loss = tfa.losses.TripletSemiHardLoss()
    model.compile(opt, pretain_loss, ['accuracy'])
    model.fit(x, y, epochs=epochs)

    # save after (continued) training
    model.save('../saved-models/openface.h5')


def train_of_dnn(epochs=1, bigger_class_n=True):
    """
    Trains the custom OF 143/10 model on the given dataset, based on a pretrained OpenFace model.
    Either starts training / fine-tuning from scratch, or continues with a found saved model state.

    :param epochs: how many training epochs long to train for this function call
    :param bigger_class_n: whether to train the OF 143 model, instead of the OF 10 model
        (also deciding which subset of the PubFig data is used)
    """

    # setup model
    save_path = '../saved-models/of' + '143' if bigger_class_n else '10' + '.h5'
    try:  # continue training
        model = tf.keras.models.load_model(save_path)
        print('Saved model state found. Continue training:')
    except (ImportError, IOError):
        print('No saved state for the complete OF' + '143' if bigger_class_n else '10' + 'model found.')
        try:  # OpenFace pretrained, start training OF 143/10
            base_model: tf.keras.Model = tf.keras.models.load_model('../saved-models/openface.h5')
            print('Pretrained OpenFace model loaded.')
            top_model = build_of_custom_part(bigger_class_n)
            for layer in base_model.layers:  # freeze base part layers?
                layer.trainable = False
            model = tf.keras.Model([base_model.input], [top_model.layers[-1]])
        except (ImportError, IOError):  # OpenFace not pretrained yet
            print('No pretrained OpenFace model found. Pretrain the OpenFace model first.')
            return

    # get data
    # TODO

    # train model
    opt = tf.keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(opt, 'categorical_crossentropy', ['accuracy'])

    # save model after training
    model.save(save_path)


def build_detector_model():
    """

    """

    model = tf.keras.Sequential()
    model.add(tf.keras.Input((14, 14, 512)))
    model.add(tf.keras.layers.Conv2D(196, (3, 3), padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPool2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(196, (3, 3), padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv2D(196, (3, 3), padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Flatten())  # 196
    model.add(tf.keras.layers.Dense(2, activation='softmax'))  # binary simplex

    model.summary()

    return model


if __name__ == '__main__':
    ''' UNCOMMENT IF USING ON WORKSTATION
    os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"]='4,5' '''
    #build_openface_model()
