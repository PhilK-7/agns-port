import os
import random

import numpy as np
import tensorflow as tf
from PIL import Image

from attacks_helpers import load_mask, merge_images_using_mask, pad_glasses_image
from dcgan import load_real_images


class LocalResponseNormalization(tf.keras.layers.Layer):
    """
    The Local Response Normalization layer. Normalizes high frequency features with radial masks.
    Applies the parameters needed in OpenFace NN4.small2.v1.
    """

    def __init__(self):
        super(LocalResponseNormalization, self).__init__()

    def get_config(self):
        conf = super().get_config().copy()
        return conf

    def call(self, inputs, **kwargs):
        return tf.nn.local_response_normalization(inputs, alpha=1e-4, beta=0.75)


class L2Pooling(tf.keras.layers.Layer):
    """
    The L2 Pooling layer. Computes a kind of Euclidean norm using average pooling.
    Uses pooling size 3, as needed in OpenFace NN4.small2.v1.
    """

    def __init__(self):
        super(L2Pooling, self).__init__()

    def get_config(self):
        conf = super().get_config().copy()
        return conf

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

    def get_config(self):
        conf = super().get_config().copy()
        return conf

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
        self.ul2p = use_l2_pooling

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

    def get_config(self):
        conf = super().get_config().copy()
        conf.update({
            'conv_output_sizes': self.cos,
            'reduce_sizes': self.rs,
            'use_l2_pooling': self.ul2p
        })

        return conf

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

    def get_config(self):
        conf = super().get_config().copy()
        conf.update({
            'conv_output_sizes': self.cos,
            'reduce_sizes': self.rs,
        })

        return conf

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


class GlassesFacesMerger(tf.keras.layers.Layer):
    """
    Layer for executing dodging / impersonation attacks.
    Receives generated fake glasses meant for fooling a face recognition system.
    Merges those glasses with face images of the given target.
    """

    def __init__(self, data_path: str, target_path: str, n_inputs: int, output_size: tuple = (224, 224)):
        """
        The constructor for the merger layer.

        :param data_path: path to the 'data' directory
        :param target_path: relative path of the target´s image dataset, based on data_path
        :param n_inputs: the number of samples in an input batch, fixed here
        :param output_size: specifies the desired output size of the images of this layer, a duple of two integers
        """
        super(GlassesFacesMerger, self).__init__(name='Merger')

        self.dap = data_path
        self.tap = target_path
        tp = data_path + target_path
        self.target_ims = [tp + im for im in os.listdir(tp)]  # get paths in target ds
        assert len(output_size) == 2
        self.mask_img = load_mask(data_path, 'eyeglasses/eyeglasses_mask_6percent.png')  # load mask tensor
        self.outsize = output_size
        self.n_inputs = n_inputs

    def get_config(self):
        conf = super().get_config().copy()
        # conf.update(dict...) if any parameters given

        return conf

    def call(self, inputs, **kwargs):
        """
        Merges received glasses images with some face images of the given target.

        :param inputs: a tensor of generated glasses, expected shape (n_inputs, 64, 176, 3), values range [-1, 1]
        :return: a tensor of shape (n_inputs, output_size, 3)
        """

        if inputs.shape[0] != self.n_inputs:  # handle special case when Sequential API calls this function to add layer
            return tf.zeros((self.n_inputs, self.outsize[0], self.outsize[1], 3))
        face_ds = load_real_images(self.dap, self.tap, self.n_inputs, self.outsize)
        merged_images = []

        # merge faces and glasses
        face_ims = face_ds.take(1)  # one batch of face images
        for i, face_img in enumerate(face_ims):

            # NOTE: output range here is [0, 255]
            merged_img = merge_images_using_mask(self.dap, face_img, pad_glasses_image(inputs[i]),
                                                 mask=self.mask_img)

            # resize result again if desired image size is not 224x224
            if self.outsize != (224, 224):  # TODO test block; do not use pillow
                img = merged_img.numpy()
                img = Image.fromarray(img)
                img = img.resize(self.outsize)
                img = np.asarray(img)
                img = tf.convert_to_tensor(img)
                merged_img = img
            merged_images.append(merged_img)

        # combine results and scale to range needed for face recognition networks
        result = tf.stack(merged_images)
        result = tf.reshape(result, result.shape[1:])

        return result
