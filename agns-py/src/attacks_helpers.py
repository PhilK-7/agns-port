import os
import time

import tensorflow as tf
from PIL import Image
import numpy as np

crop_coordinates = [53, 25, 53 + 64, 25 + 176]


def pad_glasses_image(glass: tf.Tensor):
    """
    Pads a generated glasses image as reverse transformation to the cropping that was applied to the original data.

    :param glass: the glasses image, represented as tensor (range [-1, 1])
    :return: a bigger tensor that represents 224x224 pixels, with black padding added
    """

    img = tf.Variable(tf.fill([224, 224, 3], -1.))  # initialize black 224x224 image

    # assign all values from the generated glasses image
    for i in range(crop_coordinates[0], crop_coordinates[2]):
        for j in range(crop_coordinates[1], crop_coordinates[3]):
            img = img[i, j].assign(glass[i - crop_coordinates[0], j - crop_coordinates[1]])

    return img


def load_mask(data_path: str, mask_path: str) -> tf.Tensor:
    """
    Loads the mask for glasses images.

    :param data_path: the path to the data directory
    :param mask_path: the relative path to the mask image (from 'data')
    :return: the mask as tensor, with float values in [0, 1]
    """
    mask_path = data_path + mask_path  # full path
    mask_img = Image.open(mask_path)
    mask_img = np.asarray(mask_img)
    mask_img = tf.convert_to_tensor(mask_img)
    mask_img = tf.cast(mask_img, tf.float32)
    mask_img = mask_img / 255  # scale to [0, 1]

    return mask_img


def merge_images_using_mask(data_path: str, img_a: tf.Tensor, img_b: tf.Tensor,
                            mask_path: str = '', mask: tf.Tensor = None) -> tf.Tensor:
    """
    Merges two images A and B, using a provided filter mask.

    :param data_path: the path to the data directory
    :param img_a: the first image (face), that a part of the other image should be overlayed on (range [0, 255]);
        assumed to be of shape (224, 224)
    :param img_b: the second image (glasses), that should (in part) be overlayed on the other one (range [-1, 1]);
        the same shape as img_a
    :param mask_path: the relative path (from data) to a filter mask that determines which pixels of image B
        should be put onto image A - the mask has only black and white pixels that are interpreted in a boolean manner;
        can be left empty if mask is already supplied as tensor
    :param mask: alternatively, the mask is already loaded and can be supplied as tensor here
        (use this when needing the same mask repeatedly for efficiency)
    :return: a tensor where the pixels of image B are put over those in image A
        as specified by the mask (range [0, 255]), of image shape 224x224
    """

    # load mask and convert it to boolean mask tensor
    if mask is None:
        mask_img = load_mask(data_path, mask_path)
    else:
        mask_img = mask

    glasses_image = tf.add(img_b, tf.ones(img_b.shape)) * 127.5  # scale glasses image to have same range as face image
    # TEST
    timg = glasses_image.numpy()
    timg = timg.astype(np.uint8)
    save_img_from_tensor(timg, 'merge')
    #
    face_image = tf.cast(img_a, tf.float32)

    # merge images
    masked_glasses_img = tf.math.multiply(glasses_image, mask_img)  # cancel out pixels that are outside of mask area
    invert_mask_img = -(mask_img - tf.ones(mask_img.shape))  # flip 0 and 1 in mask
    masked_face_img = tf.math.multiply(face_image, invert_mask_img)  # remove pixels that will be replaced
    merged_img = masked_face_img + masked_glasses_img
    merged_img = tf.cast(merged_img, tf.uint8)

    return merged_img


def scale_integer_to_zero_one_tensor(tensor: tf.Tensor) -> tf.Tensor:
    """
    Receives a tensor of (unsigned) integer values in [0, 255], and scales the values to [0., 1.].

    :param tensor: a tf.Tensor with int values in range between 0 and 255
    :return: a tensor of the same size as the input, with tf.float32 values between 0 and 1
    """
    t = tf.cast(tensor, tf.float32)
    t = t / 255.

    return t


def convert_to_numpy_slice(imgs, i: int) -> np.ndarray:
    """
    Receives images represented by a tensor (value range [-1, 1]),
     and converts one image specified by index to a ndarray with range [0, 255].

    :param imgs: the image as a tensor-like object
    :param i: the index of the image in the input tensor
    :return: a NumPy array with scaled integer values that represents a slice of the input tensor
    """
    imgs: np.ndarray = imgs.numpy()
    img = imgs[i]
    img = (img + 1) * 127.5
    img = img.astype(np.uint8)

    return img


def save_img_from_tensor(img, name: str, use_time: bool = True):
    """
    Saves an image given by a NumPy array to a file in the out directory. Might be useful for debugging purposes.

    :param img: the image represented as ndarray
    :param name: a name (prefix) to save the image in the 'out' directory
    :param use_time: whether to append a timestamp to the output file name
    """

    img = Image.fromarray(img)  # numpy array -> pillow image
    if not os.path.exists('../out'):  # setup 'out' folder if missing
        os.mkdir('../out')
    filename = '../out/' + name + '_' + (str(time.time() if use_time else '')) + '.png'  # compose name
    img.save(filename)  # save to file


def add_merger_to_generator(generator, data_path, target_path, n_inputs, output_size=(224, 224)):
    """
    Receives a generator model and adds a GlassesFacesMerger on top of it, given the parameters.

    :param generator: the generator model object (likely a tf.keras.models.Sequential)
    :param data_path: path to the 'data' directory
    :param target_path: relative path of the target´s dataset (from 'data')
    :param n_inputs: the fixed number of inputs, predetermines what the entire models input batch size must be
    :param output_size: the desired image output size
    :return: a new generator model that has merged faces with glasses as output images
    """
    model = tf.keras.models.Sequential(generator.layers, name='Gen_Merge')  # copy rest layers
    from special_layers import GlassesFacesMerger
    model.add(GlassesFacesMerger(data_path, target_path, n_inputs, output_size))  # add merging layer

    return model


def strip_softmax_from_face_recognition_model(facenet, n_classes):
    """
    Removes the last layer (softmax classification output) from a face recognition model,
    and adds an equivalent last dense layer without softmax activation that just outputs the logits.

    :param facenet: the face recognition model
    :param n_classes: the number of output classes
    """
    model = tf.keras.models.Sequential(facenet.layers[:-1])
    model.add(tf.keras.layers.Dense(n_classes, name='Logits'))  # dense layer without activation
    model.layers[-1].set_weights(facenet.layers[-1].get_weights())  # copy classification layer weights
    #model.summary()

    return model
