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
    face_image = tf.cast(img_a, tf.float32)

    # merge images
    masked_glasses_img = tf.math.multiply(glasses_image, mask_img)  # cancel out pixels that are outside of mask area
    invert_mask_img = -(mask_img - tf.ones(mask_img.shape))  # flip 0 and 1 in mask
    masked_face_img = tf.math.multiply(face_image, invert_mask_img)  # remove pixels that will be replaced
    merged_img = masked_face_img + masked_glasses_img
    merged_img = tf.cast(merged_img, tf.uint8)

    return merged_img
