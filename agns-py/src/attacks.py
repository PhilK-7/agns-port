import eyeglass_generator as gen
import eyeglass_discriminator as dis
import face_nets as fns
import dcgan
import os
import random
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

data_path = '../data/'
crop_coordinates = [53, 25, 53 + 64, 25 + 176]


def show_img_from_tensor(img: tf.Tensor, vrange):
    """
    Shows an image with matplotlib.

    :param img: the image to be shown, represented by a tf.Tensor
    :param vrange: the range of the tensor values given as iterable of two values
    """

    # scale to range [0, 255]
    img_scaled = tf.add(tf.cast(img, tf.float32), tf.constant(-vrange[0], dtype=tf.float32))
    img_scaled *= (255. / (vrange[1] + (-vrange[0])))
    img_scaled = tf.cast(img_scaled, tf.uint8)
    print(img_scaled)

    # show image
    plt.figure()
    plt.imshow(img_scaled)
    plt.show()


def pad_glasses_image(glass):
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


def merge_images_using_mask(img_a: tf.Tensor, img_b: tf.Tensor, mask_path):
    """
    Merges two images A and B, using a provided filter mask.

    :param img_a: the first image (face), that a part of the other image should be overlayed on
    :param img_b: the second image (glasses), that should (in part) be overlayed on the other one
    :param mask_path: the relative path (from data) to a filter mask that determines which pixels of image B
        should be put onto image A - the mask has only black and white pixels that are interpreted in a boolean manner
    """

    # load mask and convert it to boolean mask tensor

    # merge images


def merge_face_images_with_fake_glasses(rel_path, generator: tf.keras.Model, n_samples):
    """
    Draws some random samples from the given face image directory (relative to data path),
    and puts them together with generated fake eyeglasses.

    :param rel_path: the relative path of the face image directory (from 'data')
    :param generator: the generator model used for generating fake eyeglasses
    :param n_samples: how many samples overall to output
    :return: a specified amount of faces with generated eyeglasses on them, with size 224x224
    """

    # sample n face images
    face_ds_paths = os.listdir(data_path + rel_path)
    face_samples_paths = random.sample(face_ds_paths, n_samples)

    # generate n fake glasses
    random_vectors = tf.random.normal([n_samples, 25])
    generated_eyeglasses = generator.predict(random_vectors)
    print(random_vectors[0])
    show_img_from_tensor(pad_glasses_image(generated_eyeglasses[0]), [-1, 1])

    for face_img in face_samples_paths:
        # open image and process it
        img = Image.open(data_path + rel_path + face_img)
        img = img.resize((224, 224))
        img = np.asarray(img)
        img = tf.convert_to_tensor(img)

    # TODO mask out pixels to replace with help of mask, generate fake glasses, and merge them ...


# TODO implement dodging attack


# TODO implement impersonation attack


# TODO implement custom DCGAN training step (see AGN in Matlab Code)

if __name__ == '__main__':
    gen = gen.build_model()
    gen.load_weights('../saved-models/gweights')
    merge_face_images_with_fake_glasses('/pubfig/dataset_aligned/Danny_Devito/aligned/', gen, 10)
