import eyeglass_generator as gen
import eyeglass_discriminator as dis
import face_nets as fns
import dcgan
import dcgan_utils
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

    :param img_a: the first image (face), that a part of the other image should be overlayed on (range [0, 255])
    :param img_b: the second image (glasses), that should (in part) be overlayed on the other one (range [-1, 1])
    :param mask_path: the relative path (from data) to a filter mask that determines which pixels of image B
        should be put onto image A - the mask has only black and white pixels that are interpreted in a boolean manner
    :return: a tensor where the pixels of image B are put over those in image A
        as specified by the mask (range [0, 255])
    """

    # load mask and convert it to boolean mask tensor
    mask_path = data_path + mask_path
    mask_img = Image.open(mask_path)
    mask_img = np.asarray(mask_img)
    mask_img = tf.convert_to_tensor(mask_img)
    mask_img = tf.cast(mask_img, tf.float32)
    mask_img = mask_img / 255  # scale to [0, 1]

    glasses_image = (img_b + 1) * 127.5  # scale glasses image to have same range as face image
    face_image = tf.cast(img_a, tf.float32)

    # merge images
    masked_glasses_img = tf.math.multiply(glasses_image, mask_img)  # cancel out pixels that are outside of mask area
    invert_mask_img = -(mask_img - 1)  # flip 0 and 1 in mask
    masked_face_img = tf.math.multiply(face_image, invert_mask_img)  # remove pixels that will be replaced
    merged_img = masked_face_img + masked_glasses_img
    merged_img = tf.cast(merged_img, tf.uint8)

    return merged_img


def merge_face_images_with_fake_glasses(rel_path, generator: tf.keras.Model, n_samples):
    """
    Draws some random samples from the given face image directory (relative to data path),
    and puts them together with generated fake eyeglasses.

    :param rel_path: the relative path of the face image directory (from 'data')
    :param generator: the generator model used for generating fake eyeglasses
    :param n_samples: how many samples overall to output
    :return: a specified amount of faces with generated eyeglasses on them, with size 224x224 (as one tensor)
    """

    # sample n face images
    face_ds_paths = os.listdir(data_path + rel_path)
    face_samples_paths = random.sample(face_ds_paths, n_samples)

    # generate n fake glasses
    random_vectors = tf.random.normal([n_samples, 25])
    generated_eyeglasses = generator.predict(random_vectors)
    mask_path = 'eyeglasses/eyeglasses_mask_6percent.png'
    merged_images = []

    for i, face_img in enumerate(face_samples_paths):
        # open face image and process it
        img = Image.open(data_path + rel_path + face_img)
        img = img.resize((224, 224))
        img = np.asarray(img)
        img = tf.convert_to_tensor(img)

        merged_img = merge_images_using_mask(img, pad_glasses_image(generated_eyeglasses[i]), mask_path)
        merged_images.append(merged_img)

    return tf.stack(merged_images)


def compute_custom_loss(target, predictions):
    """
    Computes a custom loss that is used instead of cross-entropy for the face recognition networks.
    This optimizes the gradients to focus on one specific target class.
    """
    pass


def do_attack_training_step(gen, dis, facenet, real_glasses_a, real_glasses_b, target_path, g_opt, d_opt, bs, target):

    # update discriminator
    with tf.GradientTape() as d_tape:
        random_vectors = tf.random.normal([bs/2, 25])
        fake_glasses = gen.predict(random_vectors)

        # train discriminator on real and fake glasses
        real_output = dis(real_glasses_a, training=True)
        fake_output = dis(fake_glasses, training=True)
        dis_loss_a = dcgan_utils.get_discrim_loss(fake_output, real_output)
    dis_gradients = d_tape.gradient(dis_loss_a, dis.trainable_variables)
    d_opt.apply_gradients(zip(dis_gradients, dis.trainable_variables))

    # update generator
    with tf.GradientTape() as g_tape:

        # pass another half-batch of fake glasses to compute gradients for generator
        random_vectors = tf.random.normal([bs / 2, 25])
        other_fake_glasses = gen.predict(random_vectors)
        fake_output = dis(other_fake_glasses, training=False)  # get discriminator output for generator
        real_output = dis(real_glasses_b, training=False)
        dis_loss_b = dcgan_utils.get_discrim_loss(fake_output, real_output)

        # switch to face recognition net
        attack_images = merge_face_images_with_fake_glasses(target_path, gen, bs / 2)
        facenet_output = facenet.predict(attack_images)
        custom_facenet_loss = compute_custom_loss(target, facenet_output)

    gen_gradients_glasses = g_tape.gradient(dis_loss_b, gen.trainable_variables)
    gen_gradients_attack = g_tape.gradient(custom_facenet_loss, gen.trainable_variables)
    gen_gradients_attack = - gen_gradients_attack  # for dodging attack
    g_opt.apply_gradients(zip(gen_gradients_glasses, gen.trainable_variables))
    g_opt.apply_gradients(zip(gen_gradients_attack, gen.trainable_variables))

    return g_opt, d_opt


# TODO implement dodging attack


# TODO implement impersonation attack


if __name__ == '__main__':
    gen = gen.build_model()
    gen.load_weights('../saved-models/gweights')
    examples = merge_face_images_with_fake_glasses('/pubfig/dataset_aligned/Danny_Devito/aligned/', gen, 10)
