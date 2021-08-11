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


def merge_images_using_mask(img_a: tf.Tensor, img_b: tf.Tensor, mask_path: str):
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


def merge_face_images_with_fake_glasses(rel_path, generator: tf.keras.Model, n_samples: int):
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


def compute_custom_loss(target: int, predictions: tf.Tensor):
    """
    Computes a custom loss that is used instead of cross-entropy for the face recognition networks.
    This optimizes the gradients to focus on one specific target class.
    Minimizing the loss is the objective for dodging attacks, while it should be maximized for impersonation attacks.

    :param target: the target index, so n for target with index n / the n+1-th person of a set of target classes
    :param predictions: the logits that are output of the layer before the softmax (output) layer
        in a classification model, a tensor
    """
    target_logit = predictions[target]
    other_logits_sum = tf.reduce_sum(predictions) - target_logit

    return target_logit - other_logits_sum


def join_gradients(gradients_a: tf.Tensor, gradients_b: tf.Tensor, kappa: float) -> tf.Tensor:
    """
    Joins two sets of computed gradients using a weighting factor and returns one set of gradients of the same size.

    :param gradients_a: a tensor of gradients
    :param gradients_b: another tensor of gradients, has the same shape as gradients_a
    :param kappa: a number between 0 and 1, weighs the two different gradient sets
    :return: a tensor of joined gradients, with the same shape as the original sets
    """
    assert 0 <= kappa <= 1  # check that kappa in correct range
    gradients = tf.Variable(gradients_a)  # copy gradients shape

    # compute joined gradients
    for i in range(gradients.shape[0]):
        d1 = tf.Variable(gradients_a[i])
        d2 = tf.Variable(gradients_b[1])
        norm_1 = tf.norm(tf.reshape(d1, np.prod(d1.shape)))
        norm_2 = tf.norm(tf.reshape(d2, np.prod(d2.shape)))

        if norm_1 > norm_2:
            d1 = d1 * (norm_2 / norm_1)
        else:
            d2 = d2 * (norm_1 / norm_2)
        gradients[i] = kappa * d1 + (1 - kappa) * d2

    # convert and check
    gradients = tf.convert_to_tensor(gradients)
    assert gradients.shape == gradients_a.shape

    return gradients


def do_attack_training_step(gen, dis, facenet, target_path, target, real_glasses_a, real_glasses_b, g_opt, d_opt, bs,
                            kappa, dodging=True) \
        -> (tf.keras.optimizers.Adam, tf.keras.optimizers.Adam, tf.Tensor, tf.Tensor):
    """
    Performs one special training step to adjust the GAN for performing a dodging / impersonation attack.
    This requires the current optimizers, a given target, real glasses image, and of course the DCGAN model.
    Trains the DCGAN with glasses as well as attacker images.

    :param gen: the generator model
    :param dis: the discriminator model
    :param facenet: the face recognition model
    :param target_path: the path to the target´s image directory, relative to 'data'
    :param target: the target´s index
    :param real_glasses_a: a batch of real glasses images, sized according to bs
    :param real_glasses_b: another batch of real glasses images, sized according to bs
    :param g_opt: the generator´s optimizer object
    :param d_opt: the discriminator´s optimizer object
    :param bs: the training batch size
    :param kappa: a weighting factor to balance generator gradients gained from glasses and attacker images
    :param dodging: whether to train for a dodging attack, if false instead train for impersonation attack
    :return g_opt: the updated generator optimizer
    :return d_opt: the updated discriminator optimizer
    :return objective_d: the discriminator´s objective
    :return objective_f: the face recognition net´s objective
    """

    # update discriminator
    with tf.GradientTape() as d_tape:
        random_vectors = tf.random.normal([bs / 2, 25])
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
        dis_output = tf.concat([fake_output, real_output])
        dis_loss_b = dcgan_utils.get_discrim_loss(fake_output, real_output)

        # switch to face recognition net
        attack_images = merge_face_images_with_fake_glasses(target_path, gen, bs / 2)
        facenet_cut = tf.keras.models.Sequential(facenet.layers[:-1])  # TODO also works with non-linear models?
        # TODO whatif image sizes are 96
        facenet_logits_output = facenet_cut.predict(attack_images)  # the logits as output
        custom_facenet_loss = compute_custom_loss(target, facenet_logits_output)
        facenet_output = facenet.predict(attack_images)

    # apply gradients from discriminator and face net to generator
    gen_gradients_glasses = g_tape.gradient(dis_loss_b, gen.trainable_variables)
    gen_gradients_attack = g_tape.gradient(custom_facenet_loss, gen.trainable_variables)
    if dodging:
        gen_gradients_attack = - gen_gradients_attack  # reverse attack gradients for dodging attack
    gen_gradients = join_gradients(gen_gradients_glasses, gen_gradients_attack, kappa)
    g_opt.apply_gradients(zip(gen_gradients, gen.trainable_variables))

    # compute objectives
    objective_d = tf.reduce_mean(dis_output)  # average confidence of the discriminator in fake images
    objective_f = facenet_output[target]  # face net´s confidence that images originate from target

    return g_opt, d_opt, objective_d, objective_f


def check_objective_met(gen, facenet, target_path: str, bs) -> bool:
    """

    """

    # generate fake eyeglasses
    random_vectors = tf.random.normal([bs / 2, 25])
    glasses = gen.predict(random_vectors)




# TODO implement dodging attack


# TODO implement impersonation attack


if __name__ == '__main__':
    gen = gen.build_model()
    gen.load_weights('../saved-models/gweights')
    examples = merge_face_images_with_fake_glasses('/pubfig/dataset_aligned/Danny_Devito/aligned/', gen, 10)
