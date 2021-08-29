import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

import dcgan_utils
import eyeglass_generator as gen_module
from attacks_helpers import load_mask, merge_images_using_mask, pad_glasses_image
from setup import setup_params


def scale_tensor_to_std(tensor: tf.Tensor, vrange: list) -> tf.Tensor:
    """
    Scales a given tensor to the range [0, 255], also casting to UINT8.

    :param tensor: the tensor, assuming float values
    :param vrange: the range of the values in the input tensor, given in a list of two numerical values
    :return: the tensor with scaled integer values
    """
    img_scaled = tf.add(tf.cast(tensor, tf.float32), tf.constant(-vrange[0], dtype=tf.float32))
    img_scaled *= (255. / (vrange[1] + (-vrange[0])))
    img_scaled = tf.cast(img_scaled, tf.uint8)

    return img_scaled


def show_img_from_tensor(img: tf.Tensor, vrange):
    """
    Shows an image with matplotlib.

    :param img: the image to be shown, represented by a tf.Tensor
    :param vrange: the range of the tensor values given as iterable of two values
    """

    # scale to range [0, 255]
    img_scaled = scale_tensor_to_std(img, vrange)

    # show image
    plt.figure()
    plt.imshow(img_scaled)
    plt.show()


@DeprecationWarning
def merge_face_images_with_fake_glasses(data_path: str, rel_path, gen: tf.keras.Model, n_samples: int) -> tf.Tensor:
    """
    Draws some random samples from the given face image directory (relative to data path),
    and puts them together with generated fake eyeglasses.
    NOTE: now only used for demonstration purpose, do NOT use it in Tensorflow models

    :param data_path: the path to the data directory
    :param rel_path: the relative path of the face image directory (from 'data')
    :param gen: the generator model used for generating fake eyeglasses
    :param n_samples: how many samples overall to output
    :return: a specified amount of faces with generated eyeglasses on them, with size 224x224 (as one tensor)
    """

    # sample n face images
    face_ds_paths = os.listdir(data_path + rel_path)
    face_samples_paths = random.sample(face_ds_paths, n_samples)

    # generate n fake glasses
    random_vectors = tf.random.normal([n_samples, 25])
    generated_eyeglasses = gen(random_vectors)
    merged_images = []

    # preload mask
    mask_path = 'eyeglasses/eyeglasses_mask_6percent.png'
    mask_img = load_mask(data_path, mask_path)

    for i, face_img in enumerate(face_samples_paths):
        # open face image and process it
        img = Image.open(data_path + rel_path + face_img)
        img = img.resize((224, 224))
        img = np.asarray(img)
        img = tf.convert_to_tensor(img)

        merged_img = merge_images_using_mask(data_path, img, pad_glasses_image(generated_eyeglasses[i]), mask=mask_img)
        merged_images.append(merged_img)

    return tf.stack(merged_images)


def compute_custom_loss(target: int, predictions):
    """
    Computes a custom loss that is used instead of cross-entropy for the face recognition networks.
    This optimizes the gradients to focus on one specific target class.
    Minimizing the loss is the objective for dodging attacks, while it should be maximized for impersonation attacks.

    :param target: the target index, so n for target with index n / the n+1-th person of a set of target classes
    :param predictions: the logits that are output of the layer before the softmax (output) layer
        in a classification model, a tensor-like object
    :return: the custom loss weighing target and non-target predictions
    """

    preds = tf.reduce_mean(predictions, 0)  # average over half-batch: (hbs, n_classes) -> (n_classes)
    target_logit = preds[target]
    other_logits_sum = tf.subtract(tf.reduce_sum(preds), tf.constant(target_logit))
    res = tf.subtract(target_logit, other_logits_sum)

    return res


def join_gradients(gradients_a, gradients_b, kappa: float):
    """
    Joins two sets of computed gradients using a weighting factor and returns one set of gradients of the same size.

    :param gradients_a: a list of gradient tensors
    :param gradients_b: another list of gradient tensors, with the same respective sizes as in gradients_a
    :param kappa: a number between 0 and 1, weighs the two different gradient sets
    :return: a list of the tensors of joined gradients, with the same shape as the original sets
    """
    assert 0 <= kappa <= 1  # check that kappa in correct range
    joined_gradients = []

    for i_gr in range(len(gradients_a)):  # join every pair of gradient tensors
        t_a, t_b = gradients_a[i_gr], gradients_b[i_gr]  # gradient tensors
        norm_a, norm_b = tf.norm(t_a), tf.norm(t_b)  # norms

        # scale one gradient tensor, depending on relation of their norms
        if norm_a > norm_b:
            t_a = t_a * (norm_b / norm_a)
        else:
            t_b = t_b * (norm_a / norm_b)

        joined_gradient = kappa * t_a + (1 - kappa) * t_b  # weighted sum is the joined gradient
        joined_gradients.append(joined_gradient)

    return joined_gradients


# @tf.function  # NOTE: if decorated with tf.Function, this function cannot be properly debugged
def do_attack_training_step(gen, dis, gen_ext, facenet, target: int,
                            real_glasses_a: tf.Tensor, real_glasses_b: tf.Tensor,
                            g_opt, d_opt, bs: int, kappa: float, dodging=True, verbose=False):
    """
    Performs one special training step to adjust the GAN for performing a dodging / impersonation attack.
    This requires the current optimizers, a given target, real glasses image, and of course the DCGAN model.
    Trains the DCGAN with glasses as well as attacker images.

    :param gen: the generator model, generates fake glasses
    :param dis: the discriminator model, critics glasses
    :param gen_ext: the generator model + added merger on top, generates merged face images with glasses
    :param facenet: the face recognition model, without softmax
    :param target: the target´s index
    :param real_glasses_a: a batch of real glasses images, sized according to bs
    :param real_glasses_b: another batch of real glasses images, sized according to bs
    :param g_opt: the generator´s optimizer object
    :param d_opt: the discriminator´s optimizer object
    :param bs: the training batch size
    :param kappa: a weighting factor to balance generator gradients gained from glasses and attacker images
    :param dodging: whether to train for a dodging attack, if false instead train for impersonation attack
    :param verbose: whether to print additional information for the attack training step; useful for debugging
    :return: the updated generator model, the updated discriminator model, the updated extended generator model
     (updated with the same gradients as the normal generator), the updated generator optimizer,
     the updated discriminator optimizer, the discriminator´s objective, the face recognition net´s objective
    """

    # assert batch size assumptions
    assert bs % 2 == 0
    half_batch_size = bs // 2

    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape, tf.GradientTape() as g_tape_s:

        # update discriminator phase:

        # generate fake glasses
        random_vectors_a = tf.random.normal([half_batch_size, 25])
        fake_glasses = gen(random_vectors_a)

        # train discriminator on real and fake glasses
        real_output = dis(real_glasses_a, training=True)
        fake_output = dis(fake_glasses, training=True)
        dis_loss_a = dcgan_utils.get_discrim_loss(fake_output, real_output)

        # update generator phase:

        # pass another half-batch of fake glasses to compute gradients for generator
        random_vectors_b = tf.random.normal([half_batch_size, 25])
        other_fake_glasses = gen(random_vectors_b)

        fake_output = dis(other_fake_glasses, training=True)  # get discriminator output for generator
        real_output = dis(real_glasses_b, training=True)
        dis_output = tf.concat([fake_output, real_output], 0)
        dis_loss_b = dcgan_utils.get_gen_loss(fake_output)
        if verbose:
            print(50 * '-')
            print(f'The gen loss from dis: {dis_loss_b}')
            print(50 * '-')

        # switch to face recognition net

        print('Generating attack images...')
        attack_images = gen_ext(random_vectors_b, training=True)
        facenet_output = facenet(attack_images, training=True)  # the logits as output
        assert len(gen.trainable_variables) == len(gen_ext.trainable_variables)

        custom_facenet_loss = compute_custom_loss(target, facenet_output)
        if verbose:
            print(90 * '=')
            print(f'The facenet logits: {facenet_output}')
            print(90 * '-')
        print(f'The special facenet loss: {custom_facenet_loss}')
        print(90 * '-')

    # APPLY BLOCK: dis
    dis_gradients = d_tape.gradient(dis_loss_a, dis.trainable_variables)
    d_opt.apply_gradients(zip(dis_gradients, dis.trainable_variables))

    # APPLY BLOCK: gen
    # apply gradients from discriminator and face net to generator

    gen_gradients_glasses = g_tape.gradient(dis_loss_b, gen.trainable_variables)
    gen_gradients_attack = g_tape_s.gradient(custom_facenet_loss, gen_ext.trainable_variables)
    if verbose:
        print(90 * '=')
        print(f'Gen gradients normal: {gen_gradients_glasses}')
        print(90 * '-')
        print(f'Gen gradients attack: {gen_gradients_attack}')
        print(90 * '-')
    if dodging:
        gen_gradients_attack = [-gr for gr in gen_gradients_attack]  # reverse attack gradients for dodging attack
    gen_gradients = join_gradients(gen_gradients_glasses, gen_gradients_attack, kappa)
    g_opt.apply_gradients(zip(gen_gradients, gen.trainable_variables))  # apply to original generator
    g_opt.apply_gradients(zip(gen_gradients, gen_ext.trainable_variables))  # also sync extended generator

    # compute objectives
    objective_d = tf.reduce_mean(dis_output)  # average confidence of the discriminator in glasses images (half fake)
    facenet_output = tf.nn.softmax(facenet_output, axis=0)  # probabilities, shape (half_batch_size, n_classes)
    objective_f = facenet_output[:, target]  # face net´s confidence that images originate from target
    objective_f = tf.reduce_mean(objective_f)  # average over batch dimension
    if verbose:
        print(100 * '_')
    print('Attack iteration done.')

    return gen, dis, gen_ext, g_opt, d_opt, objective_d, objective_f


def check_objective_met(data_path: str, gen, facenet, target: int, target_path: str, mask_path: str,
                        stop_prob: float, bs: int, facenet_in_size=(224, 224), scale_to_polar=False,
                        dodge=True) -> bool:
    """
    Checks whether the attack objective has been yet met. It tries generated fake glasses with a face image dataset
    and checks whether the face recognition network can be fooled successfully.

    :param data_path: the path to the data directory
    :param gen: the generator model (of the DCGAN)
    :param facenet: the face recognition model, without softmax
    :param target: the target´s index
    :param target_path: relative path to target dataset (from 'data')
    :param mask_path: relative path to mask image (from 'data')
    :param stop_prob: a stopping probability, related to the face net´s output target probabilities (a value in [0, 1])
    :param bs: the training batch size, must be an even number
    :param facenet_in_size: the face net´s image input size
    :param scale_to_polar: whether to rescale the merged images´ value range to [-1., 1.]
    :param dodge: whether to check for a successful dodging attack (check for impersonation attack instead if false)
    :return: whether an attack could be performed successfully
    """

    # sanity check assumptions
    assert bs % 2 == 0
    assert 0 <= stop_prob <= 1

    # generate fake eyeglasses
    random_vectors = tf.random.normal([bs // 2, 25])
    glasses = gen.predict(random_vectors)
    mask = load_mask(data_path, mask_path)  # get mask tensor

    # get full target dataset (scaled to range [-1, 1])
    img_files = os.listdir(data_path + target_path)
    n = len(img_files)
    data_tensors = []
    for img_file in img_files:
        img = tf.image.decode_png(tf.io.read_file(data_path + target_path + img_file), channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (224, 224))  # resize to standard
        data_tensors.append(img)

    for i_g in range(bs // 2):  # iterate over generated glasses
        # generate new dataset instance for iteration
        face_ds = tf.data.Dataset.from_tensor_slices(data_tensors)
        face_ds = face_ds.shuffle(1000)
        # classify faces with eyeglasses
        probs = tf.Variable(tf.zeros((n,)))  # variable tensor that holds predicted probabilities
        # get current fake glasses
        g = glasses[i_g]
        g = pad_glasses_image(g)  # zero pad
        last_face_ims_inner_iter = None  # save last set of merged attack images here

        for i_f in range(0, n, bs // 2):  # iterate over half-batches of faces
            n_iter = np.min([bs // 2, n - i_f])
            face_ims_iter = face_ds.take(n_iter)  # take next subset of faces for inner iteration
            # merge faces images and current glasses
            merged_ims = []
            for face_img in face_ims_iter:
                face_img = (face_img * 2) - 1  # scale to [-1., 1.]
                # merge to image tensor size (n_iter, 224, 224, 3) with value range [0., 1.]
                mimg = merge_images_using_mask(data_path, face_img, g, mask=mask)
                mimg = tf.image.resize(mimg, facenet_in_size)  # resize (needed for OF)
                if scale_to_polar:  # rescale to [-1., 1.] (needed for OF)
                    mimg = (mimg * 2) - 1
                merged_ims.append(mimg)
            face_ims_iter = tf.stack(merged_ims)
            last_face_ims_inner_iter = face_ims_iter
            # classify the faces
            faces_preds = tf.nn.softmax(facenet.predict(face_ims_iter))

            # write predicted confidences for target into probabilities
            for i_t in range(n_iter):
                probs = probs[i_f + i_t].assign(faces_preds[i_t, target])  # get target prediction probability

        # check if mean target probability in desired range
        mean_prob = tf.reduce_mean(probs).numpy()
        if (mean_prob <= stop_prob and dodge) or (mean_prob >= stop_prob and not dodge):
            # pick random successful image and show it
            random_show_img = last_face_ims_inner_iter[random.randint(0, last_face_ims_inner_iter.shape[0])]
            show_img_from_tensor(random_show_img, ([-1, 1] if scale_to_polar else [0, 1]))

            return True  # attack successful if facenet fooled with at least one glasses image

    return False  # no single successful attack


def execute_attack(data_path: str, target_path: str, mask_path: str, fn_img_size, g_path: str, d_path: str,
                   fn_path: str, n_bigger_class: bool, ep: int, lr: float, kappa: float, stop_prob: float, bs: int,
                   target_index: int, dodging: bool):
    """
    Executes an attack with all the given parameters. Checks success after every attack training epoch.

    :param data_path: the path to the 'data' directory
    :param target_path: the relative path of the target directory from 'data'
    :param mask_path: the relative path of the mask from 'data'
    :param fn_img_size: the facenet´s input size as iterable of two integers
    :param g_path: the path of the saved generator model
    :param d_path: the path of the saved discriminator model
    :param fn_path: the path of the saved face recognition model
    :param n_bigger_class: whether the face predictor has 143 output classes (instead of only 10)
    :param ep: how many epochs to execute the attack training loop
    :param lr: the learning rate for the optimizer´s of the generator and discriminator
    :param kappa: a weighting number in [0., 1.] for joining gradients of discriminator and facenet
    :param stop_prob: the stopping probability that determines when an attack is deemed successful
    :param bs: the training batch size
    :param target_index: the index of the target in the given dataset (starting from 0)
    :param dodging: whether to execute a dodging attack, instead of an impersonation attack
    """
    pass
    # TODO implement generic


if __name__ == '__main__':
    # demo for merged images
    dap = setup_params(True)

    # run to see example of merged attack images
    generator = gen_module.build_model()
    generator.load_weights('../saved-models/gweights')
    inputs = tf.random.normal((3, 25))
    outputs = generator.predict(inputs)
    tap = 'pubfig/dataset_aligned/Gisele_Bundchen/aligned/'
    map = 'eyeglasses/eyeglasses_mask_6percent.png'
    face_imgs = os.listdir(dap + tap)
    for i, fim in enumerate(random.sample([dap + tap + fi for fi in face_imgs], 3)):
        fimg = tf.image.decode_png(tf.io.read_file(fim), channels=3)
        fimg = tf.image.convert_image_dtype(fimg, tf.float32)
        fimg = tf.image.resize(fimg, (224, 224))
        fimg = (fimg * 2) - 1
        gimg = outputs[i]
        gimg = pad_glasses_image(gimg)
        mimg = merge_images_using_mask(dap, fimg, gimg, map)
        show_img_from_tensor(mimg, [0, 1])
