import os
import random
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras.models import load_model

from attack.attacks_helpers import load_glasses_mask, merge_images_using_mask, pad_glasses_image, \
    strip_softmax_from_face_recognition_model, add_merger_to_generator, initialize_faceadder_physical, warp_image
from networks import dcgan, dcgan_utils, eyeglass_generator, eyeglass_discriminator
from networks.special_layers import LocalResponseNormalization, L2Normalization, InceptionModule, InceptionModuleShrink
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
    mask_img = load_glasses_mask(data_path, mask_path)

    for i, face_img in enumerate(face_samples_paths):
        # open face image and process it
        img = Image.open(data_path + rel_path + face_img)
        img = img.resize((224, 224))
        img = np.asarray(img)
        img = tf.convert_to_tensor(img)

        merged_img = merge_images_using_mask(data_path, img, pad_glasses_image(generated_eyeglasses[i]), mask=mask_img)
        merged_images.append(merged_img)

    return tf.stack(merged_images)


@tf.function
def compute_custom_loss(target: int, predictions, weight_factor: int = 1, apply_softmax: bool = False):
    """
    Computes a custom loss that is used instead of cross-entropy for the face recognition networks.
    This optimizes the gradients to focus on one specific target class.
    Minimizing the loss is the objective for dodging attacks, while it should be maximized for impersonation attacks.

    :param target: the target index, so n for target with index n / the n+1-th person of a set of target classes
    :param predictions: the logits that are output of the layer before the softmax (output) layer
        in a classification model, a tensor-like object of shape (half_batch_size, n_classes)
    :param weight_factor: an optional weighting factor for the target / evader class,
        should be the number of classes
    :param apply_softmax: whether to apply softmax to the logits to get probabilities;
        also shift the loss to have 0 as minimum (avoiding negative losses)
    :return: the custom loss weighing target and non-target predictions
    """

    if apply_softmax:
        predictions = tf.nn.softmax(predictions)  # logits -> probabilities
    target_logits = predictions[:, target]  # logits for the target  # logits_target
    logits_sum = tf.reduce_sum(predictions, axis=1)  # sum of all logits (along batch dimension)
    other_logits_sum = logits_sum - target_logits  # sum_i!=target logits_i
    res = (1 if apply_softmax else 0) + weight_factor * target_logits - other_logits_sum

    return tf.reduce_mean(res)  # average loss in batch


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


def do_attack_training_step(gen, dis, gen_ext, face_classifier, target: int,
                            real_glasses_a: tf.Tensor, real_glasses_b: tf.Tensor,
                            g_opt, d_opt, bs: int, kappa: float, dodging=True, use_ce_loss: bool = False,
                            verbose=False):
    """
    Performs one special training step to adjust the GAN for performing a dodging / impersonation attack.
    This requires the current optimizers, a given target, real glasses image, and of course the DCGAN model.
    Trains the DCGAN with glasses as well as attacker images.

    :param gen: the generator model, generates fake glasses
    :param dis: the discriminator model, critics glasses
    :param gen_ext: the generator model + added merger on top, generates merged face images with glasses
    :param face_classifier: the face recognition model, without softmax
    :param target: the target´s index
    :param real_glasses_a: a batch of real glasses images, sized according to bs
    :param real_glasses_b: another batch of real glasses images, sized according to bs
    :param g_opt: the generator´s optimizer object
    :param d_opt: the discriminator´s optimizer object
    :param bs: the training batch size
    :param kappa: a weighting factor to balance generator gradients gained from glasses and attacker images
    :param dodging: whether to train for a dodging attack, if false instead train for impersonation attack
    :param use_ce_loss: whether to use sparse categorical cross entropy loss instead of the special one
        for impersonation attacks (only experimental)
    :param verbose: whether to print additional information for the attack training step; useful for debugging
    :return: the updated generator model, the updated discriminator model, the updated extended generator model
     (updated with the same gradients as the normal generator), the updated generator optimizer,
     the updated discriminator optimizer, the discriminator´s objective, the face recognition net´s objective,
     and the custom face recognition loss
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
        dis_loss_b = dcgan_utils.get_gen_loss(dis_output)
        if verbose:
            print(50 * '-')
            print(f'The gen loss from dis: {dis_loss_b}')
            print(50 * '-')

        # switch to face recognition net

        print('Generating attack images...')
        attack_images = gen_ext(random_vectors_b, training=True)
        fc_output = face_classifier(attack_images, training=True)  # the logits as output

        custom_fc_loss = compute_custom_loss(target, fc_output)
        if not dodging and use_ce_loss:
            cel = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            truth = [target for _ in range(half_batch_size)]
            custom_fc_loss = cel(truth, fc_output)
        if verbose:
            print(90 * '=')
            print(f'The face classifier´s logits: {fc_output}')
            print(90 * '-')
        loss_objective = 'MINIMIZED' if (dodging or use_ce_loss) else 'MAXIMIZED'
        print(f'The special face classifier loss to be {loss_objective} : {custom_fc_loss}')
        print(90 * '-')

    # APPLY BLOCK: dis
    dis_gradients = d_tape.gradient(dis_loss_a, dis.trainable_variables)
    d_opt.apply_gradients(zip(dis_gradients, dis.trainable_variables))

    # APPLY BLOCK: gen
    # apply gradients from discriminator and face net to generator

    gen_gradients_glasses = g_tape.gradient(dis_loss_b, gen.trainable_variables)
    gen_gradients_attack = g_tape_s.gradient(custom_fc_loss, gen_ext.trainable_variables)
    if verbose:
        print(90 * '=')
        print(f'Gen gradients normal: {gen_gradients_glasses}')
        print(90 * '-')
        print(f'Gen gradients attack: {gen_gradients_attack}')
        print(90 * '-')
    if not dodging:  # reverse attack gradients for dodging attack to maximize loss (optimizers minimize)
        gen_gradients_attack = [-gr for gr in gen_gradients_attack]
    gen_gradients = join_gradients(gen_gradients_glasses, gen_gradients_attack, kappa)
    g_opt.apply_gradients(zip(gen_gradients, gen.trainable_variables))  # apply to original generator
    g_opt.apply_gradients(zip(gen_gradients, gen_ext.trainable_variables))  # also sync extended generator

    # compute objectives
    objective_d = tf.reduce_mean(dis_output)  # average confidence of the discriminator in glasses images (half fake)
    fc_output = tf.nn.softmax(fc_output, axis=1)  # probabilities, shape (half_batch_size, n_classes)
    objective_f = fc_output[:, target]  # face net´s confidence that images originate from target
    objective_f = tf.reduce_mean(objective_f)  # average over batch dimension
    if verbose:
        print(100 * '_')
    print('Attack iteration done.')

    return gen, dis, gen_ext, g_opt, d_opt, objective_d, objective_f, custom_fc_loss


def check_objective_met(data_path: str, gen, face_classifier, target: int, target_ds_tensors: List[tf.Tensor],
                        mask_path: str,
                        stop_prob: float, bs: int, fc_in_size=(224, 224), scale_to_polar: bool = False,
                        dodge: bool = True, physical: bool = False) -> [bool, float]:
    """
    Checks whether the attack objective has been yet met. It tries generated fake glasses with a face image dataset
    and checks whether the face recognition network can be fooled successfully.

    :param data_path: the path to the data directory
    :param gen: the generator model (of the DCGAN)
    :param face_classifier: the face recognition model, without softmax
    :param target: the target´s index
    :param target_ds_tensors: a list of the target (or impersonator) dataset´s images as tensors
    :param mask_path: relative path to mask image (from 'data')
    :param stop_prob: a stopping probability, related to the face net´s output target probabilities (a value in [0, 1])
    :param bs: the training batch size, must be an even number
    :param fc_in_size: the face classifier´s image input size
    :param scale_to_polar: whether to rescale the merged images´ value range to [-1., 1.]
    :param dodge: whether to check for a successful dodging attack (check for impersonation attack instead if false)
    :param physical: if the attack is physical
    :return: whether an attack could be performed successfully, and the best mean probability (confidence in target)
    """

    # sanity check assumptions
    assert bs % 2 == 0
    assert 0 <= stop_prob <= 1

    # help function per call
    def draw_random_image(imgs: tf.Tensor, scaled_to_polar: bool):
        """
        Draws a random image (merged face + glasses) from a given set.

        :param imgs: a tensor of merged images
        :param scaled_to_polar: whether the images´ value range were scaled to [-1. 1.]
        """
        random_show_img = imgs[random.randint(0, imgs.shape[0] - 1)]
        show_img_from_tensor(random_show_img, ([-1, 1] if scaled_to_polar else [0, 1]))

    # generate fake eyeglasses
    random_vectors = tf.random.normal([bs // 2, 25])
    glasses = gen.predict(random_vectors)
    mask = load_glasses_mask(data_path, mask_path)  # get mask tensor

    # get full target dataset (scaled to range [-1, 1])
    last_face_ims_inner_iter = None  # save last set of merged attack images here
    best_mean = 1.0 if dodge else 0.0  # track best mean probs

    n = len(target_ds_tensors)
    # zero pad glasses images
    glasses = pad_glasses_image(glasses)

    for i_g in range(bs // 2):  # iterate over generated glasses
        # generate new dataset instance for iteration

        # classify faces with eyeglasses
        probs = tf.Variable(tf.zeros((n,)))  # variable tensor that holds predicted probabilities
        # get current fake glasses
        g = glasses[i_g]

        for i_f in range(0, n, bs // 2):  # iterate over half-batches of faces
            n_iter = np.min([bs // 2, n - i_f])
            # take next half-batch of faces for inner iteration
            face_ims_iter = target_ds_tensors[i_f: i_f + min(i_f + (bs // 2), n)]
            # merge faces images and current glasses
            merged_ims = []

            for face_img in face_ims_iter:
                if physical:
                    face_img, flow = initialize_faceadder_physical(face_img, mask)  # compute transformation
                    glass_img = warp_image(g, flow, 1)  # warp glasses
                    face_img = face_img * 2  # to range [0., 2.]
                    merged_img = glass_img + face_img  # merge images
                else:
                    face_img = (face_img * 2) - 1  # scale to [-1., 1.]
                    # merge to image tensor size (n_iter, 224, 224, 3) with value range [0., 1.]
                    merged_img = merge_images_using_mask(data_path, face_img, g, mask=mask)

                merged_img = tf.image.resize(merged_img, fc_in_size)  # resize (needed for OF)
                if scale_to_polar:  # rescale to [-1., 1.] (needed for OF)
                    merged_img = (merged_img * 2) - 1
                merged_ims.append(merged_img)

            face_ims_iter = tf.stack(merged_ims)
            last_face_ims_inner_iter = face_ims_iter
            # classify the faces
            preds = face_classifier.predict(face_ims_iter)
            faces_preds = tf.nn.softmax(preds, axis=1)

            # write predicted confidences for target into probabilities
            for i_t in range(n_iter):
                target_preds = faces_preds[i_t, target]
                probs = probs[i_f + i_t].assign(target_preds)  # get target prediction probability

        # check if mean target probability in desired range
        mean_prob = tf.reduce_mean(probs).numpy()
        if (dodge and mean_prob < best_mean) or (not dodge and mean_prob > best_mean):
            best_mean = mean_prob
        # check success criterion
        if (mean_prob <= stop_prob and dodge) or (mean_prob >= stop_prob and not dodge):
            # pick random successful image and show it
            print(f'>>>>>>> mean_prob: {mean_prob}')
            draw_random_image(last_face_ims_inner_iter, scale_to_polar)  # show example for successful image

            return True, best_mean  # attack successful if face classifier fooled with at least one glasses image

    draw_random_image(last_face_ims_inner_iter, scale_to_polar)  # show one image per function call
    print(f'Best mean prob in iteration: {best_mean}')

    return False, best_mean  # no single successful attack


def produce_attack_stats_plots(loss_history: list, objective_histories: tuple, mp_history: list):
    """
    Plots a 2x2 diagram with one line plot for every history.
    :param loss_history: the history of the custom face classifier loss
    :param objective_histories: the histories of objective_d and objective_f, in a tuple
    :param mp_history: the history of the mean prob
    """

    # get epochs axis (x)
    n_epochs = len(loss_history)
    epoch_numbers = range(1, n_epochs + 1)

    # one subplot for every history
    fig, grid = plt.subplots(2, 2, figsize=(8, 6))
    grid[0, 0].plot(epoch_numbers, objective_histories[0])
    grid[0, 1].plot(epoch_numbers, objective_histories[1])
    grid[0, 0].title.set_text('Objective-D')
    grid[0, 1].title.set_text('Objective-F')
    grid[1, 0].plot(epoch_numbers, loss_history)
    grid[1, 0].title.set_text('Loss')
    grid[1, 1].plot(epoch_numbers, mp_history)
    grid[1, 1].title.set_text('Mean Prob')

    # plot and save
    plt.show()
    if not os.path.exists('../../saved-plots'):
        os.mkdir('../../saved-plots')
    now = time.time()
    fig.savefig('../../saved-plots/attack_stats__' + str(now) + '.png')


def execute_attack(data_path: str, target_path: str, mask_path: str, fc_img_size, g_path: str, d_path: str,
                   fn_path: str, ep: int, lr: float, kappa: float, stop_prob: float, bs: int,
                   target_index: int, vgg_not_of: bool, dodging: bool, physical: bool = False):
    """
    Executes an attack with all the given parameters. Checks success after every attack training epoch.

    :param data_path: the path to the 'data' directory
    :param target_path: the relative path of the target (or impersonator) directory from 'data'
    :param mask_path: the relative path of the mask from 'data'
    :param fc_img_size: the face classifier´s input size as iterable of two integers
    :param g_path: the path of the saved generator model
    :param d_path: the path of the saved discriminator model
    :param fn_path: the path of the saved face recognition model
    :param ep: how many epochs to execute the attack training loop
    :param lr: the learning rate for the optimizer´s of the generator and discriminator
    :param kappa: a weighting number in [0., 1.] for joining gradients of discriminator and face classifier
    :param stop_prob: the stopping probability that determines when an attack is deemed successful
    :param bs: the training batch size
    :param target_index: the index of the target in the given dataset (starting from 0)
    :param vgg_not_of: whether the used face classifier is a VGG, instead of OpenFace
    :param dodging: whether to execute a dodging attack, instead of an impersonation attack
    :param physical: whether the executed attack is physical (person is wearing real glasses model)
    """

    # load models and do some customization
    print('Loading models...')
    # load face classifier that needs to be fooled
    if vgg_not_of:
        face_model = load_model(fn_path)
    else:
        custom_objects = {'LocalResponseNormalization': LocalResponseNormalization,
                          'InceptionModule': InceptionModule,
                          'InceptionModuleShrink': InceptionModuleShrink,
                          'L2Normalization': L2Normalization}
        face_model = load_model(fn_path, custom_objects=custom_objects)
    face_model = strip_softmax_from_face_recognition_model(face_model)
    gen_model = eyeglass_generator.build_model()
    gen_model.load_weights(g_path)
    dis_model = eyeglass_discriminator.build_model()
    dis_model.load_weights(d_path)
    gen_model_ext = add_merger_to_generator(gen_model, data_path, target_path, bs // 2, fc_img_size,
                                            vgg_not_of, physical=physical)  # must be updated (synced) during attack
    print('All models loaded.')
    face_model.summary()

    # get glasses dataset to draw two half-batches from each training epoch (already shuffled and batched)
    print('Loading glasses dataset...')
    glasses_ds = dcgan.load_real_images(data_path, alt_bs=bs, sample_limit=1000)  # only part of dataset for attack
    print('Glasses dataset ready.')

    # get target/impersonator dataset
    if dodging:
        print('Loading target dataset...')
    else:
        print('Loading impersonator dataset...')
    target_img_files = os.listdir(data_path + target_path)
    n = len(target_img_files)
    data_tensors = []
    for img_file in target_img_files:
        img = tf.image.decode_png(tf.io.read_file(data_path + target_path + img_file), channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)  # scale is [0., 1.]
        img = tf.image.resize(img, (224, 224))  # resize to standard
        data_tensors.append(img)
    target_ds_tensors = data_tensors
    print('Other dataset ready.')

    # perform special training
    print('Perform special training:')
    current_ep = 1
    g_opt, d_opt = tf.keras.optimizers.Adam(learning_rate=lr), tf.keras.optimizers.Adam(learning_rate=lr)
    bmp = 1.0 if dodging else 0.0
    loss_history = []
    mp_history = []
    obj_d_history, obj_f_history = [], []
    done = False  # success flag

    while current_ep <= ep:
        print(f'======= Attack training epoch {current_ep}. =======')

        # get real glasses as half-batches
        glasses = glasses_ds.take(1)  # take one batch (dataset is repeated infinitely)
        glasses = [p for p in glasses]  # unravel
        glasses = tf.stack(glasses)
        glasses = tf.reshape(glasses, glasses.shape[1:])
        glasses_a, glasses_b = glasses[:bs // 2], glasses[bs // 2:]
        if current_ep == 1:
            print(glasses_a.shape)

        # execute one attack step
        gen_model, dis_model, gen_model_ext, g_opt, d_opt, obj_d, obj_f, l = do_attack_training_step(gen_model,
                                                                                                     dis_model,
                                                                                                     gen_model_ext,
                                                                                                     face_model,
                                                                                                     target_index,
                                                                                                     glasses_a,
                                                                                                     glasses_b,
                                                                                                     g_opt, d_opt,
                                                                                                     bs, kappa, dodging)
        # after copying the model instances, both versions of the generator model are equal and updated
        # the discriminator is also updated; but the face classifier stays exactly the same!

        print(f'Dis. average trust in manipulated fake glasses + real glasses: {obj_d.numpy()}')
        print(f'Face classifier´s average trust that attack images belong to target: {obj_f.numpy()}')

        # check whether attack already successful
        print('Checking attack progress...')
        obj_d_history.append(obj_d)
        obj_f_history.append(obj_f)
        loss_history.append(l)
        done, mp = check_objective_met(data_path, gen_model, face_model, target_index, target_ds_tensors, mask_path,
                                       stop_prob,
                                       bs,
                                       fc_img_size, not vgg_not_of, dodging)
        # update mean prob record
        mp_history.append(mp)
        if (dodging and mp < bmp) or (not dodging and mp > bmp):
            bmp = mp
            print(f'<<< Current best mean prob: {bmp}')
        if done:
            attack_name = 'Dodging' if dodging else 'Impersonation'
            print(f'<<<<<< {attack_name} attack successful! >>>>>>')
            break
        else:
            print('No attack success yet. \n\n')

        current_ep += 1

    # if attack failed, provide best score
    if current_ep > ep and not done:
        print(f'Attack failed. \nBest mean prob: {bmp}')
    if current_ep > 4:
        produce_attack_stats_plots(loss_history, (obj_d_history, obj_f_history), mp_history)  # draw statistics plots


if __name__ == '__main__':
    # demo for merged images
    dap = setup_params(True)

    # run to see example of merged attack images
    generator = eyeglass_generator.build_model()
    generator.load_weights('../../saved-models/gweights')
    inputs = tf.random.normal((3, 25))
    outputs = generator.predict(inputs)
    tap = 'pubfig/dataset_aligned/Gisele_Bundchen/aligned/'
    maskp = 'eyeglasses/eyeglasses_mask_6percent.png'
    face_imgs = os.listdir(dap + tap)
    for i, fim in enumerate(random.sample([dap + tap + fi for fi in face_imgs], 3)):
        fimg = tf.image.decode_png(tf.io.read_file(fim), channels=3)
        fimg = tf.image.convert_image_dtype(fimg, tf.float32)
        fimg = tf.image.resize(fimg, (224, 224))
        fimg = (fimg * 2) - 1
        gimg = outputs[i]
        gimg = pad_glasses_image(gimg)
        mimg = merge_images_using_mask(dap, fimg, gimg, maskp)
        show_img_from_tensor(mimg, [0, 1])
