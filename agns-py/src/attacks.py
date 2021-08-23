import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

import dcgan_utils
import eyeglass_generator as gen_module
from attacks_helpers import load_mask, merge_images_using_mask, pad_glasses_image, convert_to_numpy_slice, \
    save_img_from_tensor


def scale_tensor_to_std(tensor: tf.Tensor, vrange: list) -> tf.Tensor:
    """
    Scales a given tensor to the range [0, 255], also casting to UINT8.

    :param tensor: the tensor, assuming float values
    :param vrange: the range of the values in the input tensor, given in a list of two numerical values
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


#@DeprecationWarning
def merge_face_images_with_fake_glasses(data_path: str, rel_path, gen: tf.keras.Model, n_samples: int) -> tf.Tensor:
    """
    Draws some random samples from the given face image directory (relative to data path),
    and puts them together with generated fake eyeglasses.

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


@DeprecationWarning
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
    preds = tf.reduce_mean(predictions, 0)  # average over half-batch

    return preds[target]

    '''
    preds = tf.reduce_mean(predictions, 0)  # average over half-batch: (hbs, n_classes) -> (n_classes)
    target_logit = preds[target]
    other_logits_sum = tf.subtract(tf.reduce_sum(preds), tf.constant(target_logit))
    res = tf.subtract(target_logit, other_logits_sum)

    return res'''


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


# @tf.function  # NOTE: if decorated with tf.Function, this function cannot be properly debugged
def do_attack_training_step(data_path: str, gen, dis, gen_ext, facenet, target_path: str, target: int,
                            real_glasses_a: tf.Tensor, real_glasses_b: tf.Tensor,
                            g_opt, d_opt, bs: int, kappa: float, dodging=True, verbose=True) \
        -> (tf.keras.optimizers.Adam, tf.keras.optimizers.Adam, tf.Tensor, tf.Tensor):
    """
    Performs one special training step to adjust the GAN for performing a dodging / impersonation attack.
    This requires the current optimizers, a given target, real glasses image, and of course the DCGAN model.
    Trains the DCGAN with glasses as well as attacker images.

    :param data_path: the path to the data directory
    :param gen: the generator model, generates fake glasses
    :param dis: the discriminator model, critics glasses
    :param gen_ext: the generator model + added merger on top, generates merged face images with glasses
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
    :param verbose: whether to print additional information for the attack training step
    :return g_opt: the updated generator optimizer
    :return d_opt: the updated discriminator optimizer
    :return objective_d: the discriminator´s objective
    :return objective_f: the face recognition net´s objective
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
        if verbose:
            for i in range(3):
                glass = convert_to_numpy_slice(other_fake_glasses, i)
                save_img_from_tensor(glass, 'fake-b')
        fake_output = dis(other_fake_glasses, training=True)  # get discriminator output for generator
        real_output = dis(real_glasses_b)
        dis_output = tf.concat([fake_output, real_output], 0)
        dis_loss_b = dcgan_utils.get_gen_loss(fake_output)
        if verbose:
            print(50 * '-')
            # print(f'The gen loss from dis: {dis_loss_b}')
            print(50 * '-')

        # switch to face recognition net, but remove softmax

        print('Generating attack images...')
        # TODO Connection somwhere <= attackimages broken?
        # TODO test
        composed_model = tf.keras.models.Sequential([gen_ext, facenet])
        facenet_output = composed_model(random_vectors_b, training=True)
        #
        
        attack_images = gen_ext(random_vectors_b, training=True)  # merged images
        if verbose:
            mims = (attack_images * 2) - 1
            for i in range(4):
                mimg = convert_to_numpy_slice(mims, random.randint(0, half_batch_size-1))
                save_img_from_tensor(mimg, 'merged')
        # TODO whatif image sizes are 96
        facenet_output = facenet(attack_images, training=True)  # the logits as output
        '''sw = tf.Variable(tf.fill((143, ), 0.5/(143 - 1)))  # TODO dehardcode
        sw = sw[target].assign(0.5)'''
        alt_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        targets = tf.fill((half_batch_size,), target)
        custom_facenet_loss = alt_loss(targets, facenet_output)
        #print(g_tape_s.watched_variables())
        if verbose:
            print(90 * '=')
            #print(f'The facenet logits: {facenet_output}')
            print(90 * '-')
            print(f'The special facenet loss: {custom_facenet_loss}')
            print(90 * '-')

    # APPLY BLOCK: dis
    dis_gradients = d_tape.gradient(dis_loss_a, dis.trainable_variables)
    d_opt.apply_gradients(zip(dis_gradients, dis.trainable_variables))

    # APPLY BLOCK: gen
    # apply gradients from discriminator and face net to generator
    # TODO test:
    for i in range(len(facenet.layers)):
        facenet.layers[i].trainable = True
    for i in range(len(gen_ext.layers)):
        gen_ext.layers[i].trainable = True
    #
    gen_gradients_glasses = g_tape.gradient(dis_loss_b, gen.trainable_variables)
    gen_gradients_attack = g_tape_s.gradient(custom_facenet_loss, gen_ext.trainable_variables)
    if verbose:
        print(90 * '=')
        #print(f'Gen gradients normal: {gen_gradients_glasses}')
        print(90 * '-')
        print(f'Gen gradients attack: {gen_gradients_attack}')
        print(90 * '-')
    if dodging:
        gen_gradients_attack = [-gr for gr in gen_gradients_attack]  # reverse attack gradients for dodging attack
    gen_gradients = join_gradients(gen_gradients_glasses, gen_gradients_attack, kappa)
    g_opt.apply_gradients(zip(gen_gradients, gen.trainable_variables))

    # compute objectives
    objective_d = tf.reduce_mean(dis_output)  # average confidence of the discriminator in fake images
    objective_f = facenet_output[target]  # face net´s confidence that images originate from target
    if verbose:
        print(100 * '_')

    return g_opt, d_opt, objective_d, objective_f


def check_objective_met(data_path: str, gen, facenet, target: int, target_path: str, mask_path: str,
                        stop_prob: float, bs: int, dodge=True) -> bool:
    """
    Checks whether the attack objective has been yet met. It tries generated fake glasses with a face image dataset
    and checks whether the face recognition network can be fooled successfully.

    :param data_path: the path to the data directory
    :param gen: the generator model (of the DCGAN)
    :param facenet: the face recognition model
    :param target: the target´s index
    :param target_path: relative path to target dataset (from 'data')
    :param mask_path: relative path to mask image (from 'data')
    :param stop_prob: a stopping probability, related to the face net´s output target probabilities (a value in [0, 1])
    :param bs: the training batch size, must be an even number
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
    img_files = os.listdir(target_path)
    data_tensors = []
    for img_file in img_files:
        img = tf.image.decode_png(tf.io.read_file(data_path + target_path + img_file), channels=3)
        data_tensors.append(img)

    for i_g in range(bs // 2):  # iterate over generated glasses
        # generate new dataset instance for iteration
        face_ds = tf.data.Dataset.from_tensor_slices(data_tensors)
        face_ds = face_ds.shuffle(1000)
        # classify faces with eyeglasses
        n = tf.data.experimental.cardinality(face_ds)
        probs = tf.Variable(tf.zeros((n, 1)))  # variable tensor that holds predicted probabilities
        # get current fake glasses
        g = glasses[i_g]  # TODO need to also resize?!

        for i_f in range(0, n, bs // 2):
            n_iter = np.min([bs // 2, n - i_f])
            face_ims_iter = face_ds.take(n_iter)  # take next subset of faces for inner iteration
            # merge faces images and current glasses
            merged_ims = []
            for face_img in face_ims_iter:
                merged_ims.append(merge_images_using_mask(data_path, face_img, g, mask=mask))
            face_ims_iter = tf.stack(merged_ims)
            # classify the faces
            faces_preds = facenet.predict(face_ims_iter)

            # check if mean target probability in desired range
            for i_t in range(n_iter):
                probs[i_f + i_t] = faces_preds[i_t, target]  # get target prediction probability
            mean_prob = tf.reduce_mean(probs)
            if (mean_prob <= stop_prob and dodge) or (mean_prob >= stop_prob and not dodge):
                return True  # attack successful

    return False  # no single successful attack


# TODO implement dodging attack


# TODO implement impersonation attack

# TODO show attack results
def show_attack_results():
    pass


if __name__ == '__main__':
    # LE BOILERPLATE SHIAT
    # set parameters
    USE_REMOTE = False  # set depending whether code is executed on remote workstation or not
    if USE_REMOTE:
        os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = '2'
        dap = os.path.expanduser('~') + '/storage-private/data/'
    else:
        dap = '../data/'

    # run to see example of merged attack images
    generator = gen_module.build_model()
    generator.load_weights('../saved-models/gweights')
    examples = merge_face_images_with_fake_glasses(dap, 'pubfig/dataset_aligned/Gisele_Bundchen/aligned/',
                                                   generator, 10)
    show_img_from_tensor(examples[0], [0, 255])
    show_img_from_tensor(examples[1], [0, 255])
    show_img_from_tensor(examples[2], [0, 255])
