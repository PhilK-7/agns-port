import math
import os
import random
import time
import imageio
import shutil
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
from tensorflow.python.framework.errors_impl import NotFoundError

from networks import eyeglass_discriminator, eyeglass_generator, dcgan_utils
from setup import setup_params

'''
Some parts were taken from official tutorials:
https://www.tensorflow.org/tutorials/generative/dcgan, and
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
'''

BATCH_SIZE = 32


def preprocess_real_glasses_images(data_path: str):
    """
    Preprocesses the dataset of eyeglasses (cropping).
    This enables faster loading of this dataset.

    :param data_path: the path to the data directory
    """

    # get image paths and info
    path = data_path + 'eyeglasses/'
    new_path = path + 'cropped/'
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    img_files = os.listdir(path)

    # info for showing progress bar
    max_index = len(img_files)
    current_prog_str = ''
    current_index = 0
    start_time = time.time()

    print('Preprocessing images...')

    for file_path in img_files:
        if file_path.endswith('.png'):
            file_path = os.path.join(path, file_path)

            # show progress bar
            prog_str = dcgan_utils.display_custom_loading_bar('Processing', current_index, max_index)
            if prog_str != current_prog_str:
                current_prog_str = prog_str
                print(current_prog_str)

            # crop image
            img = Image.open(file_path)
            img = img.crop((25, 53, 201, 117))  # crop to correct size
            img_matrix = np.asarray(img)

            # save in new file
            img = Image.fromarray(img_matrix)
            filename = file_path.split('/')[-1]
            img.save(new_path + filename)

            current_index += 1

    # needed time
    end_time = time.time()
    total_time = round(end_time - start_time) + 1
    print(f'Preprocessing all images took {total_time} seconds.')


def load_real_images(data_path: str, alt_path=None, alt_bs=None, sample_limit: int = -1, resize_shape=None):
    """
    Loads the training images for the DCGAN from path 'data/eyeglasses/cropped' at the same level.
    This transforms them into an efficient Tensorflow Dataset, with image values ranging in [-1, 1].
    Alternatively can load other image datasets.

    :param data_path: the path to the data directory
    :param alt_path: an optional alternative path to load another image dataset instead
    :param alt_bs: optional different specified batch size
    :param sample_limit: whether to use only a specified amount of sample instead of the whole dataset
    :param resize_shape: optional target resized shape, must be a list of two integer values if given
    :return: the images as Tensorflow PrefetchDataset
    """

    # get paths
    ds_path = data_path + ('eyeglasses/cropped/' if alt_path is None else alt_path)
    img_files = os.listdir(ds_path)
    # limit if specified
    if sample_limit != -1:
        assert sample_limit > 0
        random.shuffle(img_files)
        img_files = img_files[:sample_limit]  # take only sample_limit random samples
    data_tensors = []

    # load images and transform to needed range
    for img_name in img_files:
        img = tf.image.decode_png(tf.io.read_file(ds_path + img_name), channels=3)
        if resize_shape is not None:
            img = tf.image.resize(img, resize_shape)
        img = tf.image.convert_image_dtype(img, tf.float32)  # ATTENTION: this also scales to range [0, 1]
        img = img * 2 - 1  # scale [-1., 1.]
        data_tensors.append(img)

    # make Tensorflow dataset
    ds = tf.data.Dataset.from_tensor_slices(data_tensors)

    ds = ds.shuffle(1000).repeat(-1)  # shuffle and repeat infinitely
    ds = ds.batch(BATCH_SIZE if alt_bs is None else alt_bs)  # make batches
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds


def generate_samples(generator, epoch, rseed=42):
    """
    Generates a set of generator output samples from a set of (fixed) random vector inputs.
    For every epoch, nine outputs each are generated, and plotted into a single image.
    The plots are saved at 'saved-plots/samples/' with file names that include UNIX time and the overall epoch.

    :param generator: the DCGAN's generator model
    :param epoch: which total epoch this is called at, for visualization purposes
    :param rseed: a random seed, used for generating random vectors
    """

    # get the random vectors and their current predictions
    np.random.seed(rseed)
    fix_vector = np.random.standard_normal((9, 25))  # get the 9 random generator input vectors
    preds = (generator(fix_vector, training=False) + 1) / 2  # generator inference to generate samples, out range [0, 1]
    fig = plt.figure(figsize=(3, 3))

    # plot predictions
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(preds[i, :, :, :])
        plt.axis('off')

        if i == 0:
            plt.title(f'Generator @ E{epoch}')

    # save plot
    if os.path.exists('../../saved-plots/samples'):
        n = len(os.listdir('../../saved-plots/samples')) + 1  # the n-th samples image
    else:
        n = 1
        os.mkdir('../../saved-plots/samples')
    plt.savefig(f'../../saved-plots/samples/samples_{round(time.time())}_epoch{n}.png')
    # do not draw later
    plt.clf()
    plt.close(fig)


def generate_samples_gif():
    """
    Generates a GIF from the images in 'saved-plots/samples', in chronological order.
    The resulting GIF will be saved at 'saved-plots/samples-history.gif'.
    """

    file_path = '../../saved-plots/samples-history.gif'
    img_paths = sorted(os.listdir('../../saved-plots/samples/'))

    with imageio.get_writer(file_path, mode='I', fps=6) as writer:
        for path in img_paths:
            img = imageio.imread(os.path.join('../../saved-plots/samples/', path))

            writer.append_data(img)


def update_loss_dataframe(g_loss, d_loss):
    """
    Updates the loss pandas dataframe that keeps track of the DCGAN models' losses over time,
    found at 'saved-plots/losses.csv'. Creates the file if it doesn't exist.

    :param g_loss: last generator loss
    :param d_loss: last discriminator loss
    """

    new_data = pd.DataFrame(np.array([[g_loss, d_loss]]), columns=['GLoss', 'DLoss'])
    try:
        df = pd.read_csv('../../saved-plots/losses.csv', index_col=[0])  # get dataframe
        df = pd.concat([df, new_data])
    except FileNotFoundError:  # first training / new training
        df = new_data

    df.to_csv('../../saved-plots/losses.csv')  # write back


def plot_losses():
    """
    Plots the DCGAN's losses and saves the plot to 'saved-plots/dcgan_loss_history.png'.
    The pandas dataframe in 'saved-plots/losses.csv' must exist.
    """

    try:
        # get dataframe and extract info
        df = pd.read_csv('../../saved-plots/losses.csv', index_col=[0])
        epochs_so_far = len(df.index)
        iters_list = range(1, epochs_so_far + 1)
        g_losses = df['GLoss']
        d_losses = df['DLoss']

        # plot the data
        plt.plot(iters_list, g_losses, label='Generator')
        plt.plot(iters_list, d_losses, label='Discriminator')
        plt.xlabel('Epochs ')
        plt.ylabel('Model loss')
        plt.legend(loc='upper right')
        plt.title('DCGAN losses history')

        # save plot to file and show it
        plt.savefig('../../saved-plots/dcgan_loss_history.png')
        plt.show()

    except FileNotFoundError:  # no dataframe found
        print('No losses computed yet.')


def train_dcgan(data_path, n_epochs, start_fresh=False, epochs_save_period=3):
    """
    Trains the DCGAN that should learn to generate / discriminate glasses.
    It assumes there is a directory 'saved-models' at the same level, where model weights are loaded and saved to.
    Also, the losses and the current progress are tracked, and visualized with other functions.

    :param data_path: the path to the data directory
    :param n_epochs: for how many epochs to train (this session)
    :param start_fresh: whether the DCGAN training process should be started anew
    :param epochs_save_period: after how many epochs each the model should be checkpointed (also saved at session end)
    """

    # get models
    g_model = eyeglass_generator.build_model()
    d_model = eyeglass_discriminator.build_model()
    if not start_fresh:  # load parameters from previous training
        try:
            g_model.load_weights('../../saved-models/gweights')
            d_model.load_weights('../../saved-models/dweights')
            print('>>>>>>> DCGAN weights loaded.')
        except NotFoundError:
            print('>>>>>>> No weights found, using fresh initialized weights!')
    else:
        print('>>>>>>> Starting training anew!')
        # new weights; remove outdated files
        loss_hist_path = '../../saved-plots/losses.csv'
        if os.path.exists(loss_hist_path):
            os.remove(loss_hist_path)
        gend_samples_path = '../../saved-plots/samples/'
        if os.path.exists(gend_samples_path):
            shutil.rmtree(gend_samples_path)
            os.makedirs(gend_samples_path)

    # define optimizer, same as described in paper
    gen_optimizer = tf.keras.optimizers.Adam(2e-4)
    discrim_optimizer = tf.keras.optimizers.Adam(2e-4)
    g_losses, d_losses = [], []

    # custom training procedure function
    @tf.function
    def training_step(images, bs=BATCH_SIZE):
        noise = np.random.standard_normal((bs, 25))  # random input vector for generator

        with tf.GradientTape() as gen_tape, tf.GradientTape() as discrim_tape:
            generated_images = g_model(noise, training=True)  # generate fake images

            # get outputs for discriminator
            real_output = d_model(images, training=True)
            fake_output = d_model(generated_images, training=True)

            # compute losses
            gen_loss = dcgan_utils.get_gen_loss(fake_output)
            discrim_loss = dcgan_utils.get_discrim_loss(real_output, fake_output)

        # compute and apply gradients
        gen_gradients = gen_tape.gradient(gen_loss, g_model.trainable_variables)
        discrim_gradients = discrim_tape.gradient(discrim_loss, d_model.trainable_variables)
        gen_optimizer.apply_gradients(zip(gen_gradients, g_model.trainable_variables))
        discrim_optimizer.apply_gradients(zip(discrim_gradients, d_model.trainable_variables))

        return gen_loss, discrim_loss

    # get data
    print('Fetching dataset...')
    real_image_dataset = load_real_images(data_path)  # load all real images
    num_samples = len(os.listdir(data_path + 'eyeglasses/cropped/'))
    num_batches = math.ceil(num_samples / BATCH_SIZE)  # number of training data batches

    # determine total epoch number
    try:
        epochs_so_far = len(pd.read_csv('../../saved-plots/losses.csv', index_col=[0]).index)
    except FileNotFoundError:
        epochs_so_far = 0

    # training loop
    loading_bar = ''
    for epoch in range(epochs_so_far, epochs_so_far + n_epochs):
        print(f'Epoch {epoch + 1}:')
        epoch_start_time = time.time()
        batch_index = 0
        ep_g_loss_sum, ep_d_loss_sum = 0, 0

        for batch in real_image_dataset:  # mini-batch training
            nlb = dcgan_utils.display_custom_loading_bar('Training', batch_index, num_batches)
            if nlb != loading_bar:
                loading_bar = nlb
                print(nlb)
            g_loss, d_loss = training_step(batch)  # one training iteration for this batch

            # observe error on that batch
            ep_g_loss_sum += g_loss
            ep_d_loss_sum += d_loss
            batch_index += 1

        # checkpoint both parts of the model
        if epoch % epochs_save_period == 0:
            g_model.save_weights('../../saved-models/gweights')
            d_model.save('../../saved-models/dweights')
            print('Model state saved.')

        # evaluate epoch
        avg_g_loss = ep_g_loss_sum / num_batches
        g_losses.append(avg_g_loss)
        avg_d_loss = ep_d_loss_sum / num_batches
        d_losses.append(avg_d_loss)
        epoch_end_time = time.time()
        epoch_time = round(epoch_end_time - epoch_start_time) + 1
        generate_samples(g_model, epoch + 1)  # also create new samples image
        print(f'Avg. generator loss: {avg_g_loss}, '
              f'avg. discriminator loss: {avg_d_loss}')
        update_loss_dataframe(avg_g_loss, avg_d_loss)  # add losses to losses.csv
        print(f'Epoch lasted for {epoch_time} seconds.')

    # save weights, generate samples, and plot loss history
    g_model.save_weights('../../saved-models/gweights')
    d_model.save('../../saved-models/dweights')
    generate_samples_gif()
    plot_losses()


if __name__ == '__main__':
    # run to train NEW (will overwrite old!) DCGAN on glasses images
    custom_objects = {'MiniBatchDiscrimination': eyeglass_discriminator.MiniBatchDiscrimination}

    # set parameters
    dap = setup_params(True)

    # set parameters accordingly to current training need
    train_dcgan(dap, 5, start_fresh=False)
