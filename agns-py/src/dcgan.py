import eyeglass_generator
import eyeglass_discriminator
import tensorflow as tf
import numpy as np
import pathlib as pal
from PIL import Image
import os
import net_utils
import math
from tensorflow.python.framework.errors_impl import NotFoundError

'''
Some parts were taken from official tutorials:
https://www.tensorflow.org/tutorials/generative/dcgan, and
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
'''

LABEL_REAL = 0.9  # soft label (see paper)
LABEL_FAKE = 0
BATCH_SIZE = 48


def load_real_images(limit_to_first=1000):
    path = '../data/eyeglasses/'
    img_files = os.listdir(path)
    max_index = len(img_files)
    matrix = None
    current_prog_str = ''
    current_index = 0
    print('Loading images...')
    for img_file in img_files:

        if limit_to_first != -1 and current_index >= limit_to_first:  # when limit is used
            break
        if img_file.endswith('.png'):
            file_path = os.path.join(path, img_file)

            # show progress bar
            prog_str = net_utils.display_custom_loading_bar('Loading', current_index,
                                                            max_index if limit_to_first == -1 else limit_to_first)
            if prog_str != current_prog_str:
                current_prog_str = prog_str
                print(current_prog_str)

            img = Image.open(file_path)
            img = img.crop((25, 53, 201, 117))  # crop to correct size
            img_matrix = np.asarray(img)
            img_matrix = np.reshape(img_matrix, (1, 64, 176, 3))
            if matrix is None:
                matrix = img_matrix
            else:
                matrix = np.concatenate((matrix, img_matrix))
            current_index += 1

    return (matrix / 127.5) - 1  # transformation to range (-1, 1)


def train_dcgan(n_epochs, start_fresh=False):
    """

    """

    # definitions

    # get models

    g_model = eyeglass_generator.build_model()
    #g_model = eyeglass_generator.load_gen_weights(g_model)
    d_model = eyeglass_discriminator.build_model()
    # TODO only load mat weights if first training?
    # d_model = eyeglass_discriminator.load_discrim_weights(d_model)

    if not start_fresh:  # load parameters from previous training
        try:
            g_model.load_weights('../saved-models/gweights')
            d_model.load_weights('../saved-models/dweights')
            print('>>>>>>> DCGAN weights loaded.')
        except NotFoundError:
            print('>>>>>>> No weights found, using fresh initialized weights!')

    else:
        print('>>>>>>> Initialized DCGAN weights.')

    # define optimizer, same as described in paper
    gen_optimizer = tf.keras.optimizers.Adam(2e-4)
    discrim_optimizer = tf.keras.optimizers.Adam(2e-4)

    # custom training procedure function (see Tensorflow DCGAN tutorial)
    @tf.function
    def training_step(images, bs=BATCH_SIZE):
        """

        """
        noise = np.random.standard_normal((bs, 25))  # random input vector for generator

        with tf.GradientTape() as gen_tape, tf.GradientTape() as discrim_tape:
            generated_images = g_model(noise, training=True)  # generate fake images

            # get outputs for discriminator
            real_output = d_model(images, training=True)
            fake_output = d_model(generated_images, training=True)

            gen_loss = net_utils.get_gen_loss(fake_output)
            discrim_loss = net_utils.get_discrim_loss(real_output, fake_output)

        # compute and apply gradients
        gen_gradients = gen_tape.gradient(gen_loss, g_model.trainable_variables)
        discrim_gradients = discrim_tape.gradient(discrim_loss, d_model.trainable_variables)
        gen_optimizer.apply_gradients(zip(gen_gradients, g_model.trainable_variables))
        discrim_optimizer.apply_gradients(zip(discrim_gradients, d_model.trainable_variables))

        return gen_loss, discrim_loss

    # get data
    real_images = load_real_images(-1)
    print(np.shape(real_images))
    num_samples = real_images.shape[0]
    num_batches = math.ceil(num_samples / BATCH_SIZE)  # number of training data batches

    for epoch in range(n_epochs):
        print(f'Epoch {epoch + 1}:')
        batch_index = 0
        ep_g_loss_sum, ep_d_loss_sum = 0, 0

        for batch in net_utils.produce_training_batches(real_images, BATCH_SIZE):
            print(net_utils.display_custom_loading_bar('Training', batch_index, num_batches))
            g_loss, d_loss = training_step(batch)  # one training iteration for this batch

            # checkpoints both parts of the model
            g_model.save_weights('../saved-models/gweights')
            d_model.save_weights('../saved-models/dweights')

            # observe error on that batch
            ep_g_loss_sum += g_loss
            ep_d_loss_sum += d_loss
            batch_index += 1

        print(f'Avg. generator loss: {ep_g_loss_sum / num_batches}, '
              f'avg. discriminator loss: {ep_d_loss_sum / num_batches}')

    # TODO final stuff?
    random_z = np.random.standard_normal((1, 25))
    gen_output = np.reshape(g_model.predict(random_z), (64, 176, 3))
    gen_img = (gen_output + 1) * 127.5
    out_img = Image.fromarray(gen_img, mode='RGB')
    out_img.show()


if __name__ == '__main__':
    train_dcgan(42)
