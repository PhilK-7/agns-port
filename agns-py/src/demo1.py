import os

from tensorflow.keras.models import load_model

import eyeglass_discriminator
import eyeglass_generator
from attacks_helpers import add_merger_to_generator, strip_softmax_from_face_recognition_model
import dcgan
import attacks
import tensorflow as tf

# subject n° 19 digital dodging attack -vs- VGG143
if __name__ == '__main__':
    # LE BOILERPLATE SHIAT
    # set parameters
    USE_REMOTE = True  # set depending whether code is executed on remote workstation or not
    if USE_REMOTE:
        os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = '3'
        dap = os.path.expanduser('~') + '/storage-private/data/'
    else:
        dap = '../data/'

    ep = 100
    lr = 5e-5
    weight_decay = 1e-5  # TODO also?
    kappa = 0.25
    stop_prob = 0.01
    bs = 32

    # load models and set more values
    print('Loading models...')
    face_model = load_model('../saved-models/vgg_143.h5')  # needs to be fooled
    gen_model = eyeglass_generator.build_model()
    gen_model.load_weights('../saved-models/gweights')
    dis_model = eyeglass_discriminator.build_model()
    dis_model.load_weights('../saved-models/dweights')
    print('All models loaded.')
    target = 19
    img_path = 'pubfig/dataset_aligned/Danny_Devito/aligned/'  # relative to 'data'
    img_size = (224, 224)  # input size for VGG
    mask_path = 'eyeglasses/eyeglasses_mask_6percent.png'

    # get special versions of gen/fn
    gen_model_ext = add_merger_to_generator(gen_model, dap, img_path, bs // 2, img_size, True)
    face_model = strip_softmax_from_face_recognition_model(face_model, 143)

    # get glasses dataset to draw two half-batches from each training epoch (already shuffled and batched)
    print('Loading glasses dataset...')
    glasses_ds = dcgan.load_real_images(dap)
    print('Glasses dataset ready.')

    # perform special training
    print('Perform special training:')
    current_ep = 1
    g_opt, d_opt = tf.keras.optimizers.Adam(learning_rate=lr), tf.keras.optimizers.Adam(learning_rate=lr)

    while current_ep <= ep:
        print(f'Attack training epoch {current_ep}.')

        # get real glasses as half-batches
        glasses = glasses_ds.take(1)  # take one batch
        glasses = [p for p in glasses]  # unravel
        glasses = tf.stack(glasses)
        glasses = tf.reshape(glasses, glasses.shape[1:])
        glasses_a, glasses_b = glasses[:bs // 2], glasses[bs // 2:]
        if current_ep == 1:
            print(glasses_a.shape)

        g_opt, d_opt, obj_d, obj_f = attacks.do_attack_training_step(gen_model, dis_model,
                                                                     gen_model_ext,
                                                                     face_model, target, glasses_a, glasses_b,
                                                                     g_opt, d_opt, bs, kappa)
        # TODO what to do with obj values?
        print(obj_d, obj_f)
        if attacks.check_objective_met(dap, gen_model, face_model, target, img_path, mask_path, stop_prob, bs, img_size,
                                       dodge=True):
            print('<<<<<< Dodging attack successful! >>>>>>')
            break

        current_ep += 1
