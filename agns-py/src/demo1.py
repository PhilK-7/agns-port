from tensorflow.keras.models import load_model

import eyeglass_discriminator
import eyeglass_generator
import dcgan
import attacks
import tensorflow as tf

# subject n° 19 digital dodging attack -vs- VGG143
if __name__ == '__main__':
    ep = 100
    lr = 5e-5
    weight_decay = 1e-5
    kappa = 0.8
    stop_prob = 0.02
    bs = 32

    # load models and set more values
    face_model = load_model('../saved-models/VGG143.h5')  # needs to be fooled
    gen_model = eyeglass_generator.build_model()
    gen_model.load_weights('../saved-models/gweights')
    dis_model = eyeglass_discriminator.build_model()
    dis_model.load_weights('../saved-models/dweights')
    target = 19
    img_path = 'pubfig/dataset_aligned/Danny_Devito/aligned'  # relative to 'data'
    img_size = (224, 224)  # input size for VGG

    # get glasses dataset to draw two half-batches from each training epoch
    glasses_ds = dcgan.load_real_images()

    # perform special training
    current_ep = 0
    g_opt, d_opt = tf.keras.optimizers.Adam(learning_rate=lr), tf.keras.optimizers.Adam(learning_rate=lr)
    while current_ep <= ep:
        glasses_a = glasses_ds.take(bs / 2)
        glasses_b = glasses_ds.take(bs / 2)
        g_opt, d_opt, obj_d, obj_f = attacks.do_attack_training_step(gen_model, dis_model, face_model, img_path, target,
                                                                     gl, 0, g_opt, d_opt, bs, kappa)
        if attacks.check_objective_met(gen_model, face_model, target, img_path, stop_prob, bs, True):
            print('<<<<<< Dodging attack successful! >>>>>>')
            break

        current_ep += 1
