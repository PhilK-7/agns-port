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

    # get glasses dataset
    glasses_ds = dcgan.load_real_images()

    # perform special training
    current_ep = 0
    done = False
    g_opt, d_opt = tf.keras.optimizers.Adam(learning_rate=lr), tf.keras.optimizers.Adam(learning_rate=lr)
    while current_ep <= ep and not done:
        pass  # todo
        # TODO stopping criterion

        current_ep += 1
