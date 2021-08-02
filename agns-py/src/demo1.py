from tensorflow.keras.models import load_model

import eyeglass_discriminator
import eyeglass_generator
import dcgan

# subject n° 19 digital dodging attack -vs- VGG143
if __name__ == '__main__':
    ep = 1
    lr = 5e-5
    weight_decay = 1e-5
    kappa = 0.8
    stop_prob = 0.02

    # TODO load model and attacker images
    face_model = load_model('../saved-models/VGG143.h5')  # needs to be fooled
    gen_model = eyeglass_generator.build_model()
    gen_model.load_weights('../saved-models/gweights')
    dis_model = eyeglass_discriminator.build_model()
    dis_model.load_weights('../saved-models/dweights')
    target = 19
    img_path = '../data/pubfig/dataset_/Danny_Devito'
    img_size = (224, 224)  # input size for VGG

    # TODO execute attack and show results

    # TODO set generator trainable, discriminator not trainable
    # TODO train DCGAIN further and stop when objective met
