import attacks
from setup import setup_params

# subject n° 19 digital dodging attack -vs- VGG143
if __name__ == '__main__':
    dap = setup_params(True, 4)

    # some hyperparameters
    ep = 100  # TODO 1 ?!
    lr = 5e-5
    kappa = 0.25
    stop_prob = 0.01
    bs = 32

    # set paths and more values
    target = 19
    target_path = 'pubfig/dataset_aligned/Danny_Devito/aligned/'  # relative to 'data'
    img_size = (224, 224)  # input size for VGG
    mask_path = 'eyeglasses/eyeglasses_mask_6percent.png'
    fn_path = '../saved-models/vgg_143.h5'
    g_path = '../saved-models/gweights'
    d_path = '../saved-models/dweights'

    # execute dodging attack
    attacks.execute_attack(dap, target_path, mask_path, img_size, g_path, d_path, fn_path, True, ep, lr, kappa,
                           stop_prob, bs, target, True, True)
