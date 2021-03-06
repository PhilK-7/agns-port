from attack.attacks import execute_attack
from setup import setup_params


# subject n° 6 physical impersonation attack -vs- VGG 10

def main(gpus: tuple = (0,)):
    dap = setup_params(True, gpus)

    ep = 1
    stop_prob = 0.999
    kappa = 0.25
    lr = 5e-5

    # set values
    target = 6
    img_size = (224, 224)
    impersonator_path = 'demo-data2/'
    fc_path = '../../saved-models/vgg_10.h5'
    g_path = '../../saved-models/gweights'
    d_path = '../../saved-models/dweights'
    mask_path = 'eyeglasses/eyeglasses_mask_6percent.png'

    # execute physical impersonation attack
    print('Trying to impersonate Mahmood Sharif against George Clooney...')
    execute_attack(dap, impersonator_path, mask_path, img_size, g_path, d_path, fc_path, ep, lr, kappa,
                   stop_prob, 32,
                   target, True, False, True)


if __name__ == '__main__':
    main((3,))

