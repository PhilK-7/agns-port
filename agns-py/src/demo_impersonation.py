from setup import setup_params
from attacks import execute_attack


# subject n° 1 digital impersonation attack -vs- OpenFace 10


def main(gpus: tuple = (0,)):
    dap = setup_params(True, gpus)

    ep = 1  # maximum attack tries
    stop_prob = 0.924
    kappa = 0.25
    lr = 5e-5

    # set values
    target = 1  # target to impersonate: Barack Obama
    # impersonator chosen: Eva Mendes
    impersonator_path = 'pubfig/dataset_aligned_10/Eva_Mendes/aligned/'
    img_size = (96, 96)  # input size for OpenFace
    mask_path = 'eyeglasses/eyeglasses_mask_6percent.png'
    fn_path = '../saved-models/of10.h5'
    g_path = '../saved-models/gweights'
    d_path = '../saved-models/dweights'

    # execute impersonation attack
    print('Trying to impersonate Eva Mendes against Barack Obama...')
    execute_attack(dap, impersonator_path, mask_path, img_size, g_path, d_path, fn_path, False, ep, lr, kappa,
                   stop_prob, 32,
                   target, False, False)


if __name__ == '__main__':
    main((2,))
