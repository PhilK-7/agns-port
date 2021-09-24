from setup import setup_params
from attack.attacks import execute_attack


# subject n° 1 digital impersonation attack -vs- OpenFace 10


def main(gpus: tuple = (0,), set_correct_path=False):
    dap = setup_params(True, gpus)

    ep = 500  # maximum attack tries
    stop_prob = 0.924
    kappa = 0.25
    lr = 5e-5

    # set values
    target = 1  # target to impersonate: Barack Obama
    # impersonator chosen: Eva Mendes
    impersonator_path = 'pubfig/dataset_aligned_10/Eva_Mendes/aligned/'
    img_size = (96, 96)  # input size for OpenFace
    mask_path = 'eyeglasses/eyeglasses_mask_6percent.png'
    fc_path = '../saved-models/of10.h5'  # very weird: somehow in this file .. instead ../.. ?!, but right in main demo
    g_path = '../saved-models/gweights'
    d_path = '../saved-models/dweights'

    # very weird directory issue (see above) -> pick either .. or ../..
    if set_correct_path:
        fc_path = '../' + fc_path
        g_path, d_path = '../' + g_path, '../' + d_path

    # execute impersonation attack
    print('Trying to impersonate Eva Mendes against Barack Obama...')
    execute_attack(dap, impersonator_path, mask_path, img_size, g_path, d_path, fc_path, ep, lr, kappa,
                   stop_prob, 32,
                   target, False, False)


if __name__ == '__main__':
    main((2,))
