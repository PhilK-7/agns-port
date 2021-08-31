from setup import setup_params
from attacks import execute_attack

# subject n° 1 digital impersonation attack -vs- OpenFace 10
if __name__ == '__main__':
    dap = setup_params(True)

    ep = 100  # TODO 1 ?!
    stop_prob = 0.924
    kappa = 0.25
    lr = 5e-5

    # set values
    target = 1
    target_path = 'pubfig/dataset_aligned_10/Barack_Obama/aligned/'
    img_size = (96, 96)  # input size for OpenFace
    mask_path = 'eyeglasses/eyeglasses_mask_6percent.png'
    fn_path = '../saved-models/of10.h5'
    g_path = '../saved-models/gweights'
    d_path = '../saved-models/dweights'

    # execute impersonation attack
    execute_attack(dap, target_path, mask_path, img_size, g_path, d_path, fn_path, False, ep, lr, kappa, stop_prob, 32,
                   target, False, False)
    # TODO fix bug causing objective checker to fail after epoch 2
    # TODO check again: target =/= impersonator !