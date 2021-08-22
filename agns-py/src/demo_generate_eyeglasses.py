import os

import eyeglass_generator as gen
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # LE BOILERPLATE SHIAT
    # set parameters
    USE_REMOTE = True  # set depending whether code is executed on remote workstation or not
    if USE_REMOTE:
        os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = '2'
        dap = os.path.expanduser('~') + '/storage-private/data/'
    else:
        dap = '../data/'

    # load generator
    print('Loading model...')
    model = gen.build_model()
    model.load_weights('../saved-models/gweights')
    model.build()
    print('Generator loaded.')

    # get randomness
    pseudorand = ''
    while not isinstance(pseudorand, int):
        pseudorand = input('Please enter a number as pseudo randomness:  ')
        try:
            pseudorand = int(pseudorand)
        except:
            print('\n Please enter a valid number.')

    # generate random input vectors
    print('Using pseudo randomness to generate a random vector.')
    vectors = np.random.standard_normal((16, 25))
    vectors = np.reshape(vectors, (16, 25))

    # generate glasses
    preds = model.predict(vectors)
    preds = np.reshape(preds, (16, 64, 176, 3))
    preds = gen.scale_gen_output(preds)
    print(np.shape(preds))
    glasses = [preds[i] for i in range(np.shape(preds)[0])]

    # plot glasses
    plt.figure(figsize=(4, 4))
    for i, gl in enumerate(glasses):
        plt.subplot(4, 4, i + 1)
        plt.imshow(gl)
        plt.axis('off')
    plt.show()
    gen.save_gen_output_to_file(glasses[np.random.randint(0, 15)])  # save one example to 'out'
