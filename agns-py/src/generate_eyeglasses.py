import eyeglass_generator as gen
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
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
    vectors = np.random.uniform(-1, 1, (16, 25))
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
