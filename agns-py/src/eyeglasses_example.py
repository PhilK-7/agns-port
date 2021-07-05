import eyeglass_generator as gen
import numpy as np


if __name__ == '__main__':
    print('Loading model...')
    model = gen.build_model()
    model.load_weights('../saved-models/gweights')
    model.build()
    print('Generator loaded.')
    pseudorand = ''
    while not isinstance(pseudorand, int):
        pseudorand = input('Please enter a number as pseudo randomness:  ')
        try:
            pseudorand = int(pseudorand)
        except:
            print('Please enter a valid number.')

    print('Using pseudo randomness to generate a random vector.')
    vector = np.random.uniform(-1, 1, (16, 25))
    vector = np.reshape(vector, (16, 25))
    preds = model.predict(vector)
    preds = np.reshape(preds, (16, 64, 176, 3))
    preds = gen.scale_gen_output(preds)
    print(np.shape(preds))
    gen.save_gen_output_to_file(preds[0])