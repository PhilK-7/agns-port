import os
import random

import tensorflow as tf

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from special_layers import LocalResponseNormalization, InceptionModule, InceptionModuleShrink, L2Normalization


# custom layer objects for OpenFace
custom_objects = {'LocalResponseNormalization': LocalResponseNormalization,
                  'InceptionModule': InceptionModule,
                  'InceptionModuleShrink': InceptionModuleShrink,
                  'L2Normalization': L2Normalization}

# LE BOILERPLATE SHIAT
# set parameters
USE_REMOTE = True  # set depending whether code is executed on remote workstation or not
if USE_REMOTE:
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    dap = os.path.expanduser('~') + '/storage-private/data/'
else:
    dap = '../data/'

if __name__ == '__main__':
    # which network type
    print('Choose your flavor: VGG or OpenFace?')
    input_1 = 0
    while input_1 not in (1, 2):
        try:
            net_index = input('Enter \'1\' for \'VGG\', enter \'2\' for \'OpenFace\'. \n')
            input_1 = int(net_index)
        except ValueError:
            continue
    if input_1 == 1:
        print('You chose \'VGG\'.')
    elif input_1 == 2:
        print('You chose \'OpenFace\'.')

    # how many classes
    print('Choose the classes count: normal or many?')
    input_2 = 0
    while input_2 not in (3, 4):
        try:
            count_index = input('Enter \'3\' for a normal amount of classes, or enter \'4\' for many. \n')
            input_2 = int(count_index)
        except ValueError:
            continue
    if input_2 == 3:
        print('The model has ten classes.')
    elif input_2 == 4:
        print('The model has 143 classes.')
    print('Loading model and dataset...')

    # load model
    model_path = '../saved-models/'
    if input_1 == 1:
        model_path += 'vgg_'
    if input_1 == 2:
        model_path += 'of'
    if input_2 == 3:
        model_path += '10'
    if input_2 == 4:
        model_path += '143'
    model_path += '.h5'
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    # load dataset
    ds_path = dap + 'pubfig/' + ('dataset_aligned' if input_2 == 4 else 'dataset_aligned_10')  # get correct path
    # get appropriate data generator
    if input_1 == 2:
        datagen = ImageDataGenerator(rescale=1. / 127.5, preprocessing_function=lambda t: t - 1)  # for OF
        datagen = datagen.flow_from_directory(ds_path, target_size=(96, 96))
    else:
        datagen = ImageDataGenerator(rescale=1. / 255)  # for VGG
        datagen = datagen.flow_from_directory(ds_path, (224, 224))
    class_dict = datagen.class_indices
    class_dict = {v: k for k, v in class_dict.items()}  # invert to have indices as keys

    # which target
    print('Which target to test the face recognition network on?')
    max_target_index = 9 if input_2 == 3 else 142
    print(
        f'If you want a specific target, provide an index between 0 and {max_target_index}. Otherwise enter anything.')
    given_index = input('Your target index: \n')
    target_index = -1
    try:  # check given index, only indices in range are accepted
        target_index = int(given_index)
        if target_index not in range(max_target_index):
            target_index = -1
    except ValueError:
        pass
    if target_index == -1:
        print('A random target is chosen: \n\n')
        target_index = random.randint(0, max_target_index)
    else:
        print('You chose as target: \n\n')
    target_name: str = class_dict[target_index]
    target_name = target_name.replace('_', ' ')
    print(target_name)

    # TODO: get data in tensor format, pass images, and show results incl. probs visually
    