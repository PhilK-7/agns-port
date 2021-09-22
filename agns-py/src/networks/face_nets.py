import os
import random
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras.applications.vgg16 as vgg
import tensorflow_addons as tfa
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from setup import setup_params
from special_layers import LocalResponseNormalization, L2Normalization, InceptionModule, \
    InceptionModuleShrink

# for usage from command line
# sys.path.append(path.dirname(path.dirname(path.abspath('face_nets.py'))))

# custom layer objects
custom_objects = {'LocalResponseNormalization': LocalResponseNormalization,
                  'InceptionModule': InceptionModule,
                  'InceptionModuleShrink': InceptionModuleShrink,
                  'L2Normalization': L2Normalization}

random.seed(42)  # reproducible: same train / val splits


def write_class_mapping(imgen_dict):
    """
    Writes the dictionary for the classes used in the face recognition models, if it doesn´t exist yet.

    :param imgen_dict: the dictionary returned from a tf.keras ImageDataGenerator
    """

    # determine path
    save_path = data_path + 'pubfig'
    if len(imgen_dict) == 10:
        save_path += '/class-mapping_10.txt'
    else:
        save_path += '/class-mapping_143.txt'

    if os.path.exists(save_path):
        return  # cancel if already created

    # create text from given dictionary
    textstr = '=== Class Mapping === \n\n'
    for mapping in imgen_dict.items():
        textstr += mapping[0]
        textstr += ' : '
        textstr += str(mapping[1])
        textstr += '\n'

    # make file
    with open(save_path, 'w') as file:
        file.write(textstr)


def make_model_plot(name: str, model):
    """
    Draws a model graph plot.
    NOTE: Requires Graphviz to be installed.

    :param name: the name of the model
    """
    if not os.path.exists('../../saved-plots'):
        os.mkdir('../../saved-plots')
    tf.keras.utils.plot_model(model, '../../saved-plots/' + name + '.png',
                              expand_nested=True, show_shapes=True)


def get_class_weights(gen_classes):
    """
    Computes class weights for a (training) generator in order to balanced out class differences during training.

    :param gen_classes: a DirectoryIterator´s classes object
    :return: a list of computed class weights
    """
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(gen_classes), y=gen_classes)
    weights = {i: weights[i] for i in range(len(weights))}
    print(f'Computed class weights {weights}')

    return weights


def get_original_vgg_model():
    """
    Loads the original VGG model (VGG16, type D), and cuts of the final layer (1000-class output).
    The weights are loaded from a pretrained model, trained on Imagenet data.
    The model's output is a 4096-d face descriptor.
    """
    base_model = vgg.VGG16(include_top=True)
    model = tf.keras.models.Model(base_model.input, base_model.layers[-2].output, name='VGG_Descriptor')
    # model.summary()
    make_model_plot('VGG16', model)

    return model


def build_openface_model():
    """
    Builds the nn4.small2 OpenFace model, with about 3.74M trainable parameters (3733968)
    (exact model type was inferred from paper's parameter count).
    The model's output is a 128-sphere, a face embedding vector.
    This model is the basis for the paper´s OF 143/10 models.

    :return the built OpenFace NN4.small2.v1 model tf.keras.Model object
    """

    # input part
    inp = tf.keras.layers.Input((96, 96, 3))  # input is (aligned) RBG image pf 96x96
    x = tf.keras.layers.Conv2D(64, 7, 2, padding='same', name='First_Conv2D')(inp)  # 48x48 x 64
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPool2D(3, 2, padding='same')(x)  # 24x24  x 64
    x = LocalResponseNormalization()(x)

    # Inception 2 (output size 24x24 x 192)
    x = tf.keras.layers.Conv2D(64, 1, 1, 'same', name='Inception_2_Conv2D')(x)  # 24x24 x 64
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(192, 3, padding='same')(x)  # 24x24 x 192
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.ReLU()(x)

    x = LocalResponseNormalization()(x)
    x = tf.keras.layers.MaxPool2D(3, 2, padding='same')(x)  # 12x12 x 192

    # Inception 3a (output size 12x12 x 256)
    x = InceptionModule([128, 32], [96, 16, 32, 64], 'Inception_3a')(x)

    # Inception 3b (output size 12x12 x 320)
    x = InceptionModule([128, 64], [96, 32, 64, 64], 'Inception_3b', True)(x)

    # Inception 3c (output size 6x6 x 640)
    x = InceptionModuleShrink([256, 64], [128, 32], 'Inception_3c')(x)

    # Inception 4a (output size 6x6 x 640)
    x = InceptionModule([192, 64], [96, 32, 128, 256], 'Inception_4a', True)(x)

    # Inception 4e (output size 3x3 x 1024)
    x = InceptionModuleShrink([256, 128], [160, 64], 'Inception_4e')(x)

    # Inception 5a (output size 3x3 x 736)
    x = InceptionModule([384], [96, 96, 256], 'Inception_5a', True)(x)

    # Inception 5b (output size 3x3 x 736)
    x = InceptionModule([384], [96, 96, 256], 'Inception_5b')(x)

    # final layers
    x = tf.keras.layers.AvgPool2D((3, 3))(x)  # 1x1 x 736
    x = tf.keras.layers.Flatten(name='reshape')(x)  # 736
    x = tf.keras.layers.Dense(128)(x)
    x = L2Normalization()(x)

    # assemble model
    model = tf.keras.Model(inputs=[inp], outputs=[x], name='Openface_NN4.Small2.v1')
    # model.summary()
    make_model_plot('OpenFace', model)

    return model


def build_vgg_custom_part(bigger_class_n=False):
    """
    Builds the paper's additional model part for a VGG-based face DNN.
    The inputs are 4096-d face descriptors, gained from the standard VGG model.

    :param bigger_class_n: whether to use 143 instead of 10 classes
    :return the additional part of VGG 143/10 as tf.keras.Sequential object
    """
    inp = tf.keras.layers.InputLayer((4096,), name='Descriptor_Input')
    dense = tf.keras.layers.Dense(143 if bigger_class_n else 10, name='Logits')
    simplex = tf.keras.layers.Softmax(name='Simplex')
    mname = 'VGG143_head' if bigger_class_n else 'VGG10_head'

    model = tf.keras.Sequential([
        inp,
        dense,
        simplex
    ],
        name=mname)
    # model.summary()
    make_model_plot(mname, model)

    return model


def build_of_custom_part(bigger_class_n=False):
    """
    Builds the paper's additional model part for an OpenFace-based face DNN.
    The inputs are 128-spheres, gained from the OpenFace model.

    :param bigger_class_n: whether to use 143 instead of 10 classes
    :return the additional part of OF 143/10 as tf.keras.Sequential object
    """
    inp = tf.keras.layers.InputLayer((128,), name='Sphere_Input')
    dense_1 = tf.keras.layers.Dense(286 if bigger_class_n else 12, name='Fully_Connected', activation='tanh')
    dense_2 = tf.keras.layers.Dense(143 if bigger_class_n else 10, name='Logits')
    simplex = tf.keras.layers.Softmax(name='Simplex')
    mname = 'OF143_head' if bigger_class_n else 'OF10_head'

    model = tf.keras.Sequential([
        inp,
        dense_1,
        dense_2,
        simplex
    ],
        name=mname)
    # model.summary()
    make_model_plot(mname, model)

    return model


def train_vgg_dnn(epochs: int = 1, lr: float = 5e-3, bigger_class_n=True):
    """
    Trains the complete custom VGG 143/10 model on the given dataset.
    Either starts training / fine-tuning from scratch, or continues with a found saved model state.

    :param epochs: how many training epochs long to train for this function call
    :param lr: the learning rate
    :param bigger_class_n: whether to train the VGG 143 model, instead of the VGG 10 model
        (also deciding which subset of the PubFig data is used)
    """

    # compose complete model
    vgg_base = get_original_vgg_model()
    for layer in vgg_base.layers:  # freeze VGG base layers (transfer learning)
        layer.trainable = False
    top_part = build_vgg_custom_part(bigger_class_n)
    class_suffix = '_143' if bigger_class_n else '_10'
    save_path = '../../saved-models/vgg' + class_suffix + '.h5'

    # get saved weights, or start with new transfer learning (fine-tune top layers)
    try:
        model = tf.keras.models.load_model(save_path)
        print('\n Model state loaded. Continue training...')
    except OSError:
        model = tf.keras.Sequential([vgg_base, top_part], name='VGG' + class_suffix + '_complete')
        print('\n No saved weights found. Start training new model...')

    model.summary()
    make_model_plot('vgg_143_full' if bigger_class_n else 'vgg_10_full', model)

    # load dataset, rescale + resize images
    ds_path = data_path + 'pubfig/dataset_aligned'
    if not bigger_class_n:
        ds_path += '_10/'
    else:
        ds_path += '/'

    # get part of PubFig dataset, separated by classes; also scale pixel values and image size
    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)  # scale [0, 1]
    train_gen = datagen.flow_from_directory(ds_path, (224, 224), subset='training')
    val_gen = datagen.flow_from_directory(ds_path, (224, 224), subset='validation')
    '''
    WARNING: If there are mysterious duplicate files starting with '._' in the subclass directories, the program
    will crash.
    To remove the files, go to 'dataset_' and execute this command:
    find . -type f -name ._\* -exec rm {} \;
    '''

    # do training
    opt = tf.keras.optimizers.Adam(learning_rate=lr)  # can be adjusted, use smaller rate the more progressed
    model.compile(opt, 'categorical_crossentropy', ['accuracy'])
    class_weights = get_class_weights(train_gen.classes)
    losses = model.fit(train_gen, epochs=epochs, validation_data=val_gen, class_weight=class_weights).history

    # save model state
    model.save(save_path)
    print('\n VGG model saved.')


# go to agns-port and execute align_all.sh with pubfig/dataset_ in data to get dataset_aligned
# do the same for dataset_10 instead to get only the 10 classes, when training OF10 model


def pretrain_openface_model(epochs=10, bs=160, lr=5e-3):
    """
    NOTE: Pretraining not necessary for training OF models. Also, NaN losses can occur during training.
    Trains the OpenFace NN4.small2.v1 model, as preparation for the custom OF 143/10 models.
    Uses aligned images from the PubFig dataset, from all 143 classes.

    :param epochs: the amount of epochs to train the model this function call
    :param bs: the training batch size
    :param lr: the learning rate
    """

    # get trained model
    model_path = '../../saved-models/openface.h5'
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print('\n Model save state loaded. Continue training:')
    except OSError:
        model = build_openface_model()
        print('\n No model save state found. Start training:')

    # load aligned face images and transform
    ds_path = data_path + 'pubfig/dataset_aligned'
    datagen = ImageDataGenerator(rescale=1. / 127.5, preprocessing_function=lambda t: t - 1)
    # use extra large batches for training (see triplets problem)
    datagen = datagen.flow_from_directory(ds_path, target_size=(96, 96), class_mode='sparse', batch_size=bs)

    # train model
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    pretain_loss = tfa.losses.TripletSemiHardLoss()
    model.compile(opt, pretain_loss)
    model.fit(datagen, epochs=epochs)

    # save after (continued) training
    model.save(model_path)
    print('\n Model state saved.')


def train_of_dnn(epochs: int = 1, lr: float = 5e-3, bigger_class_n=True, require_pretrained=False):
    """
    Trains the custom OF 143/10 model on the given dataset, based on the OpenFace model.
    Either starts training from scratch, or continues with a found saved model state.

    :param epochs: how many training epochs long to train for this function call
    :param lr: the learning rate
    :param bigger_class_n: whether to train the OF 143 model, instead of the OF 10 model
        (also deciding which subset of the PubFig data is used)
    :param require_pretrained: whether a pretrained copy 'openface.h5' is the base for training a new OF model copy
    """

    # setup model
    save_path = '../../saved-models/of' + ('143' if bigger_class_n else '10') + '.h5'
    try:  # continue training
        model = tf.keras.models.load_model(save_path, custom_objects=custom_objects)
        print('\n Saved model state found. Continue training:')
    except (ImportError, IOError):
        print('\n No saved state for the complete OF' + ('143' if bigger_class_n else '10') + ' model found.')
        try:
            base_model = tf.keras.models.load_model('../../saved-models/openface.h5', custom_objects=custom_objects)
            top_model = build_of_custom_part(bigger_class_n)
            print('\n Using pretrained OpenFace base model to train new OpenFace complete model.')
            model = tf.keras.Sequential([base_model, top_model])
        except (ImportError, IOError):
            if require_pretrained:
                print('\n Also no pretrained OpenFace base model found. Pretrain OpenFace first.')
                return
            else:
                print('\n Begin training new OpenFace model, without trained base OpenFace model provided.')
                of_model = build_openface_model()
                top_model = build_of_custom_part(bigger_class_n)
                model = tf.keras.Sequential([of_model, top_model])

    model.summary()
    make_model_plot('of143_full' if bigger_class_n else 'of10_full', model)

    # get data
    ds_path = data_path + 'pubfig/dataset_aligned'
    if not bigger_class_n:
        ds_path += '_10'

    # load aligned face images and transform
    datagen = ImageDataGenerator(rescale=1. / 127.5, preprocessing_function=lambda t: t - 1, validation_split=0.2)
    train_gen = datagen.flow_from_directory(ds_path, target_size=(96, 96), subset='training')
    val_gen = datagen.flow_from_directory(ds_path, (96, 96), subset='validation')
    write_class_mapping(train_gen.class_indices)

    # train model
    opt = tf.keras.optimizers.Adam(learning_rate=lr)  # adjust according to training progress
    model.compile(opt, 'categorical_crossentropy', ['accuracy'])
    class_weights = get_class_weights(train_gen.classes)
    model.fit(train_gen, epochs=epochs, validation_data=val_gen, class_weight=class_weights)

    # save model after training
    model.save(save_path)
    print('\n OpenFace model saved.')


def build_detector_model():
    """

    """

    model = tf.keras.Sequential()
    model.add(tf.keras.Input((14, 14, 512)))
    model.add(tf.keras.layers.Conv2D(196, (3, 3), padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPool2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(196, (3, 3), padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv2D(196, (3, 3), padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Flatten())  # 196
    model.add(tf.keras.layers.Dense(2, activation='softmax'))  # binary simplex

    model.summary()

    return model


if __name__ == '__main__':
    data_path = setup_params(True, (2,))

    if len(sys.argv) < 2:
        ep = 1
    else:
        ep = int(sys.argv[1])

    # VGG is good, don´t continue training
    # training calls here
    for lr in [5e-5]:
        train_of_dnn(20, lr, True)

