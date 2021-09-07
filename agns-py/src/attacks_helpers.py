import os
import time

import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
from skimage import measure, filters

crop_coordinates = [53, 25, 53 + 64, 25 + 176]  # in Matlab Code
# similar as in 'data/auxiliary/eyeglass_marks_centers.mat' in Matlab Code
glasses_center_coordinates = np.array([[34, 69], [114, 67], [190, 67], [175, 103], [134, 100], [97, 98], [52, 105]],
                                      dtype='float32')


def pad_glasses_image(glass: tf.Tensor):
    """
    Pads generated glasses image(s) as reverse transformation to the cropping that was applied to the original data.
    Can also be used for batched tensors of multiple images.

    :param glass: the glasses image, represented as 4D tensor (range [-1., 1.])
    :return: a bigger tensor that represents 224x224 pixels, with black padding added
    """

    init_shape = glass.shape
    assert len(init_shape) in (3, 4)
    glass = glass + 1  # range [0., 2.]
    # widen dimensions if single image
    if len(init_shape) != 4:
        glass = tf.reshape(glass, [1, *init_shape])

    # NOTE: if available, better use a layer than writing values to a tf.Variable, because the latter is much slower
    pad_layer = tf.keras.layers.ZeroPadding2D(((crop_coordinates[0], 224 - crop_coordinates[2]),
                                               (crop_coordinates[1], 224 - crop_coordinates[3])))
    glass = pad_layer(glass)  # reverse transformation to crop: black padding
    glass = glass - 1  # back to [-1., 1.]

    # remove batch dimension if single image
    if len(init_shape) != 4:
        glass = tf.reshape(glass, glass.shape[1:])

    return glass


def load_glasses_mask(data_path: str, mask_path: str) -> tf.Tensor:
    """
    Loads the mask for glasses images.

    :param data_path: the path to the data directory
    :param mask_path: the relative path to the mask image (from 'data')
    :return: the mask as tensor, with float values in [0, 1]
    """
    mask_path = data_path + mask_path  # full path
    mask_img = tf.image.decode_png(tf.io.read_file(mask_path), channels=3)
    mask_img = tf.image.convert_image_dtype(mask_img, tf.float32)  # scales to [0., 1.]

    return mask_img


def merge_images_using_mask(data_path: str, img_a: tf.Tensor, img_b: tf.Tensor,
                            mask_path: str = '', mask: tf.Tensor = None):
    """
    Merges two images A and B, using a provided filter mask.

    :param data_path: the path to the data directory; can be empty if mask is given as tensor
    :param img_a: the first image (face), that a part of the other image should be overlayed on (range [-1., 1.]);
        assumed to be of shape (224, 224)
    :param img_b: the second image (glasses), that should (in part) be overlayed on the other one (range [-1., 1.]);
        the same shape as img_a
    :param mask_path: the relative path (from data) to a filter mask that determines which pixels of image B
        should be put onto image A - the mask has only black and white pixels that are interpreted in a boolean manner;
        can be left empty if mask is already supplied as tensor
    :param mask: alternatively, the mask is already loaded and can be supplied as tensor here
        (use this when needing the same mask repeatedly for efficiency)
    :return: a tensor-like object where the pixels of image B are put over those in image A
        as specified by the mask (range [0., 1.]), of image shape 224x224
    """

    # load mask and convert it to boolean mask tensor
    if mask is None:
        mask_img = load_glasses_mask(data_path, mask_path)
    else:
        mask_img = mask

    # process face image
    fimg = img_a + 1  # to range to [0., 2.]
    inv_mask_img = -mask_img + 1  # inverted mask
    fimg = tf.math.multiply(fimg, inv_mask_img)  # remove pixels in glass area

    # process glasses image
    gimg = img_b + 1  # to range [0., 2.]
    gimg = tf.math.multiply(gimg, mask_img)  # remove pixels outside of glass area
    gimg = gimg - 1  # back to range [-1., 1.]

    '''timg = (gimg + 1) * 127.5
    timg = timg.numpy()
    timg = timg.astype(np.uint8)
    save_img_from_tensor(timg, 'merge')'''

    # merge images
    merged = gimg + fimg  # merge images, output scale [-1., 1.]
    merged = (merged + 1) / 2  # scale [0., 1.]

    '''timg = merged * 255
    timg = timg.numpy()
    timg = timg.astype(np.uint8)
    save_img_from_tensor(timg, 'merged')'''

    return merged


def scale_integer_to_zero_one_tensor(tensor: tf.Tensor) -> tf.Tensor:
    """
    Receives a tensor of (unsigned) integer values in [0, 255], and scales the values to [0., 1.].

    :param tensor: a tf.Tensor with int values in range between 0 and 255
    :return: a tensor of the same size as the input, with tf.float32 values between 0 and 1
    """
    t = tf.cast(tensor, tf.float32)
    t = t / 255.

    return t


def scale_zero_one_to_integer_tensor(tensor: tf.Tensor) -> tf.Tensor:
    """
    Receives a tensor of float values in [0., 1.] and scales them to [0, 255], converting to uint8.

    :param tensor: a tf.Tensor with float values between 0 and 1
    :return: a tensor of the same size as the input, with tf.uint8 values between 0 and 255
    """
    t = tf.cast(tensor, tf.uint8)
    t = t * 255

    return t


def convert_to_numpy_slice(imgs, i: int) -> np.ndarray:
    """
    Receives images represented by a tensor (value range [-1, 1]),
     and converts one image specified by index to a ndarray with range [0, 255].

    :param imgs: the images as a tensor-like object
    :param i: the index of the image in the input tensor
    :return: a NumPy array with scaled integer values that represents a slice of the input tensor
    """
    imgs: np.ndarray = imgs.numpy()
    img = imgs[i]
    img = (img + 1) * 127.5
    img = img.astype(np.uint8)

    return img


def save_img_from_tensor(img, name: str, use_time: bool = True):
    """
    Saves an image given by a NumPy array to a file in the out directory. Might be useful for debugging purposes.

    :param img: the image represented as ndarray
    :param name: a name (prefix) to save the image in the 'out' directory
    :param use_time: whether to append a timestamp to the output file name
    """

    img = Image.fromarray(img)  # numpy array -> pillow image
    if not os.path.exists('../out'):  # setup 'out' folder if missing
        os.mkdir('../out')
    filename = '../out/' + name + '_' + (str(time.time() if use_time else '')) + '.png'  # compose name
    img.save(filename)  # save to file


def add_merger_to_generator(generator, data_path, target_path, n_inputs, output_size=(224, 224),
                            scale_to_zero_base: bool = False, new_version: bool = True, physical: bool = False):
    """
    Receives a generator model and adds a GlassesFacesMerger on top of it, given the parameters.

    :param generator: the generator model object (likely a tf.keras.models.Sequential)
    :param data_path: path to the 'data' directory
    :param target_path: relative path of the target´s dataset (from 'data')
    :param n_inputs: the fixed number of inputs, predetermines what the entire models input batch size must be
    :param output_size: the desired image output size
    :param scale_to_zero_base: whether the resizer layer should scale the values to [0., 1.] range
    :param new_version: use combination of newer multiple layers instead of GlassesFacesMerge
    :param physical: whether to use the FaceAdder to perform a physical attack
    :return: a new generator model that has merged faces with glasses as output images
    """
    model = tf.keras.models.Sequential([generator], name='Gen_Merge')  # copy rest layers
    from special_layers import GlassesFacesMerger, BlackPadding, Resizer, FaceAdder
    if not new_version:
        model.add(GlassesFacesMerger(data_path, target_path, n_inputs, output_size))  # add merging layer
    else:
        model.add(BlackPadding(data_path))
        model.add(FaceAdder(data_path, target_path, physical))
        model.add(Resizer(output_size, scale_to_zero_base))

    return model


def strip_softmax_from_face_recognition_model(facenet):
    """
    Removes the last layer (softmax classification output) from a face recognition model.

    :param facenet: the face recognition model
    :return: a copy of the model, but without softmax at the end
    """

    # cut off only last layer: softmax (Simplex)
    model = tf.keras.models.Sequential([*facenet.layers[0].layers, *facenet.layers[1].layers[:-1]],
                                       name='Facenet_No_Softmax')

    return model


def find_green_marks(img: tf.Tensor):
    """
    Computes the center points of the green marks in a given image with worn model eyeglasses.
    NOTE: This function is optimized for the provided images in 'demo-data2'.

    :param img: the image given as a tensor of integers in range [0, 255], shape (224, 224, 3), type uint8
    :return: a list of duples, which hold the x- and y-coordinate per center point respectively
    """

    # use thresholds to keep only green mark areas
    r_t, g_t, b_t = 150, 150, 145
    red_binary = tf.math.logical_not(tf.math.less(img[:, :, 0], r_t))
    green_binary = tf.math.logical_not(tf.math.greater(img[:, :, 1], g_t))
    blue_binary = tf.math.less(img[:, :, 2], b_t)
    binary = tf.math.logical_and(tf.math.logical_and(red_binary, green_binary), blue_binary)  # combine thresholding
    binary_filter = tf.cast(binary, tf.uint8)  # needed numerical
    binary_filter = tf.stack([binary_filter for _ in range(3)], axis=2)  # broadcast for multiplication
    filtered_image = tf.math.multiply(img, binary_filter)  # keep only green marks

    # find connected components
    img = filtered_image.numpy()
    img = filters.gaussian(img, 1.0)  # smooth for better connectivity
    img_binary = img > filters.threshold_mean(img)
    labels, num = measure.label(img_binary, return_num=True)
    if num != 7:
        print('Warning: An incorrect number of blobs has been recognized.')

    # compute blob centers
    blob_coordinates_values = [np.where(labels == [i, i, i]) for i in range(1, 8)]  # List[Tuple[ndarray x3]]
    blob_coordinates_values = [(blob_coordinates_values[i][0], blob_coordinates_values[i][1]) for i in range(7)]
    blob_centres = [(np.mean(blob_coordinates_values[i][0]), np.mean(blob_coordinates_values[i][1])) for i in range(7)]
    blob_centres = [(int(round(blob_centres[i][0])), int(round(blob_centres[i][1]))) for i in range(7)]
    # NOTE: y and x coordinates are swapped (also the case when plotted etc.), as also done in Matlab code

    # order blob centers (are already grouped in two sets)
    blob_centres = [*sorted(blob_centres[0:3], key=lambda e: e[1]),
                    *sorted(blob_centres[3:], key=lambda e: e[1], reverse=True)]

    # visualize found points
    labels = labels * 32  # make visible
    for bc in blob_centres:
        labels[bc[0], bc[1]] = [255, 0, 0]  # mark centers
    plt.imshow(labels)
    plt.show()

    # flip y and x to (x, y) as needed further
    blob_centres = [(b[1], b[0]) for b in blob_centres]
    blob_centres = np.array([[bc[0], bc[1]] for bc in blob_centres], dtype='float32')

    return blob_centres
