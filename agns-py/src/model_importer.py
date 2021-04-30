from scipy import io

'''
The purpose of this module is to import saved model weights from .mat files,
and to transform them so that Tensorflow / Keras can load those model weights.
'''


def load_mat_model_weights(mat_file_path):
    """
    Loads model weights from a .mat file that contains a dictionary.
    :param mat_file_path: a path to a .mat file
    :return: a list containing the weight matrices for every layer of a model, from start to end
    """

    mat_file = io.loadmat(mat_file_path)  # load .mat file containing data from path
    matrix = mat_file['matrices'][0]  # data in numpy array

    return [mat for mat in matrix]  # outer numpy array as list


# TODO Note unsure about (1, n) matrices?

'''
gen.mat weight parameters:
    (25, 7040) -> fully connected layer weights
    (1, 7040) -> fully connected layer bias
    (1, 7040) -> batch normalization layer 1
    (160, 80, 5, 5) -> deconvolutional layer 1 weights
    (1, 80) -> deconvolutional layer 1 bias
    (1, 80) -> batch normalization layer 2
    (80, 40, 5, 5) -> deconvolutional layer 2 weights
    (1, 40) -> deconvolutional layer 2 bias
    (1, 40) -> batch normalization layer 3
    (40, 20, 5, 5) -> deconvolutional layer 3 weights
    (1, 20) -> deconvolutional layer 3 bias
    (1, 20) -> batch normalization layer 4
    (20, 3, 5, 5) -> deconvolutional layer 4 weights

discrim.mat weight parameters:
    (20, 3, 5, 5) -> convolutional layer 1 weights
    (40, 20, 5, 5) -> convolutional layer 2 weights
    (1, 40) -> convolutional layer 2 bias
    (1, 40) -> batch normalization layer 1
    (80, 40, 5, 5) -> convolutional layer 3 weights
    (1, 80) -> convolutional layer 3 bias
    (1, 80) -> batch normalization layer 2
    (160, 80, 5, 5) -> convolutional layer 4 weights
    (1, 160) -> convolutional layer 4 bias
    (1, 160) -> batch normalization layer 3
    (7040, 1) -> fully connected
    
'''


if __name__ == '__main__':
    data = load_mat_model_weights('../discrim.mat')
    print(data)
    print(len(data))
    for a in data:
        print(a.shape)

