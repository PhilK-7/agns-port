from scipy import io
import numpy as np


@DeprecationWarning
def load_dcgan_mat_model_weights(mat_file_path):
    """
    Loads model weights from a DCGAN .mat file that contains a dictionary.
    :param mat_file_path: a path to a .mat file
    :return: a list containing the weight matrices for every layer of a model, from start to end
    """

    mat_file = io.loadmat(mat_file_path)  # load .mat file containing data from path
    matrix = mat_file['matrices'][0]  # data in numpy array

    return [mat for mat in matrix]  # outer numpy array as list


@DeprecationWarning
def load_fr_mat_model_weights(mat_file_path):
    mat_file = io.loadmat(mat_file_path)
    print(mat_file)


'''
The meaning of the saved matrices was inferred from the original MATLAB code.

gen.mat weight parameters:
    (25, 7040) -> fully connected layer weights
    (1, 7040) -> batch normalization layer 1 mu
    (1, 7040) -> batch normalization layer 1 v
    (160, 80, 5, 5) -> deconvolutional layer 1 weights
    (1, 80) -> batch normalization layer 2 mu
    (1, 80) -> batch normalization layer 2 v
    (80, 40, 5, 5) -> deconvolutional layer 2 weights
    (1, 40) -> batch normalization layer 3 mu
    (1, 40) -> batch normalization layer 3 v
    (40, 20, 5, 5) -> deconvolutional layer 3 weights
    (1, 20) -> batch normalization layer 4 mu
    (1, 20) -> batch normalization layer 4 v
    (20, 3, 5, 5) -> deconvolutional layer 4 weights

discrim.mat weight parameters:
    (20, 3, 5, 5) -> convolutional layer 1 weights
    (40, 20, 5, 5) -> convolutional layer 2 weights
    (1, 40) -> batch normalization layer 1 mu
    (1, 40) -> batch normalization layer 1 v
    (80, 40, 5, 5) -> convolutional layer 3 weights
    (1, 80) -> batch normalization layer 2 mu
    (1, 80) -> batch normalization layer 2 v
    (160, 80, 5, 5) -> convolutional layer 4 weights
    (1, 160) -> batch normalization layer 3 mu
    (1, 160) -> batch normalization layer 3 v
    (7040, 1) -> fully connected
    
'''

if __name__ == '__main__':
    # (deprecated) show model weights from saved .mat files
    # load_fr_mat_model_weights('../../matlab-models/openface10-recognition-nn.mat')
    data = io.loadmat('../../matlab-models/openface10-recognition-nn.mat')
    print(data)
    print(data.keys())
    fw = data['None']
    print(np.shape(fw))
    '''
    print(data)
    print(len(data))
    for a in data:
        print(a.shape)'''
