# agns-port
Repository for the Python port of the "agns" (Adversarial Generative Networks) paper.

## Requirements
- Python 3.8+
- the packages listed in the requirements.txt
- plus pyviz package and Graphviz additionaly if you want graphs of the models 
(comment out `plot_model` to ignore)
- at least one NVidia GPU, with 11+ GB VRAM
- the correct NVidia CUDA version + CUDNN installed
- at least 2 GB of free storage

## Dependencies
Install requirements in your venv with `pip install -r requirements.txt`.
Python 3.8 recommended.
It is *not* recommended using another Tensorflow, Numpy version etc. than provided. Especially Tensorflow can cause
some unexpected problems with this.
The code has a directory 'dependencies' that just contains a shape predictor file and a script that calls Python
bindings for Dlib.

## Images
The files (glasses dataset, PubFig dataset) are provided. Because the original, full PubFig dataset could
not be downloaded from the main source, a third-party provided subset was used. From there, 143 directories were
selected. Those give the basis for the 143 classes. Another subset of those classes was chosen for the 10 classes
problem formulation, as the non-celebrity (researcher) images were not provided.

Only a few images of the researcher(s) were given. This is not sufficient to include them as targets
in a face recognition network, but enough to perform impersonation attacks.

At the top level directory, a shell script 'align_all.sh' is provided, in case you want to create aligned versions
from the original images again, you can e.g. execute `bash align_all.sh` (it will take some minutes).
It uses Dlib to align face images to a 68-landmark pose.

#### Datasets (data/)

| Directory | What | # Images |
| :---------| :----| :--------|
| eyeglasses | normal eyeglasses images (+ 1 mask) | 16681 |
| eyeglasses/cropped | eyeglasses images cropped to 64x176 | 16680 |
| pubfig/dataset_ | some celebrities from PubFig (143) | 6551 |
| pubfig/dataset_10 | 10 selected celebrities from dataset_ | 321 |
| pubfig/dataset_aligned | images from dataset_ aligned to 68-landmarks-pose | 6551 |
| pubfig/dataset_aligned_10 | aligned version of dataset_10 | 321 |
| demo-data2 | researcher Mahmood Sharif wearing glasses with green marks | 6 |

## Models
The face recognition model are based on VGG-16, as well as FaceNet (OpenFace) small version.
The models were trained on aligned images in the given PubFig dataset.
The DCGAN´s purpose is to generate fake eyeglasses, just like those in the provided eyeglasses
dataset. As small change to the paper, mini-batch discrimination was added to the generator
in order to have a wider diversity of colors, and slightly better looking results.

#### Face Recognition Models (networks/face_nets.py)

| Model | Based on | Save name | Size (params) | Image input size | Values input range | Trained on |
| :-----| :--------| :---------| :-------------| :----------------| :------------------| :--------- |
VGG10   | VGG-16   | vgg_10.h5 |   134,301,514 | 224 x 224        | [0., 1.] | pubfig/dataset_aligned_10
VGG143  | VGG-16   | vgg_143.h5|   134,846,415 | 224 x 224        | [0., 1.] | pubfig/dataset_aligned
OF10    | FaceNet  | of10.h5   |     3,744,958 | 96 x 96          | [-1., 1.] | pubfig/dataset_aligned_10
OF143   | FaceNet  | of143.h5  |     3,821,215 | 96 x 96          | [-1., 1.] | pubfig/dataset_aligned

#### DCGAN (networks/dcgan.py, dcgan_utils.py, eyeglass_discriminator.py, eyeglass_generator.py)
The parts of the DCGAN (Deep Convolutional Adversarial Generative Network) model used for generating eyeglasses.
The only difference in the architecture (compared to the paper) is additional mini-batch discrimination that
was used to generate more diverse and slightly better fake images. Note that 'mode collapse' is still possible,
as it is a problem of GANs in general.

| Model | Save Name | Size (params) | Output | Values input range |
| :-----| :---------| :-------------| :------| :------------------|
| Generator | gweights (TF format) | 633,403 | 64 x 176 RGB image | (standard) normal distribution |
| Discriminator | dweights (TF format) | 3,809,801 |  value in [0., 1.], confidence fake/real | [-1., 1.] |

#### Special Layers (networks/special_layers.py)

| Class Name | Functionality |
| :----------| :-------------|
| LocalResponseNormalization | LR normalization within specific radius |
| L2Pooling | special average pooling with Euclidean norm |
| L2Normalization | just Euclidean norm |
| InceptionModule | 'original' Inception Module, four parallel Conv/Pool paths |
| InceptionModuleShrink | similar, but only three paths, and shrinks image |
| BlackPadding | pads glasses images with black, also applies mask to remove artifacts |
| FaceAdder | merges fake glasses and faces |
| Resizer | resizes images (might also scale values) |


## Training
There are already pretrained models provided in saved-models. If you want to train the face recognition
models (VGG / OpenFace 10/143), go to `face_nets.py`. It has two functions to train those models
from scratch, or also to continue training existing models.
If in the right directory ('../../saved-models') saved models exist, the training continues from that
checkpoint, otherwise training is started from the begin.
The VGG models are easier to train.
The training function for those is `train_vgg_dnn`. Pay attention that the parameter `bigger_class_n`
determines whether the 143-class version is trained, or the 10-class one.
For 
training the OpenFace models, it is very recommended to properly pretrain the base model first with
`pretrain_openface_model`. Then, if the loss is considerably low, continue and train the full model
with `train_of_dnn`.

There is no guarantee to succeed at reaching your goal when training a deep learning model. It is
highly non-deterministic, and different things can go wrong. In general, it is good practice to
keep a validation set during training, verify model functionality by hand, and better train a
model n times for x epochs, instead of training it for n * x epochs straight. Why?
At least here in this code, the training and validation splits are different each time. This means
each time, the alignment of data is different. Also it makes sense to change the learning rate between
different training sessions, decreasing it over time. The validation accuracy is a weak, but useful
indicator how much a model is progressed. Don´t train a model too much: this means it overfits to
the data, and thus generalizes poorly. If the validation accuracy only drops for an extended period
of time, this is an indicator of overfitting. It is recommended to finish the training process at a
rather stable validation accuracy.

Trained models can be loaded with `tf.keras.load_model`. The most popular save formats are .h5 and
the Tensorflow format (recognized by .index / .data-... endings). Some models, particularly those
that use custom implemented layers that receive extra parameters, need to be loaded with explicitly
given custom objects to be restored.

## Code Remarks
Functions not used anymore are marked with a DepreciationWarning. They still might be useful, but are
not necessarily tested with the current state of the code. There is also a deprecated package.
There are some code snippets in functions left commented out. Their purpose was to show or 
save images, intermediate results in different functions.


## Execution
There are different demos, all files that are mainly demonstrating functionality (but might do
more) start with 'demo'. Go to demo_main and start the script, that lets you pick which of the
normal demos to execute. You can specify the GPU(s) to use in parameter `gpus`.

Also take note that content plotted with Matplotlib remotely will show up in the SciView of
PyCharm Pro.

### Demos

| File | Purpose |
| :----| :-------|
| demo_main.py | launch other demos
| demo_face_recognition.py | select and test face recognition models
| demo_generate_eyeglasses.py | generate fake eyeglasses
| demo_dodging.py | perform dodging attack
| demo_impersonation.py | perform impersonation attack
| demo_impersonation_real.py | perform physical impersonation attack
