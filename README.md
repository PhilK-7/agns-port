# agns-port
Repository for the Python port of the "agns" (Adversarial Generative Networks) paper code.

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
The original files (glasses dataset, PubFig dataset) are provided. Because the original, full PubFig dataset could
not be downloaded from the main source, a third-party provided subset was used. From there, 143 directories were
selected. Those give the basis for the 143 classes. Another subset of those classes was chosen for the 10 classes
problem formulation, as the non-celebrity (researcher) images were not provided.

At the top level directory, a shell script 'align_all.sh' is provided, in case you want to create aligned versions
from the original images again, you can e.g. execute `bash align_all.sh`(it will take some minutes).
It used Dlib to align face images to a 68-landmark pose.

## Models
The face recognition model are based on VGG-16, as well as FaceNet (OpenFace) small version.
The models were trained on aligned images in the given PubFig dataset.
The DCGAN´s purpose is to generate fake eyeglasses, just like those in the provided eyeglasses
dataset. As small change to the paper, mini-batch discrimination was added to the generator
in order to have a wider diversity of colors, and slightly better looking results.

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
not necessarily tested with the current state of the code.

## Execution
There are different demos, all files that are exclusively for demonstration purposes (but might do
more) start with 'demo'. Go to demo_main and start the script, which lets you pick which of the
normal demos to execute. You can specifiy the GPU(s) to use in parameter `gpus`.
... Also take note that content plotted with Matplotlib remotely will show up in the SciView of
PyCharm Pro.
