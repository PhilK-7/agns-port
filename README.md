# agns-port
Repository for the WIP Python port of the "agns" (Adversarial Generative Networks) paper code.

## Requirements
- Python 3.8+
- at least one NVidia GPU, with 10+ GB VRAM
- the correct NVidia CUDA version + CUDNN installed
- at least 2 GB of free storage

## Dependencies
install requirements in your venv with `pip install -r requirements.txt`
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
from the original images again, you can e.g. execeute `bash align_all.sh`(it will take some minutes).
It used Dlib to align face images to a 68-landmark pose.

## Execution

