# agns-port
Rep for the WIP Python port of the "agns" paper code.

# Dependencies
install requirements in your venv with `pip install -r requirements.txt`
Python 3.8 recommended.
It is *not* recommended using another Tensorflow, Numpy version etc. than provided. Especially Tensorflow can cause
some unexpected problems with this.
The code has a directory 'dependencies' that just contains a shape predictor file and a script that calls Python
bindings for Dlib.

# Images
The original files (glasses dataset, PubFig dataset) are provided. Because the original, full PubFig dataset could
not be downloaded from the main source, a third-party provided subset was used. From there, 143 directories were
selected. Those give the basis for the 143 classes. Another subset of those classes was chosen for the 10 classes
problem formulation, as the non-celebrity (researcher) images were not provided.

At the top level directory, a shell script 'align_all.sh' is provided, in case you want to create aligned versions
from the original images again. It used Dlib to align face images to a 68-landmark pose.

# Execution

