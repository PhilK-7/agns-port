#!/usr/bin/env bash
# just add a '10' in next line to align 10-class dataset instead
cd agns-py/data/pubfig/dataset_10

# WARNING: this also deletes an existing aligned directory
echo Deleting aligned images, if exist...
rm -r ../dataset_aligned
LOCATION=$(pwd)
DLIBPATH=../../../../src/dependencies

# compose paths for script and shape predictor
SCRIPTPATH="$DLIBPATH/face_alignment.py"
PREDPATH="$DLIBPATH/shape_predictor_5_face_landmarks.dat"

echo Current directory: $LOCATION
echo Number of subjects: $(ls | wc -l)

# replace annoying whitespaces, if necessary
for d in *\ *; do mv "$d" "${d// /_}"; done

echo Start face alignment...
for dir in $(ls | egrep -i '[A-Z]\w+_[A-Z]\w+'); do
	cd $dir
	echo Aligning faces of $(pwd)
	for img in $(ls); do
		
		python $SCRIPTPATH $PREDPATH "$(pwd)/$img" 
	done
	cd ..
done

# now copy all aligned directories into a single dataset folder
cd ..
echo Moving all aligned images to new dataset folder... 
mkdir dataset_aligned
cd dataset_aligned
DSAPATH=$(pwd)
cd $LOCATION
for dir in $(ls | egrep -i '[A-Z]\w+_[A-Z]\w+'); do
	mkdir "$DSAPATH/$dir"
	mv "$dir/aligned" "$DSAPATH/$dir"
done

echo DONE!
