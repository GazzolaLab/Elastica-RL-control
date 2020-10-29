#!/bin/bash

# parallel version based on https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop
# Set up working variables
DIR=$PWD/images
STARTFOLDER=$PWD
IMAGES=$PWD/images

# Prepare image folder
cd $DIR


# In all povray files replace the default include file with the appropriate one
SEDSTRING=s/scenepovray/${SCENENAME}/g
sed -ie ${SEDSTRING} *.pov


task(){
   echo "Processing $f file..."

	filenameOnly=${f##*/}
	filenameNoExt=${filenameOnly%.*}
	echo $filenameNoExt
	echo $filenameOnly

	## Select your preferred resolution
    # povray -H2160 -W3840 Quality=11 Antialias=on $filenameOnly
 	povray -H1080 -W1920 Quality=11 Antialias=on $filenameOnly
	# povray -H250 -W400 Quality=11 Antialias=on $filenameOnly Verbose=Off
	# povray -H120 -W200 Quality=11 Antialias=on $filenameOnly Verbose=Off
}

# Increasing N runs the loop in parallel
N=1
(
# Loop over all povray files and produce images
for f in $DIR/moving_arm**.pov
do
   ((i=i%N)); ((i++==0)) && wait
   task & 
done
)

wait

# Go back to original folder
cd $STARTFOLDER
