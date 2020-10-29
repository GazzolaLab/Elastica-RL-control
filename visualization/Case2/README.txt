This folder contains scripts to render visualizations of Elastica simulations used in the paper. In the code for training and testing, when you test a policy the solution history will be saved to the `./data/` folder. This folder is just copied here for convenience.  

To run the visualization need to install both ffmpeg and POVray. On Macs there are home-brew installs that generally work, we have not testing POVray on any other OS. 

To generate a video, there are several scripts you need to run in the `./rendering/` folder. 

1. python ./generate_POVray_files.py
2. ./render_frames.sh
3. ./make_vid.sh

generate_POVray_files.py converts the numpy arrays from the data folder to POVray objects. 

render_frames.sh renders individual frames based on these files. 

make_vid.sh combines the frames into a movie. 

You can edit the camera position in the `camera_position.inc` files. 