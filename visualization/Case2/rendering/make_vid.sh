#!/bin/bash


ffmpeg -threads 8 -r 30 -i images/moving_arm%02d.png -b:v 90M -vcodec mpeg4 ./output_video.mp4
