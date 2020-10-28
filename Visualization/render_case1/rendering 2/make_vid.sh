#!/bin/bash


ffmpeg -threads 8 -r 60 -i images/moving_arm%02d.png -b:v 90M -vcodec mpeg4 ./test_vid.mp4
