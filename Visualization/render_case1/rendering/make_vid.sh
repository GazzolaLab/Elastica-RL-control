#!/bin/bash


ffmpeg -threads 8 -r 180 -i images/moving_arm%02d.png -b:v 90M -vcodec mpeg4 ./Problem1_1x-realtime.mp4
