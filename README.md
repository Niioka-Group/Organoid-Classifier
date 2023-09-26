# Organoid-Classifier

## Overview
This is the source code of the manuscript.

This deep learning model predicts the percentage of the area expressing RAX on the image from a bright-field image of a developing hypothalamic-pituitary complex. The input is the image of the developing organoid (day30), and the predicted result is output as A (70<%RAX), B (40≤%RAX<70), and C (%RAX<40). 

More detail of this model are shown in our paper;

Human Pluripotent Stem Cell Culture Outcome Predicted by Deep Learning.


## Setup

Linux (Ubuntu 20.04 LTS)

GPU (A100, 80 GB, NVIDIA)

Python 

PyTorch 1.10.1+cu111

Torchvision 0.11.2+cu111


