# Organoid-Classifier

## Overview
This is the source code of the manuscript.

This deep learning model predicts the percentage of the area expressing RAX on the image from a bright-field image of a developing hypothalamic-pituitary complex. The input is the image of the developing organoid (day30), and the predicted result is output as A (70<%RAX), B (40≤%RAX<70), and C (%RAX<40). 

More detail of this model are shown in our paper;

Human Pluripotent Stem Cell Culture Outcome Predicted by Deep Learning.

## Folders

“ROC_AUC” contains the formula for calculating ROC and AUC for our models and experts

“eff” contains the code for training our EfficientNet model, and evaluating and visualizing the results of classification.

“vit” contains the code for training our Visual Transformer model, and evaluating and visualizing the results of classification.

“ensemble” contains the code classifying the images using the ensemble of our models, EfficientNet and Visual Transformer.


## Setup

Linux (Ubuntu 20.04 LTS)

GPU (A100, 80 GB, NVIDIA)

Python 

PyTorch 1.10.1+cu111

Torchvision 0.11.2+cu111

## License

Organoid-Classifier is licensed under the GPL-3.0 license.
