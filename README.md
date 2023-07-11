# SatelliteImageClassificationUsingPython
In this repository, we are classifying a Landsat 8 image using Machine learning. We have classified an image of the Kathmandu valley and its surrounding areas into four major landcover types, viz. Forest, Agriculture, Urban and Water. 

# Kathmandu Valley Satellite Image Classification

This repository contains the code and data for classifying a satellite image of the Kathmandu Valley and its neighboring areas into four landcovers, viz. Forest, Urban, Water, and Open/cultivated Land with an accuracy of 86 percent.

## Introduction

The objective of this project is to classify a satellite image of the Kathmandu valley and its neighboring areas into four land covers, viz. Forest, Urban, Water, and Open/cultivated Land with an accuracy of 86 percent in Python. The image was preprocessed using ArcMap 10.8.2 and training samples were taken from the preprocessed image. Callback was performed during training, checkpoint was saved during training and hyperparameter tuning was performed during training.
All the references were taken from the ''https://github.com/pratik-tan10/Python/blob/main/Notebooks/Satellite%20Image%20Classification%20Using%20ArcMap%20and%20Python.ipynb'' 
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- ArcMap 10.8.2

### Installing

1. Clone the repository.
2. Open the project in your preferred IDE.
3. Install the required packages using pip.

```sh
pip install -r requirements.txt
```

### Preprocessing

The image was preprocessed using ArcMap 10.8.2. The following steps were performed:

1. The image was imported into ArcMap.
2. The image was georeferenced.
3. The image was clipped to the study area.
4. The image was resampled to a resolution of 30 meters.
5. The image was converted to reflectance values.
6. Training samples were taken from the preprocessed image.

### Classification using Python

The following steps were performed during classification:

1. Callback was performed during training.
2. Checkpoints were saved during training.
3. Hyperparameter tuning was performed during training.

## Results

The satellite image was classified into four land covers with an accuracy of 86 percent using TensorFlow to train the model, gdal to read the image, and keras_tuner for hyperparameter tuning.

## Authors

-  Pratyush Dhungana(https://github.com/pratyush-dh)

## Acknowledgments

- Pratik Dhungana (https://github.com/pratik-tan10)
```

