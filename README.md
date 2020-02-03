# How to train YOLOv3 using Darknet on Google Colaboratory

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robingenz/object-detection-yolov3-google-colab/blob/master/Object_Detection_YOLOv3_Google_Colab.ipynb)

- [Introduction](#introduction)
- [Setup](#setup)
  - [Install dependencies](#install-dependencies)
  - [Check hardware accelerator](#check-hardware-accelerator)
  - [Mount Google Drive](#mount-google-drive)
- [Train](#train)
  - [Start a new training](#start-a-new-training)
  - [Continue training](#continue-training)
- [Test](#test)
  - [Run the detector](#run-the-detector)
  - [Display an image in VM](#display-an-image-in-vm)

## Introduction 

This guide explains how to train your own custom dataset with YOLOv3 using Darknet on Google Colaboratory.  

> Colaboratory is a research tool for machine learning education and research. It’s a Jupyter notebook environment that requires no setup to use.  

This guide uses the following folder structure:  

```bash
└── object-detection
    ├── backup                  # Folder for *.weights files
    ├── darknet53.conv.74       # Pretrained convolutional weights
    ├── data                    # Folder for images and labels
    ├── object.names            
    ├── test.txt
    ├── train.txt
    ├── trainer.data            
    └── yolov3-tiny.cfg         # YOLOv3 configuration file
```

You can find more information at https://pjreddie.com/darknet/yolo/.  

**Jupyter notebook**: This repository also contains a [Jupyter notebook](https://github.com/robingenz/object-detection-yolov3-google-colab/blob/master/Object_Detection_YOLOv3_Google_Colab.ipynb). 
You can load this public notebook directly from GitHub (with no authorization step required): [Object_Detection_YOLOv3_Google_Colab.ipynb](https://colab.research.google.com/github/robingenz/object-detection-yolov3-google-colab/blob/master/Object_Detection_YOLOv3_Google_Colab.ipynb)

## Setup

### Install dependencies

First you need to install all the required system dependencies:
```python
!apt-get update
!apt-get upgrade

!apt-get install -y build-essential
!apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
!apt-get install -y libavcodec-dev libavformat-dev libswscale-d
!apt-get install -y libopencv-dev

!apt-get install -y g++-5
!apt-get install -y gcc-5
```

### Check hardware accelerator

Then you have to select the GPU as hardware accelerator (Edit -> Notebook Settings).  
To check this:  

```python
!/usr/local/cuda/bin/nvcc --version

import tensorflow as tf
device_name = tf.test.gpu_device_name()
if not '/device:GPU:0' in device_name:
    print('\nERROR: GPU is not selected as hardware accelerator!')
else:
    print(device_name)
```

Darknet needs to be cloned and compiled. 
You will use the [forked version](https://github.com/AlexeyAB/darknet) from [AlexeyAB](https://github.com/AlexeyAB).  

```python
!git clone https://github.com/AlexeyAB/darknet
%cd darknet

!ls
!sed -i 's/OPENCV=0/OPENCV=1/g' Makefile
!sed -i 's/GPU=0/GPU=1/g' Makefile
!make
```

### Mount Google Drive

There are [different ways](https://colab.research.google.com/notebooks/io.ipynb) to load and save data from external sources.
I recommend to use Google Drive.
If you prefer another option, you can skip this step.  

You can mount your Google Drive on your runtime using an authorization code:

```python
from google.colab import drive
drive.mount('/content/gdrive')

!ls "/content/gdrive/My Drive/"
```

To avoid mistakes, I recommend creating a symlink:

```python
!ln -s "/content/gdrive/My Drive/object-detection/" "/content/object-detection"
```

You can unmount Google Drive with this command:  

```python
drive.flush_and_unmount()
```


## Train

### Start a new training

Start a new training:  

```python
!./darknet detector train /content/object-detection/trainer.data /content/object-detection/yolov3-tiny.cfg /content/object-detection/darknet53.conv.74 -dont_show 
```

### Continue training

Continue training:  

```python
!./darknet detector train /content/object-detection/trainer.data /content/object-detection/yolov3-tiny.cfg backup/yolov3-tiny_last.weights -dont_show 
```

## Test

### Run the detector

To run the detector for `data/test.jpg`:

```python
!./darknet detector test /content/object-detection/trainer.data /content/object-detection/yolov3-tiny.cfg /content/object-detection/backup/yolov3-tiny_last.weights /content/object-detection/data/test.jpg -dont_show 
```

Darknet prints out the objects it detected, its confidence, and how long it took to find them.  
Darknet will save the detections in `./predictions.jpg`.

### Display an image in VM

The following function will help you to display the image in the remote VM:  

```python
def detect(path):
  import cv2
  import matplotlib.pyplot as plt
  %matplotlib inline

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()

detect('predictions.jpg')
```

Thanks to [Ivan Goncharov](https://github.com/ivangrov) for this helper function!