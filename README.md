# Machine-Learning-ND_Capstone
## Capstone Project Report
Kamil Kaczmarczyk
September 20th, 2017

## Vision-Perception Module
## Situational Awareness System for Autonomous Vehicles

### Description

This is a repository of a capstone project of Machine Learning NanoDegree from Udacity.

### Goal

Goal of the work is to detect and localize aircraft and birds in images or frames of a video that from aircraft point of view in the air so that potential hazards and obstacles in the air could be avoided.

### Documentation

Full detailed report is provided in the [capstone_report.md](https://github.com/Kamil-K/Machine-Learning-ND_Capstone/blob/master/capstone_report.md) available also in a [.pdf](https://github.com/Kamil-K/Machine-Learning-ND_Capstone/blob/master/capstone_report.pdf) format.

### Design

Code is written in Python using TensorFlow library for Machine Learning.

It is divided into three chapters:

- **Chapter 1:** Data pre-processing - [Capstone Part 01 - Dataset Preparation & Exploration.ipynb](https://github.com/Kamil-K/Machine-Learning-ND_Capstone/blob/master/Capstone%20Part%2001%20-%20Dataset%20Preparation%20%26%20Exploration.ipynb)
- **Chapter 2:** Approach 1 implementation based on Support Vector Machines and Histogram of Oriented Gradients - [Capstone Part 02 - Apply SVM.ipynb](https://github.com/Kamil-K/Machine-Learning-ND_Capstone/blob/master/Capstone%20Part%2002%20-%20Apply%20SVM.ipynb)
- **Chapter 3:** Approach 2 implementation based on Transfer Learning from Convolutional Neural Network AlexNet - [Capstone Part 03 - Apply CNN with Transfer Learning from AlexNet.ipynb](https://github.com/Kamil-K/Machine-Learning-ND_Capstone/blob/master/Capstone%20Part%2003%20-%20Apply%20CNN%20with%20Transfer%20Learning%20from%20AlexNet.ipynb)

### Pre-requisite files

It is required for the Chapter 3 work for Transfer Learning to use pre-trained model weights which can be accessed [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d880c_bvlc-alexnet/bvlc-alexnet.npy)

### Dataset:

Dataset used for this project consists of pictures downloaded from the popular web search engine google in the section of images. It contains and is divided into four distinct classes of pictures:
- aircraft pictures of Boeing 737 and Cessna 172 models in flight - 400 images<p>
![alt tex](https://github.com/Kamil-K/Machine-Learning-ND_Capstone/blob/master/dataset_examples/aircraft/0_.jpg "Aircraft")
- birds pictures also in flight mostly on the background of sky - 367 images<p>
![alt tex](https://github.com/Kamil-K/Machine-Learning-ND_Capstone/blob/master/dataset_examples/birds/0_.jpg "Bird")
- sky images containing either a clear sky or clouded and ocluded sky images - 407 images<p>
![alt tex](https://github.com/Kamil-K/Machine-Learning-ND_Capstone/blob/master/dataset_examples/sky/0_.jpg "Sky")
- ground images containing a mix of various pictures from flight of cities, fields, mountains and other landscapes where most of the image area is covered by ground so that it does not contain a lot of sky in it - 407 images<p>
![alt tex](https://github.com/Kamil-K/Machine-Learning-ND_Capstone/blob/master/dataset_examples/ground/0_.jpg "Ground")

Sample of dataset images are available [here](https://github.com/Kamil-K/Machine-Learning-ND_Capstone/tree/master/dataset_examples)

### Sample final results

The models results on test images - accuracies are:
- **Approach 1: 0.8805** for the Support Vector Machines based on Histogram of Orientation Gradients features
- **Approach 2: 0.9916** for the Convolutional Neural Networks based on AlexNet and Transfer Learning from ImageNet dataset

For approach 2 with CNN Transfer Learning based on AlexNet sample on test images of detection and classification as well as final precision and recall of aircraft and bird class are presented below.

**Approach 2 - Transfer Learning applied on AlexNet CNN on New Images**
 
**Confusion Matrix**
 
 |              |  Aircraft (actual) | Bird (actual) | Non-Aircraft & Non-Bird (actual) |
|--------------|--------------------|-----------------------|--------------------------|
|   Aircraft (predicted)   |     11     |      0       |      0       |
| Bird (predicted) |     0      |      3       |      1       |
| Non-Aircraft & Non-Bird (predicted) |     0      |      3       |      X       |
 
 **Precision & Recall**
 
|                          |  Aircraft          | Bird                  |
|--------------------------|--------------------|-----------------------|
|   Precision              |     1              |      0,75             |
| Recall                   |     1              |      0,5              |

 **Object Detection & Localization**
 
| Column 1  | Column 2 | Column 2  | Column 4 | Column 5  |
|-----------|----------|-----------|----------|-----------|
| Original test image | Aircraft detections bounding boxes |aircraft detections heatmap - thresholded | birds detections bounding boxes | birds detections heatmap - thresholded |

 ![alt tex](https://github.com/Kamil-K/Machine-Learning-ND_Capstone/blob/master/resources/CNN_final.png "CNN Final") 