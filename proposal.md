---
layout: page
title: Proposal
permalink: /proposal/
---

## Introduction/Background

#### Introduction
Facial expression classification has been an important problem in ML and CV ever since the inception of both fields. Recognizing facial emotions has always been a subjective, time-consuming, and strictly human task. Well-designed ML models that can predict facial expressions, emotions, and attitudes from just a snapshot of a face have the potential to optimize processes in the realms of human-computer interaction, psychology, security, and marketing.

#### Literature Review
Artificial neural networks have been trained on data extracted from the Facial Action Coding System (FACS) to classify facial expressions based on the individual movements of specific muscle groups [1]. FACS extracts muscle movements from image headshots, which are then fed into the NN. Others have directly fed facial images into CNNs after facial detection and background removal. There have also been comparisons between classical machine learning models compared to newer deep learning models [7].

#### Benchmarks

| Dataset | Best Performing Model | Accuracy |
| ------- | --------------------- | -------- |
| AffectNet [3] | DDAMFN [4] | 67.03% |
| RAF_DB [6]    | PAtt-Lite  | 95.05% |

#### Datasets

| Dataset (Linked) | Datapoints |
| ------- | --------------------- | -------- |
| [AffectNet](http://mohammadmahoor.com/affectnet/) | > 400,000 |
| [RAF_DB](http://www.whdeng.cn/raf/model1.html#dataset)    | > 30,000  |
| [Kaggle Face Expression Recognition](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset/) | > 10,000 |



## Problem Definition
Problem: Classifying human face expressions is difficult, subjective, and time-consuming, but is incredibly useful in many fields such as:
* Psychology: Mental health diagnostics and treatment by detecting subtle signs of negative emotional outlooks.
* Security and Justice: Analyze emotions during testimonies to predict criminal intent, testimonial legitimacy, or suspicious behaviors.
* Marketing: Assess marketing performance and advertising effectiveness, or fine tune targeted advertising campaigns to ensure positive emotional responses.


## Methods
#### Data Preprocessing Methods
1. Gaussian Blur: Effectively reduce noise by applying a Gaussian kernel on the image, discarding unwanted information that may impair the performances of machine learning models
2. Sobel Edge Detection: Effectively captures the edges, regions with high spatial frequency, in the image using a Sobel operator (a kind of convolutional kernel), extracting essential facial features for the classification task.
3. Histogram of Oriented Gradient (skimage.feature.hog): Captures image features by utilizing the frequency of gradients and their orientation. This is effective for localized edge detection which helps identify key features such as eyes and mouth.
4. Image augmentation: Stretch and rotate the existing dataset to increase model robustness and accuracy.

#### ML Algorithms/Models
1. Convolutional Neural Networks: A popular approach for image recognition. It utilizes the structure of a neural network and applies a kernel on each convolutional layer to detect patterns within images. Effective due to the ability to model complex non-linear relationships.
2. Support Vector Machine: Find the optimal hyperplane that differentiates image classes in the feature space. Effective in image classification tasks due to its ability to cope with high-dimensional data, like images. 
3. Multinomial Logistic Regression (sklearn.linear_model.LogisticRegression): Multinomial Logistic Regression uses a sigmoid function to return the probability that a particular result belongs to a category. Maximum likelihood estimation can be used to optimize the model and minimize the cost. Effective at finding correlation of certain features and categories.

## Results and Discussion

