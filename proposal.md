---
layout: page
title: Proposal
permalink: /proposal/
---

## Introduction/Background

#### Introduction
Facial expression classification has been an important problem in ML and CV ever since the inception of both fields. Recognizing facial emotions has always been a subjective, time-consuming, and strictly human task. Well-designed ML models that can predict facial expressions, emotions, and attitudes from just a snapshot of a face have the potential to optimize processes in the realms of human-computer interaction, psychology, security, and marketing.

#### Literature Review
Artificial neural networks have been trained on data extracted from the Facial Action Coding System (FACS) to classify facial expressions based on the individual movements of specific muscle groups [1]. FACS extracts muscle movements from image headshots, which are then fed into the neural net. Others have directly fed facial images into CNNs after facial detection and background removal [2]. There have also been comparisons between classical machine learning models compared to newer deep learning models [3].

#### Benchmarks

| Dataset | Best Performing Model | Accuracy |
| ------- | --------------------- | -------- |
| AffectNet [4] | DDAMFN [5] | 67.03% |
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
3. Histogram of Oriented Gradient ([skimage.feature.hog](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html)): Captures image features by utilizing the frequency of gradients and their orientation. This is effective for localized edge detection which helps identify key features such as eyes and mouth.
4. Image augmentation: Stretch and rotate the existing dataset to increase model robustness and accuracy.

#### ML Algorithms/Models
1. Convolutional Neural Networks: A popular approach for image recognition. It utilizes the structure of a neural network and applies a kernel on each convolutional layer to detect patterns within images. Effective due to the ability to model complex non-linear relationships.
2. Support Vector Machine: Find the optimal hyperplane that differentiates image classes in the feature space. Effective in image classification tasks due to its ability to cope with high-dimensional data, like images. 
3. Multinomial Logistic Regression ([sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)): Multinomial Logistic Regression uses a sigmoid function to return the probability that a particular result belongs to a category. Maximum likelihood estimation can be used to optimize the model and minimize the cost. Effective at finding correlation of certain features and categories.

## Results and Discussion

#### Quantitative Metrics
The following metrics will be used as defined in lecture:
1. Accuracy
2. Precision
3. Recall
4. F1 Score
5. Confusion Matrix

#### Project Goals
* High overall accuracy. This shows the model has a high chance of correctly identifying facial expressions.
* Balanced precision and recall for each facial expression category. This demonstrates the model’s ability to correctly identify each expression without false positives/negatives. 
* Minimal bias. This ensures the model performs well regardless of the context (e.g. age, gender, ethnicity)


#### Expected Results
* Around 60~90% accuracy (lower using simpler models and higher using deep learning models such as CNN) [7]
* Some expressions are inherently more difficult to recognize e.g. smile and smirk
* Limitations under certain circumstances e.g. low-light


## Gantt Chart
Our Gantt Chart can be found [here](https://github.com/e019chen/ML-Facial-Expression-Recognition/blob/jekyll-integration/GanttChart.xlsx).


## Contribution Table

| Name | Proposal Contributions |
| ---- | ---------------------- |
| Andrew G. | Methods, Github Pages, Presentation |
| Andrew H. | Methods, Presentation, Slides |
| Chris | Introduction, Problem Definition, Presentation |
| Edison | Methods, Presentation |
| Euan | Results, Gantt Chart, Presentation |


## References
[1] B. Büdenbender, T. T. A. Höfling, A. B. M. Gerdes, and G. W. Alpers, “Training machine learning algorithms for automatic facial coding: The role of emotional facial expressions’ prototypicality,” PLOS ONE, vol. 18, no. 2, p. e0281309, Feb. 2023, doi: [https://doi.org/10.1371/journal.pone.0281309](https://doi.org/10.1371/journal.pone.0281309)

[2] V. Hosur, “Facial Emotion Detection Using Convolutional Neural Networks,” Available: IEEE Xplore, Oct. 2022, Accessed: Jan. 21, 2024. [Online]. [https://ieeexplore.ieee.org/document/9972510](https://ieeexplore.ieee.org/document/9972510)

[3] A. Mohammad Hemmatiyan-Larki, F. Rafiee-Karkevandi and M. Yazdian-Dehkordi, "Facial Expression Recognition: a Comparison with Different Classical and Deep Learning Methods," 2022 International Conference on Machine Vision and Image Processing (MVIP), Ahvaz, Iran, Islamic Republic of, 2022, pp. 1-5, doi: [https://doi.org/10.1109/MVIP53647.2022.9738553](https://doi.org/10.1109/MVIP53647.2022.9738553)

[4] “Facial Expression Recognition (FER) on AffectNet,” paperswithcode.com. [https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet](https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet)

[5] S. Zhang, H. Teng, Y. Zhang, Y. Wang, and Z. Song, “A Dual-Direction Attention Mixed Feature Network for Facial Expression Recognition,” Electronics, vol. 12, no. 17, pp. 3595–3595, Aug. 2023, doi: [https://doi.org/10.3390/electronics12173595](https://doi.org/10.3390/electronics12173595)

[6] “Facial Expression Recognition (FER) on RAF-DB,” paperswithcode.com. [https://paperswithcode.com/sota/facial-expression-recognition-on-raf-db](https://paperswithcode.com/sota/facial-expression-recognition-on-raf-d)

[7] T. Debnath, M. M. Reza, A. Rahman, A. Beheshti, S. S. Band, and H. Alinejad-Rokny, “Four-layer ConvNet to facial emotion recognition with minimal epochs and the significance of data diversity,” Scientific Reports, vol. 12, no. 1, p. 6991, Apr. 2022, doi: [https://doi.org/10.1038/s41598-022-11173-0](https://doi.org/10.1038/s41598-022-11173-0)