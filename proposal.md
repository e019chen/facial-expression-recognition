---
layout: page
title: Proposal
permalink: /proposal/
---

## Background

#### Introduction
Facial expression classification has been an important problem in ML and CV ever since the inception of both fields. Well-designed models that can predict facial emotions (a previously human-task) have the potential to optimize processes in the realms of human-computer interaction, psychology, security, and marketing.

#### Literature Review
* Artificial neural networks have been trained on data extracted from the Facial Action Coding System to classify emotions based on the movements of specific muscle groups [1]. 
* Facial images have been fed directly into CNNs after image preprocessing [2]. 
* The effectiveness of classical ML models compared to newer deep learning models has been studied [3].

#### Benchmarks

| Dataset | Best Performing Model | Accuracy |
| ------- | --------------------- | -------- |
| AffectNet [4] | DDAMFN [5] | 67.03% |
| RAF_DB [6]    | PAtt-Lite  | 95.05% |

#### Datasets

| Dataset (Linked) | Labeled Datapoints |
| ------- | --------------------- | -------- |
| [AffectNet](http://mohammadmahoor.com/affectnet/) | 400,000+ |
| [RAF_DB](http://www.whdeng.cn/raf/model1.html#dataset)    | 30,000+  |
| [Kaggle Face Expression Recognition](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset/) | 10,000+ |


## Problem Definition
**Problem**: Classifying human face expressions is difficult, subjective, and time-consuming, but is incredibly useful in many fields such as:
* Psychology: Mental health diagnostics by detecting signs of negative emotional outlooks.
* Security and Justice: Analyze emotions during testimonies to predict criminal intent or suspicious behaviors.
* Marketing: Assess performance and advertising effectiveness, or fine tune targeted campaigns to reinforce positive emotions.


## Methods
#### Data Preprocessing Methods
1. **Gaussian Blur**: Reduce noise by applying a Gaussian kernel on the image, discarding unwanted information that may impair the performances of ML models
2. **Sobel Edge Detection**: Capture the edges and regions with high spatial frequency in the image using a Sobel operator (a convolutional kernel), extracts essential facial features for classification.
3. **Histogram of Oriented Gradient** ([skimage.feature.hog](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html)): Captures image features by utilizing the frequency of gradients and their orientation. Effective for localized edge detection to identify key face features (eyes/mouth).
4. **Image Augmentation**: Stretch and rotate to increase model robustness and accuracy.

#### ML Algorithms/Models
1. **Convolutional Neural Networks**: A popular approach for image recognition. Applies a kernel on each convolutional layer in the NN to detect specific patterns. Can model complex non-linear relationships.
2. **Support Vector Machine**: Finds the optimal hyperplane that differentiates image classes in the feature space. Effective in image classification due to its ability to cope with high-dimensional data.
3. **Multinomial Logistic Regression** ([sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)): Uses a sigmoid function to return the probability of category membership. MLE can be used to optimize the model and minimize the cost. Effective at finding the correlation of certain features and categories.

## Results and Discussion

#### Quantitative Metrics
The following classification metrics will be used as defined in lecture:
1. Accuracy
2. Precision
3. Recall
4. F1 Score
5. Confusion Matrix

#### Project Goals
* High overall accuracy. Shows a high chance of correctly identifying facial expressions.
* Balanced precision and recall for each facial expression category. Demonstrates the ability to correctly identify each expression without false positives/negatives. 
* Minimal bias. Ensures high performance regardless of the context (age, gender, ethnicity)


#### Expected Results
* Around 60~90% accuracy (lower using simpler models and higher using deep learning models) [7]
* Categorize seven expressions: angry, disgust, fear, happy, neutral, sad, surprise
    * Some expressions harder to differentiate
* Image limitations (low-lighting, obstructed views)


## Gantt Chart
Our Gantt Chart can be found [here](https://github.com/e019chen/facial-expression-recognition/blob/jekyll-integration/GanttChart.xlsx).


## Contribution Table

| Name | Proposal Contributions |
| ---- | ---------------------- |
| Andrew G. | Methods, Github Pages, Presentation |
| Andrew H. | Methods, Presentation, Slides |
| Chris | Introduction, Literature Review, Problem Definition, Presentation |
| Wei-Liang | Methods, Presentation |
| Euan | Results, Gantt Chart, Presentation |


## References
[1] B. Büdenbender, T. T. A. Höfling, A. B. M. Gerdes, and G. W. Alpers, “Training machine learning algorithms for automatic facial coding: The role of emotional facial expressions’ prototypicality,” PLOS ONE, vol. 18, no. 2, p. e0281309, Feb. 2023, doi: [https://doi.org/10.1371/journal.pone.0281309](https://doi.org/10.1371/journal.pone.0281309)

[2] V. Hosur, “Facial Emotion Detection Using Convolutional Neural Networks,” Available: IEEE Xplore, Oct. 2022, Accessed: Jan. 21, 2024. [Online]. [https://ieeexplore.ieee.org/document/9972510](https://ieeexplore.ieee.org/document/9972510)

[3] A. Mohammad Hemmatiyan-Larki, F. Rafiee-Karkevandi and M. Yazdian-Dehkordi, "Facial Expression Recognition: a Comparison with Different Classical and Deep Learning Methods," 2022 International Conference on Machine Vision and Image Processing (MVIP), Ahvaz, Iran, Islamic Republic of, 2022, pp. 1-5, doi: [https://doi.org/10.1109/MVIP53647.2022.9738553](https://doi.org/10.1109/MVIP53647.2022.9738553)

[4] “Facial Expression Recognition (FER) on AffectNet,” paperswithcode.com. [https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet](https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet)

[5] S. Zhang, H. Teng, Y. Zhang, Y. Wang, and Z. Song, “A Dual-Direction Attention Mixed Feature Network for Facial Expression Recognition,” Electronics, vol. 12, no. 17, pp. 3595–3595, Aug. 2023, doi: [https://doi.org/10.3390/electronics12173595](https://doi.org/10.3390/electronics12173595)

[6] “Facial Expression Recognition (FER) on RAF-DB,” paperswithcode.com. [https://paperswithcode.com/sota/facial-expression-recognition-on-raf-db](https://paperswithcode.com/sota/facial-expression-recognition-on-raf-d)

[7] T. Debnath, M. M. Reza, A. Rahman, A. Beheshti, S. S. Band, and H. Alinejad-Rokny, “Four-layer ConvNet to facial emotion recognition with minimal epochs and the significance of data diversity,” Scientific Reports, vol. 12, no. 1, p. 6991, Apr. 2022, doi: [https://doi.org/10.1038/s41598-022-11173-0](https://doi.org/10.1038/s41598-022-11173-0)