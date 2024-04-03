---
layout: page
title: Midterm Checkpoint
permalink: /midterm-checkpoint/
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

For this iteration, we decided to use the Kaggle Face Expression Recognition dataset for several reasons. First, it is already pre-processed to black and white, 48*48 images, which is light enough for us to quickly start implementation. Next, the Kaggle dataset is already grouped into classes, allowing us to do supervised learning without manually labelling the data.


## Problem Definition
**Problem**: Classifying human face expressions is difficult, subjective, and time-consuming, but is incredibly useful in many fields such as:
* Psychology: Mental health diagnostics by detecting signs of negative emotional outlooks.
* Security and Justice: Analyze emotions during testimonies to predict criminal intent or suspicious behaviors.
* Marketing: Assess performance and advertising effectiveness, or fine tune targeted campaigns to reinforce positive emotions.


## Methods
### Method 1 - Convolutional Neural Network (CNN)  
#### Data Preprocessing
The original facial expression dataset from Kaggle contains only a train and a test folder. To validate our model and get potential ways to adjust the hyperparameters, we select a portion of the original training dataset as the validation dataset. We now have a train: validate: test ratio of roughly 70%:15%:15%. 

Before feeding the images in the CNN model, we utilize Tensorflow’s ImageDataGenerator to apply random shifting, shearing, zooming, and flipping to help the model generalize any variations in the data. Additionally, we normalize the images so the pixel intensities are between 0 and 1 instead of 0 and 255. This helps the gradient calculations stay consistent and reduces the training time. 

Lastly, we dropped the class “disgust” from our dataset since it had a significantly smaller number of images compared to other classes. This will help us prevent the risks of imbalanced data, such as the model becoming biased towards majority classes. 

#### ML Algorithm/Model
The first ML Algorithm our team implemented is the Convolutional Neural Network (CNN) model. At a high level, CNN works by applying kernels (or filters) in convolutional layers to create feature maps. These feature maps highlight important attributes for the classification tasks. CNNs have become the to-go model for many image classification tasks, including facial expression recognition (FER) due to its effectiveness. 

One reason behind this is the ability to perform hierarchical feature learning and focus on local information. For example, the front layers in a CNN might capture simple features like edges and curves on the face, and the intermediate and deeper layers can identify the more subtle nuances in different facial expressions. We used Tensorflow to implement the CNN and trained the model on PACE clusters, with the help of GPUs.

The following image shows the layers used to create the model. The model architecture starts with two sets of Conv2D+MaxPooling2D layers, followed by a Dropout layer, a Flatten layer, a Dense layer, a Dropout layer, another Dense layer, and finally the softmax activation function. 

* Conv2D Layers: apply filters to create feature maps
* MaxPooling Layers: extract the most prominent features from the Conv2D layers while reducing overfitting
* Dropout Layers: randomly deactivate neurons to prevent overfitting
* Flatten Layer: converts the multi-dimensional output from the previous layer to 1-D array for the following Dense layer (which takes in 1-D data)
* Dense (fully-connected) Layer: performs classification based on the features
* Softmax: outputs the predicted probability distribution among different classes 

![cnn_model](../images/cnn_model.png)

### Method 2 - Multinomial Logistic Regression (In Progress)
#### Data Preprocessing
**Principle Component Analysis (PCA)**

For initial testing, we started by limiting our labels to just happy and sad. We select an equal number of happy face and sad face images to start with (1000 each). This prevents the model from being biased towards a result because it was inherently more likely to occur, and not correctly using the information provided during training. 

Next, we combine the images into two arrays of shape (1000, 48, 48), where each array represents (image, pixelX, pixelY). Then, we combine and flatten the arrays into a (2000, 2304) input array of features and a (2000,) array of labels where a happy face corresponds to 0 and a sad face corresponds to 1. After that, we use PCA to attempt to visualize the differences between the happy and sad facial images. 

We pull out the first two principal components, which account for a combined 38.8% of the total variation in all features. When graphed, there isn't much separation between the two labels and clear differences are not defined.

In the scatter plot, happy faces are purple and sad faces are yellow.

![pca_raw](../images/pca_raw.png)

**Explained Variance Ratio:** [0.28706147 0.09976334]

**Total Explained Variance:** 0.3868248061313289

From the visualization, It is clear that using just the two best features does not provide a clear difference between the happy and sad faces. That being said, we attempted PCA on the entire dataset with all seven classes using the best 10 principle components. This resulted in the following values for variance:

**Explained Variance Ratio:** [0.28883101 0.09765254 0.09433207 0.0550939  0.03070064 0.02566376 0.02143845 0.01977265 0.01776848 0.01505705]

**Total Explained Variance:** 0.6663105546417539

It looks like with 10 features, we are at least able to capture the majority of variance. More testing will need to be done to find the optimal number of features needed and whether PCA is able to capture the important features. 


**Histogram of Oriented Gradients (HOG)**

In order to extract more useful features, histogram of oriented gradients was implemented to hopefully identify important facial features. HOG specializes in edge detection as well as orientation of the edges. With this in mind, the hope was that HOG would be able to detect specific facial features such as mouths and eyes as well as their orientation which could be used as an indicator of emotion. An example of HOG can be seen below.

![hog_demo](../images/hog_demo.png)

From the picture on the left, all of the important edges are preserved as well as the orientation of the edges. However, when applied to our test dataset, it seems it does not do as well as seen below.

![hog_happy](../images/hog_happy.png)

It looks as if our images are too low resolution to provide enough information for HOG to process. Different block sizes were tried ranging from (2,2) pixels all the way up to (16,16) with orientations from 4 to 8, with little luck in capturing more information. The outline of the head is preserved, but it is very hard to tell any information about the eyes or mouth. That being said, with a block size of (4,4) pixels with 8 orientations, the number of features has gone down from 48x48=2304 to just 576.


**MTCNN Facial Feature Detection**

![mtcnn_demo](../images/mtcnn_demo.png)

We continued to try to extrapolate the most important features of the image by extracting just the shape of the mouth to feed into the model. We made this decision because that's how alot of us discern a happy face from a sad one in real life. We isolated just the mouths of all faces using MTCNN (Multi-Task Cascaded Convolutional Neural Networks), which can pinpoint the location of certain facial features with very high accuracy. Looking at the images above, it is somewhat clear as to which faces are happy or sad based just off of the mouth.

We used MTCNN to create a new array with just the pixels that correspond to a mouth from the smaller sample set of 1000 happy and 1000 sad faces. We then perform PCA again to attempt to plot the differences between the faces. While there still is no obvious difference, you can see that the happy faces (purple) are a bit more to the left and the sad faces (yellow) are a bit more to the right. This is a small improvement from using the whole face.

![pca_mtcnn](../images/pca_mtcnn.png)

**Explained Variance Ratio:** [0.46207603 0.11718947]

**Total Explained Variance:** 0.5792655024259662

#### ML Algorithm/Model
For this method, we decided to try to implement Multinomial Logistic Regression, specifically using ridge regression to prevent overfitting. As with the data processing, we decided to start by only using two categories (happy and sad) with 1000 images each to see how logistic regression would perform. We attempted to fit a logistic regression model on all 2304 features to see if a discernible difference could be learned without much preprocessing. However, there is severe overfitting as we can see from the train accuracy of 100% and the test accuracy of 64.25%. This is most likely due to the fact that we did not have enough data points to have such a high number of features.

Next, we tried running a multinomial logistic regression model on the entire dataset with all 28,821 images and all 7 classes. As expected, with the large amount of data points and features, the function was unable to converge even after increasing the number of iterations to 10,000. It is possible it would converge after more iterations, but it is clear that data pre-processing would be necessary.

After running PCA on the data and extracting the 10 best features, a logistic regression model was able to be fit with a training accuracy of 27% and a testing accuracy of 26%. Both are much lower than expected. It is possible with more tuning and using more features, a better fit can be found.

The next preprocessing method attempted was using HOG which resulted in 536 features per image. After running all of the images through HOG, a multinomial logistic regression function was able to be fit with a training accuracy of 45% and a testing accuracy of 42%. While this is still not amazing, it performed much better than with PCA. 

The last preprocessing method was to use MTCNN for facial feature detection to isolate the mouth. With this process being much more computationally intensive, we decided to once again try first on a subset of just 1000 happy and 1000 sad images. After producing new images with just the mouths, we did another run of training a logistic regression model. This time, the train accuracy is 79.8%, and the test accuracy improves to 73.3%. There is much less overfitting and an overall improvement in prediction accuracy. This shows great potential as we work to apply this to the entire data set.

## Results and Discussion
### Method 1 - Convolutional Neural Network (CNN)  
#### Visualizations and Quantitative Metrics
![cnn_metrics](../images/cnn_metrics.png)

![cnn_cm](../images/cnn_cm.png)

#### Analysis
The results and visualizations provided several insights into our CNN model’s performance. First, we observe that besides the loss, all validation metrics remain higher than the training metrics throughout the epochs. This may be due to the L2 regularizer we applied in the dense layer or adding the dropout layer. Both methods prevent overfitting by adding penalties only to the training loss. The model performs well in precision and AUROC. This suggests that the model is good at identifying positive cases among the predicted positives and distinguishing between the positive and negative classes across different thresholds. 

On the other hand, the model performs poorly in recall and accuracy, indicating that the overall correctness across all classes is low, and it misses a significant number of actual positive cases. This shows that the model is very conservative in predicting positives, leading to many false negatives. This situation is common in datasets with imbalanced classes, where the model might be biased towards the majority class. 
Also, in the confusion matrix, we can observe that in general, the matrix focuses on the diagonal elements, which represent the True positive values. The model can classify the ‘happy’, ‘neutral’, and ‘surprise’ class well. However, improvements are needed to classify other classes more accurately.

### Method 2 - Multinomial Logistic Regression (In Progress)
#### Visualizations and Quantitative Metrics
**PCA with 10 Features and Multinomial Logistic Regression**

| Training Accuracy | 0.27445265604940844 |
| Testing Accuracy | 0.2629493348429097 |
| Testing Precision | 0.1880962119545255 |
| Testing Recall | 0.17857835013413167 |
| Testing F1 | 0.1521894843233517 |

Confusion Matrix:

![pca_mlr_cm](../images/pca_mlr_cm.png)

**HOG and Multinomial Logistic Regression**

| Training Accuracy | 0.45404392630373686 |
| Testing Accuracy | 0.4167846023209737 |
| Testing Precision | 0.36715607453645527 |
| Testing Recall | 0.3498164155919157 |
| Testing F1 | 0.35243930689142605 |

Confusion Matrix:

![hog_mlr_cm](../images/hog_mlr_cm.png)

#### Analysis
While this model is still under development, we do have some visual/qualitative metrics. It is clear that while both PCA and HOG data preprocessing methods are not perfect, HOG does show significant improvement over PCA. It does seem that for both models, the happy face was the most accurately predicted label with the highest number of true and predicted labels. However, it is clear that in addition to low accuracy, both models had low precision and recall. With more testing and tuning, hopefully all of these metrics can be improved especially when using MTCNN facial feature detection on the entire dataset.

### Next Steps
Currently, we have come up with a baseline CNN model and a baseline multinomial logistic regression model with various data preprocessing techniques. In addition to creating a third model, there is much tuning and learning that can be done using the existing two models.

For the CNN model, we can try using a Keras pre-trained model to implement transfer learning since the pre-trained models have well-established model architectures and will potentially yield better results. We could also change the unbalanced dataset. For example, applying SMOTE or augmentation to upsample the images in minority classes or performing downsampling on majority classes. We can also play around with the different layers that we have chosen to try to come up with a better CNN model.

For the multinomial logistic regression model, it would be a good idea to try applying the MTCNN data preprocessing to the entire dataset to see if this would lead to better results. Additionally, playing around with the parameters of the regression function as well as with the other data preprocessing methods may yield improved results as well. 

Lastly, while we have now experimented with using a CNN for our data preprocessing, we should also continue to look into other methods that are completely free of a CNN to see how traditional machine learning algorithms/techniques compare to deep learning and CNN. While deep learning is more geared towards this application, it would be interesting to see the performance impact for the improvement in accuracy. 



## Gantt Chart
Our Gantt Chart can be found [here](https://github.com/e019chen/ML-Facial-Expression-Recognition/blob/jekyll-integration/GanttChart.xlsx).

![gc_1](../images/gc_1.png)

![gc_2](../images/gc_2.png)

![gc_3](../images/gc_3.png)


## Contribution Table

| Name | Proposal Contributions |
| ---- | ---------------------- |
| Andrew G. | HOG Preprocessing, Multinomial Logistic Regression Model, Report Editing, Github Pages |
| Andrew H. | Data Organizing, Report Documenting |
| Chris | PCA Preprocessing, MTCNN Facial Feature Recognition, Logistic Regression Model |
| Wei-Liang | CNN Model, Image Augmentation |
| Euan | Gantt Chart, Report Documenting |


## References
[1] B. Büdenbender, T. T. A. Höfling, A. B. M. Gerdes, and G. W. Alpers, “Training machine learning algorithms for automatic facial coding: The role of emotional facial expressions’ prototypicality,” PLOS ONE, vol. 18, no. 2, p. e0281309, Feb. 2023, doi: [https://doi.org/10.1371/journal.pone.0281309](https://doi.org/10.1371/journal.pone.0281309)

[2] V. Hosur, “Facial Emotion Detection Using Convolutional Neural Networks,” Available: IEEE Xplore, Oct. 2022, Accessed: Jan. 21, 2024. [Online]. [https://ieeexplore.ieee.org/document/9972510](https://ieeexplore.ieee.org/document/9972510)

[3] A. Mohammad Hemmatiyan-Larki, F. Rafiee-Karkevandi and M. Yazdian-Dehkordi, "Facial Expression Recognition: a Comparison with Different Classical and Deep Learning Methods," 2022 International Conference on Machine Vision and Image Processing (MVIP), Ahvaz, Iran, Islamic Republic of, 2022, pp. 1-5, doi: [https://doi.org/10.1109/MVIP53647.2022.9738553](https://doi.org/10.1109/MVIP53647.2022.9738553)

[4] “Facial Expression Recognition (FER) on AffectNet,” paperswithcode.com. [https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet](https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet)

[5] S. Zhang, H. Teng, Y. Zhang, Y. Wang, and Z. Song, “A Dual-Direction Attention Mixed Feature Network for Facial Expression Recognition,” Electronics, vol. 12, no. 17, pp. 3595–3595, Aug. 2023, doi: [https://doi.org/10.3390/electronics12173595](https://doi.org/10.3390/electronics12173595)

[6] “Facial Expression Recognition (FER) on RAF-DB,” paperswithcode.com. [https://paperswithcode.com/sota/facial-expression-recognition-on-raf-db](https://paperswithcode.com/sota/facial-expression-recognition-on-raf-d)