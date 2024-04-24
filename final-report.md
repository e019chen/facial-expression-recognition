---
layout: page
title: Final Report
permalink: /final-report/
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
### Method 1 - Binary Classification with Logistic Regression and Support Vector Machine
To determine if our facial expressions dataset is learnable, we started with only studying the “happy” and “sad” faces with a simple binary classifier. From our base results, we performed data processing to reduce the number of features and cross validation to select the optimal binary classifier to improve our test accuracy and strike a balance between overfitting and underfitting the data. Our analysis with binary classification provided a baseline model upon which we can improve with more sophisticated models.

#### Data Preprocessing
Although our original dataset contained seven labels with thousands of images, we simplified our binary classification model by using only 1000 images for each the “happy” and “sad” labels. By using an equal number of happy and sad faces, we will avoid any potential bias in the model caused by an unequal distribution of class labels. 

To visualize any differences between the two classes, we performed PCA to reduce the dimensions of the dataset from a flattened 2304 dimensions to only 2. This accounts for only 38.8% of the variance of the original dataset, but it is necessary to be able to visualize any differences. However, the graph below shows indiscernible separation.

![pca_raw](../images/pca_raw.png)

**First and Second Principle Components of the Entire Face.** Note that the happy faces are purple and the sad faces are yellow.

By relying on our intuition that a person’s mouth gives away most of the information regarding a person’s emotion, we used MTCNN (Multi-Task Cascaded Convolutional Neural Networks) to isolate just the mouths of all faces in order to continue to reduce the number of dimensions and noise of the dataset. MTCNN uses a Convolutional Neural Network to pinpoint the specific location of certain facial features with very high accuracy, including the mouth and eyes. When we plot just the mouths of a handful of faces from our dataset, you may be able to classify the emotion of the face just from the mouth alone.

![mtcnn_demo](../images/mtcnn_demo.png)

When we perform PCA to reduce the number of features down to two again, the graph of the first two principal components, there is a slight improvement in separation. While there still is no obvious difference, you can see that the happy faces (purple) are a bit more to the left and the sad faces (yellow) are a bit more to the right. Now, 57.9% of the variance is captured from just the first two principal components.

![pca_mtcnn](../images/pca_mtcnn.png)

**First and Second Principal Components of Just the Mouths.** Still,the happy faces are purple and the sad faces are yellow.

#### ML Algorithms/Models

To start with our baseline model, we trained a binary logistic regression model on the entire face to predict the labels “happy” or “sad”, treating each pixel as its individual feature. However, due to low test accuracy and major overfitting, we used PCA to reduce the dimensions of the dataset before training another binary logistic regression model. Since the number of dimensions of the dataset is now a hyperparameter, we performed 5-fold cross validation to determine the optimal binary logistic regression model that maximizes the test accuracy while minimizing overfitting. 

To further improve the test accuracy of our model, we trained a binary logistic regression and support vector machine model on just the mouths of all faces to differentiate the “happy” faces from the “sad” faces. Overfitting is significantly reduced, and an increase in test accuracy is visible. To further tune our support vector machine model, we performed cross-validation to determine the optimal kernel to use to handle non-linearity in our dataset. Both the polynomial and radial kernel were used, and we used 5-fold cross validation to select the optimal degree for the polynomial kernel. 


### Method 2 - Multinomial Logistic Regression with Histogram of Oriented Gradients and PCA
#### Data Preprocessing

Having trained a binary classifier with just happy and sad faces, we were ready to take on the challenge of full multiclass classification of the seven different emotions. An initial attempt was made at using multinomial logistic regression to see if there would be a good fit, but unsurprisingly, the model was unable to converge even after a very large number of iterations. This is when we started looking into feature extraction and data preprocessing to be able to limit the data necessary to classify, while taking the most important features.

We then attempted PCA on the entire dataset with all seven classes using the best 10 principle components. This resulted in the following values for variance:

**Explained Variance Ratio**: [0.28883101 0.09765254 0.09433207 0.0550939  0.03070064 0.02566376 0.02143845 0.01977265 0.01776848 0.01505705]

**Total Explained Variance**: 0.6663105546417539

It looks like with 10 features, we are at least able to capture the majority of variance. More testing will need to be done to find the optimal number of features needed and whether PCA is able to capture the important features. That being said, it became clear we would need a data preprocessor that is more geared towards images. 

In order to extract more useful features, histogram of oriented gradients was implemented to hopefully identify important facial features. HOG specializes in edge detection as well as orientation of the edges. With this in mind, the hope was that HOG would be able to detect specific facial features such as mouths and eyes as well as their orientation which could be used as an indicator of emotion. An example of HOG can be seen below.

![hog_demo](../images/hog_demo.png)

From the picture on the left, all of the important edges are preserved as well as the orientation of the edges. However, when applied to our test dataset, it seems it does not do as well as seen below.

![hog_happy](../images/hog_happy.png)

It looks as if our images are too low resolution to provide enough information for HOG to process. Different block sizes were tried ranging from (2,2) pixels all the way up to (16,16) with orientations from 4 to 8, with little luck in capturing more information. The outline of the head is preserved, but it is very hard to tell any information about the eyes or mouth. That being said, with a block size of (4,4) pixels with 8 orientations, the number of features has gone down from 48x48=2304 to just 576.


#### ML Algorithms/Models
For this method we decided to try multinomial Logistic Regression, specifically using ridge regression to prevent overfitting, we tried running a multinomial logistic regression model on the entire dataset with all 28,821 images and all 7 classes we decided to try to implement. As expected, with the large amount of data points and features, the function was unable to converge even after increasing the number of iterations to 10,000. It is possible it would converge after more iterations, but it is clear that data pre-processing would be necessary.

After running PCA on the data and extracting the 10 best features, a logistic regression model was able to be fit with a training accuracy of 27% and a testing accuracy of 26%. Both are much lower than expected. It is possible with more tuning and using more features, a better fit can be found.

The next preprocessing method attempted was using HOG which resulted in 536 features per image. After running all of the images through HOG, a multinomial logistic regression function was able to be fit with a training accuracy of 45% and a testing accuracy of 42%. While this is still not amazing, it performed much better than with PCA. 

While we had wanted to also utilize MTCNN as a precursor to multinomial logistic regression on the entire dataset, the limited computation power available to us rendered that unrealistic.


### Method 3 - Convolutional Neural Network (CNN)  
#### Data Preprocessing
The original facial expression dataset from Kaggle contains only a train and a test folder. To validate our model and get potential ways to adjust the hyperparameters, we select a portion of the original training dataset as the validation dataset. We now have a train: validate: test ratio of roughly 70%:15%:15%. 

Before feeding the images in the CNN model, we utilize Tensorflow’s ImageDataGenerator to apply random shifting, shearing, zooming, and flipping to help the model generalize any variations in the data. Additionally, we normalize the images so the pixel intensities are between 0 and 1 instead of 0 and 255. This helps the gradient calculations stay consistent and reduces the training time. 

Lastly, we dropped the class “disgust” from our dataset since it had a significantly smaller number of images compared to other classes. This will help us prevent the risks of imbalanced data, such as the model becoming biased towards majority classes. 

#### ML Algorithm/Model
The last ML Algorithm our team implemented is the Convolutional Neural Network (CNN) model. At a high level, CNN works by applying kernels (or filters) in convolutional layers to create feature maps. These feature maps highlight important attributes for the classification tasks. CNNs have become the to-go model for many image classification tasks, including facial expression recognition (FER) due to its effectiveness. 

One reason behind this is the ability to perform hierarchical feature learning and focus on local information. For example, the front layers in a CNN might capture simple features like edges and curves on the face, and the intermediate and deeper layers can identify the more subtle nuances in different facial expressions. We used Tensorflow to implement the CNN and trained the model on PACE clusters, with the help of GPUs.

The following image shows the layers used to create the model. The model architecture starts with two sets of Conv2D+MaxPooling2D layers, followed by a Dropout layer, a Flatten layer, a Dense layer, a Dropout layer, another Dense layer, and finally the softmax activation function. 

* Conv2D Layers: apply filters to create feature maps
* MaxPooling Layers: extract the most prominent features from the Conv2D layers while reducing overfitting
* Dropout Layers: randomly deactivate neurons to prevent overfitting
* Flatten Layer: converts the multi-dimensional output from the previous layer to 1-D array for the following Dense layer (which takes in 1-D data)
* Dense (fully-connected) Layer: performs classification based on the features
* Softmax: outputs the predicted probability distribution among different classes 

![cnn_model](../images/cnn_model.png)

In addition to the CNN model we created ourselves in our midterm checkpoint, we built on it with the integration of transfer learning. Transfer Learning is a technique used in deep learning to reuse a pre-trained model on a new problem. It is useful because we can take advantage of a model that’s been trained on millions of labeled data points by world class researchers for our problem using a lot less data and expertise. In our case, we chose to integrate the ResNet-50 Model. It stands for Residual Network and was introduced in the 2015 paper “Deep Residual Learning for Image Recognition”. It's a 50-layer CNN trained on the ImageNet database that uses shortcut connection to skip some layers and achieve effective classification. 

![transfer_learning](../images/transfer_learning.png)

With the addition of some layers, we adapted the model for over 1000 labels for our problem with just 7 labels. 

![transfer_layers](../images/transfer_layers.png)



## Results and Discussion
### Method 1 - Binary Classification Models
Our initial binary logistic regression model that trained on the entire face severely overfit the data, with a train accuracy of 100% and a test accuracy of only 63.75%. This is expected, since we do not have enough data points (2000) to have such a high number of features (2304). From looking at the Confusion Matrix, the model does a slightly better job predicting sad faces than predicting happy faces.

![cm_initial_log_reg](../images/cm_initial_log_reg.png)

When we performed 5-fold cross validation to determine the optimal number of dimensions to reduce to using Principal Component Analysis, we concluded that 70 dimensions produces the highest test accuracy and minimizes the overfitting that occurs when the number of dimensions becomes too high.

![cross_val_num_components](../images/cross_val_num_components.png)

The train accuracy with 70 dimensions is 72.8%, while the test accuracy is 70.6%. This is already a major improvement than our naive initial logistic regression model, which had a test accuracy of only 63.75%.

When we trained a binary logistic regression model with only the mouths of each face, the train accuracy dropped to 79.84%, and the test accuracy improved to 72.99%. There is much less overfitting and an overall improvement in prediction accuracy. From looking at the Confusion Matrix, the model does a better job than the initial Logistic Regression model, and a similar job predicting between happy and sad mouths.

![cm_mouth_log_reg](../images/cm_mouth_log_reg.png)

Our results with a trained Support Vector Machine were more promising. We were able to obtain a test accuracy of 77.17% using a SVM with a polynomial kernel with degree 1, and 77.49% using a SVM with a Radial Basis Function (RBF) kernel. When we performed 5-fold cross validation to determine the optimal degree for the polynomial kernel, it was obvious that degree 1 was the best choice, since all other degrees overfit the data without improving the test accuracy.

![cross_val_degree](../images/cross_val_degree.png)

![cm_svm_polynomial](../images/cm_svm_polynomial.png)

![cm_svm_rbf](../images/cm_svm_rbf.png)

To summarize, our binary support vector machine and binary logistic regression models trained on only the mouths performed much better than our naive binary logistic regression model trained on the entire face. By reducing the noise, removing the background, and focusing on only the important facial features, we were able to improve our test accuracy and reduce overfitting.

Comparison of Binary Classification Models. This table is sorted in increasing Test Accuracy.

| Binary Classification Model | Train Accuracy | Test Accuracy |
| ------- | --------------------- | -------- |
| Whole Face Logistic Regression | 100% | 63.75% |
| Whole Face Logistic Regression Reduced to 70 Dimensions    | 72.83% | 70.60% |
| Mouth Only Logistic Regression  | 79.83%  | 72.99% |
| Mouth Only SVM with Polynomial Kernel of Degree 1  | 77.02%  | 77.17% |
| Mouth Only SVM with RBF Kernel   | 86.85% | 77.49% |


### Method 2 - Multinomial Logistic Regression with Histogram of Oriented Gradients and PCA

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
It is clear that while both PCA and HOG data preprocessing methods are not perfect, HOG does show significant improvement over PCA. It does seem that for both models, the happy face was the most accurately predicted label with the highest number of true and predicted labels. However, it is clear that in addition to low accuracy, both models had low precision and recall. With more testing and tuning, hopefully all of these metrics can be improved especially when using MTCNN facial feature detection on the entire dataset.

























### Method 3 - Convolutional Neural Network (CNN)  
#### Visualizations and Quantitative Metrics (CNN)
![cnn_metrics](../images/cnn_metrics.png)

![cnn_cm](../images/cnn_cm.png)

#### Visualizations and Quantitative Metrics (Transfer Learning)
![transfer_images](../images/transfer_images.png)

#### Analysis
The results and visualizations provided several insights into our CNN model’s performance. First, we observe that besides the loss, all validation metrics remain higher than the training metrics throughout the epochs. This may be due to the L2 regularizer we applied in the dense layer or adding the dropout layer. Both methods prevent overfitting by adding penalties only to the training loss. The model performs well in precision and AUROC. This suggests that the model is good at identifying positive cases among the predicted positives and distinguishing between the positive and negative classes across different thresholds.

On the other hand, the model performs poorly in recall and accuracy, indicating that the overall correctness across all classes is low, and it misses a significant number of actual positive cases. This shows that the model is very conservative in predicting positives, leading to many false negatives. This situation is common in datasets with imbalanced classes, where the model might be biased towards the majority class. 
Also, in the confusion matrix, we can observe that in general, the matrix focuses on the diagonal elements, which represent the True positive values. The model can classify the ‘happy’, ‘neutral’, and ‘surprise’ class well. However, improvements are needed to classify other classes more accurately.

The transfer learning model performed slightly better than our model. Depending on the dataset, it achieves up to 15% better accuracy and double the recall. This shows it is slightly better at identifying correct classes. Similar to our model, the validation dataset performed slightly better, although to a smaller degree, probably due to the L2 regularizer too. However, it is still limited by the same problems of a small dataset.


### Conclusion and Evaluation

Our initial model built to discern “happy” and “sad” yielded promising results. Using MTCNN to extract the mouth and training on a small but balanced dataset, the logistic regression model had a 73.0% testing accuracy and SVM model had an even better result with 77.5% accuracy. After testing the waters with just two facial expressions, we were assured this was a feasible assignment and expanded our future models to discern all 7 other expressions. However, due to an unbalanced dataset, less data points, and the increased difficulty to discern more than three times the number of facial expressions, our future models did not perform as well. Using PCA with the 10 most significant features and multinomial regression yielded us 26.3% accuracy, 18.9% precision, 17.8% recall, and 15.2% F1 score. HOG and multinomial regression gave us better results, with a 41.7% testing accuracy, 36.7% precision, 35.0% recall, and 35.2% F1 Score. And as we predicted from our background research, CNN outperformed all of our other models. Using our own model, we had an accuracy of 44.0%, precision of 74.7%, recall of 17.9%, and F1 Score of 28.8%. Integrating Resnet-50 yielded the best results, with 59.6% accuracy, 74.2% precision, 41.8% recall, and 54.5% F1 Score.  

Regardless, there is still room for improvement for each model. Although transfer learning improved our CNN model, its accuracy could be further improved if we implemented freeze layers on the first few layers of the pre-trained model. This is because the first few layers of feature extraction are general and keeping the parameters trained on millions of images while only updating the later layers of the network may be the key to helping us find a balance between leveraging the advantages of transfer learning and specializing the model to solve our problem. We can also try to experiment with other pretrained models that may be better fit for our problem. Another potential improvement is we could change the unbalanced dataset. For example, applying SMOTE or augmentation to upsample the images in minority classes or performing downsampling on majority classes. We can also play around with the different layers that we have chosen to try to come up with a better CNN model.

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
| Chris | PCA Preprocessing, MTCNN Facial Feature Recognition, Binary Classification (Logistic Regression, SVM, Cross Validation) |
| Wei-Liang | CNN Model, Image Augmentation |
| Euan | Gantt Chart, Report Documenting, Transfer Learning Model |


## References
[1] B. Büdenbender, T. T. A. Höfling, A. B. M. Gerdes, and G. W. Alpers, “Training machine learning algorithms for automatic facial coding: The role of emotional facial expressions’ prototypicality,” PLOS ONE, vol. 18, no. 2, p. e0281309, Feb. 2023, doi: [https://doi.org/10.1371/journal.pone.0281309](https://doi.org/10.1371/journal.pone.0281309)

[2] V. Hosur, “Facial Emotion Detection Using Convolutional Neural Networks,” Available: IEEE Xplore, Oct. 2022, Accessed: Jan. 21, 2024. [Online]. [https://ieeexplore.ieee.org/document/9972510](https://ieeexplore.ieee.org/document/9972510)

[3] A. Mohammad Hemmatiyan-Larki, F. Rafiee-Karkevandi and M. Yazdian-Dehkordi, "Facial Expression Recognition: a Comparison with Different Classical and Deep Learning Methods," 2022 International Conference on Machine Vision and Image Processing (MVIP), Ahvaz, Iran, Islamic Republic of, 2022, pp. 1-5, doi: [https://doi.org/10.1109/MVIP53647.2022.9738553](https://doi.org/10.1109/MVIP53647.2022.9738553)

[4] “Facial Expression Recognition (FER) on AffectNet,” paperswithcode.com. [https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet](https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet)

[5] S. Zhang, H. Teng, Y. Zhang, Y. Wang, and Z. Song, “A Dual-Direction Attention Mixed Feature Network for Facial Expression Recognition,” Electronics, vol. 12, no. 17, pp. 3595–3595, Aug. 2023, doi: [https://doi.org/10.3390/electronics12173595](https://doi.org/10.3390/electronics12173595)

[6] “Facial Expression Recognition (FER) on RAF-DB,” paperswithcode.com. [https://paperswithcode.com/sota/facial-expression-recognition-on-raf-db](https://paperswithcode.com/sota/facial-expression-recognition-on-raf-d)
