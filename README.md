# Email-Spam-Detection-and-Filtering

## Project Description
A spam filter is a program used to detect unsolicited, unwanted and virus-infected emails and prevent those messages from getting to a user's inbox. Here we have built an email spam detection filter using machine learning in python by comparing the accuracy and performance of multiple machine learning algorithms for the same.

![alt text](https://github.com/omaarelsherif/Email-Spam-Detection-Using-Machine-Learning/raw/main/Images/Email_Spam_Detection_Header.png)

## Prerequisites
This is the list of required packages and modules for the given project:
* Python 3.x
* Numpy
* Pandas
* Scikit-learn

Install above packages and modules using:

`pip install -r requirements.txt`

## Workflow

![image](https://user-images.githubusercontent.com/60508605/162937966-a6c0da27-23bb-4201-9d62-aebed585964d.png)


### Data PreProcessing
Data preprocessing refers to manipulation or dropping of data before it is used in order to ensure enhanced performance. It involves the following:
* Loading the dataset
* Replace missing values in the dataset with null string
* Label encoding i.e. labelling spam and ham emails as 0 and 1
* Separating data as text (message) and labels (category)

### Splitting the dataset
The dataset is split into train and test sets in the ratio 80:20. 

### Feature Extraction
Text data is transformed into feature vectors that can be used as input for the model. Y_train and Y_test values are converted to integers. 

### Model Creation
Two models are used - namely, Logistic Regression and Naive Byaes Theorem, and their performance is compared.

### Analysis
![image](https://user-images.githubusercontent.com/60508605/162611803-1d7e46f4-584c-45db-934f-cf85bfc8f56e.png)

### Result
The Logistic Regression model gives an accuracy of 0.9659 on the test set while the Niave Bayes theorem gives an accuracy of 0.9892. 

## Installation 
* Clone the repo 
  `https://github.com/NVombat/Email-Spam-Detection-and-Filtering.git`
* Run the following command on cmd
  `python spam_predictor.py`

## References
* Spam detection in machine learning: https://towardsdatascience.com/email-spam-detection-1-2-b0e06a5c0472
* Logistic Regression: https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc
* Naive Bayes Theorem: https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/

## Contributors
* Nikhill Vombatkere
* Annanya Pandey
