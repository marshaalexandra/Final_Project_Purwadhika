# Final_Project_Purwadhika
This final project is one of the requirements for graduating from Job Connector Data Science and Machine Learning Purwadhika Start-up and Coding School.

<p align="center"  color="rgb(0, 90, 71)">
<h1>Background</h1>
</p>

<p align='justify' style="font-weight: bold;">
Customer churn refers to when customers (e.g. subscribers, users, players etc.) stop using certain company product or service. Telecommucation companies experiences a wide range of 10-67% annual customer churn rate. This industry is highly competitive due to convenient nature of the service, allowing customers to choose and switch from multiple service providers and other factors such as better price offers, more interesting packages, bad service experiences or change of customers’ personal situations. In this highly competitive market, increasing retention period of customers is a considerable concern within the telecommunication companies.
</p>

<p align="center">
<img src="https://github.com/marshaalexandra/Final_Project_Purwadhika/blob/main/customerchurn.jpeg" height="300" width="200">
</p>

<h1>Goal</h1>
<p align='justify' style="font-weight: bold;">
The customer churn would require costs for lost revenue, marketing costs for attracting new potential customers, and additional costs for replacing the churned customers with the new ones. Therefore, predicting the customers who are likely to stop using the service will represent potentially large additional revenue source if it is done in the early phase. Thus it is an imperative need to create predictive machine learning model that could correctly predict the churned customer. The goals of exploration and machine learning model for this dataset are:

<ol>
<li>To determine factors that accounts to customer churn</li>
  <li>To predict if an individual customer will churn or not</li>
</ol>
</p>


<h1>The Data</h1>
<p align='justify' style="font-weight: bold;">
The dataset that I choose is a <a href="https://www.kaggle.com/blastchar/telco-customer-churn">telco churn dataset</a> from Kaggle. It contains information about a fictional telco company that provided home phone and Internet services to 7043 customers in California in Q3.</p><br><br>
  
--------------------------------------------------------------------
 <p align="center"  color="rgb(0, 90, 71)">
<h1>Step of work</h1>
</p>
<br>

### 1. Exploratory Data Analysis (EDA)
Started with import the dataset which is telco churn dataset. After that, I do EDA to explore which features that are predictive to churn rate prediction based on the feature types (numerical and categorical) using several visualization such as pie chart, countplot, and distribution plot.

### 2. Data Cleaning and Preprocessing
I do data cleaning to correctly assign the data type on some features, drop all rows that contain missing values, and check the outliers. At preprocessing step, I did one hot encoding for 15 categorical features since most of them only contain 2-3 values, standar scalling for all numerical features, and dropped irrelevant feature. 

### 3. Modelling
I choose recall for metric evaluation score for the subsequent modelling since I want to minimize the false negative error. I cross-validated 5 models (Logistic Regression, KNN, Decision Tree, Random Forest, and Extreme Gradient Boosting) without and with imbalance dataset handling (SMOTE, NearMiss, and algorithm-based methods) to see which models give the highest cross validation score. The model that gave the best performance score and lowest standard deviation score is Random Forest with resampling Near Miss method with recall score 0.93.

<p align="center"> <img src="https://github.com/marshaalexandra/Final_Project_Purwadhika/blob/main/cross%20validation%20model.png" alt="" width="700" height="275"> </p><br>
```
CHOSEN MODEL:
RandomForestClassifier(max_depth = 3, n_estimators = 20, max_features = 4) with NearMiss()
```
<br>
<br>

### 4. Hyperparameter Tuning

I tuned hyperparameter the model using GridSearchCV. The results from hyperparameter tuning shows that the model after hyperparameter tuning have slightly higher recall score (from 0.92 to 0.93). The following below are the parameters that I tuned, classification report after hyperparameter tuning, and best parameters from Random Forest with Near Miss method:

###### *paramaters*

```
hyperparam_space = {
    'balancing__n_neighbors': [2,5,10,15,20],
    'model__n_estimators' : [10,20,100,200],
    'model__max_depth' : np.arange(3,13,2),
    'model__max_features' : ['auto',2,4,6,8],
    'model__criterion': ['gini','entropy'],
    'model__bootstrap': [True,False]
}
}
```

###### *evaluation*

```
             ===== CLASSIFICATION REPORT =====
                precision    recall  f1-score   support

           0       0.91      0.26      0.41      1549
           1       0.31      0.93      0.47       561

    accuracy                           0.44      2110
   macro avg       0.61      0.60      0.44      2110
weighted avg       0.75      0.44      0.42      2110

tn :  407  fp :  1142  fn :  39  tp :  522

======== BEST PARAMETERS SCORING FROM RECALL SCORE ========
{'balancing__n_neighbors': 15, 'model__bootstrap': False, 'model__criterion': 'entropy', 'model__max_depth': 3, 'model__max_features': 'auto', 'model__n_estimators': 10}
```

### 5. Feature Importance and Model Fitting with Feature Selection Dataset
I run feature importance to check which features matter the most in predicting churn rate in the model. Based on the feature importances, I choose 10 features that have the highest value and create new dataset containing only those 10 features. Then, I fit the model with the feature selection dataset.  

### 6. Performance Evaluation
The model gave the same recall score with the feature selection dataset. Therefore I chose the model and the feature selection dataset to deploy them to dashboard. The following below is the performance evaluation result (classification report and confusion matrix) when the model is fitted with the new dataset.

```
         ======== CLASSIFICATION REPORT  ========
              precision    recall  f1-score   support

           0       0.91      0.26      0.40      1549
           1       0.31      0.93      0.47       561

    accuracy                           0.44      2110
   macro avg       0.61      0.59      0.43      2110
weighted avg       0.75      0.44      0.42      2110

tn :  396  fp :  1153  fn :  38  tp :  523
```


<br>

--------------------------------------------------------------------
