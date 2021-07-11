---
title:  "Building Machine Learning Apps Faster With dstack.ai"
date:   2021-01-01 01:29:17 +0545
categories:
  - Data Science
  - Programming
tags:
  - data science
  - dstack
  - python 
  - web development
header:
  teaser: https://repository-images.githubusercontent.com/274089435/bf85af80-bc4b-11ea-95ba-a636ebf52ceb
---
# Building Machine Learning Apps Faster With dstack.ai
Happy New Year everyone!

This is the second part of the blog I am writing while exploring `dstack`. In this part, I will focus on an exciting feature offered by dstack which allows you to push and pull ML models from your command line interface, and share it with colleagues or other trusted people using permission management features. 

If you are new to dstack, then you can read my previous blog where I introduce dstack by building a simple data app. [blog]({{site.url}}/2020/12/26/exploring-dstack-for-building-data-apps-faster/) or [the documentation](https://docs.dstack.ai/concepts/ml-models).

You find the following when you go to dstack’s documentation:
>`dstack` decouples the development of applications from the development of ML models by offering an ML registry. This way, one can develop ML models, push them to the registry, and then later pull these models from applications.

In the first part of the blog, we pushed a visualization of Titanic Survival Dataset using dstack. In this part, I will train 3 Classifiers to classify the survival of the person and push this model to dstack. Later I will retrieve the model to my terminal using the pull function from dstack. 

## Project Structure
I will be using two scripts to demonstrate the dstack ML model registry feature - one for pushing the ML Model and another for Pulling ML Model. 
* Root File
    * Data
        * titanic_data.csv
    * titanic_push.py
    * titanic_pull.py


## File `titanic_push.py`
As usual, we start by importing dependencies. In this same file, we will be training 3 classifier models and push them to our Model Registry. 
* Decision Tree
* Random Forest
* Gradient Boosting

I am following [this](https://humansofdata.atlan.com/2016/07/machine-learning-python/) blog for training a Model.

```python
import dstack.controls as ctrl
import dstack as ds
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn import datasets, svm, tree, preprocessing, metrics
import sklearn.ensemble as ske
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score

@ds.cache()
def get_data():
    filename = "F:/Desktop/learning/dstack/blog/data/titanic_data.csv"
    return pd.read_csv(filename)

df = get_data()
df = df.drop(['Cabin'], axis=1)
df = df.dropna()

def preprocess_df(df):
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df.Sex = le.fit_transform(processed_df.Sex)
    processed_df.Embarked = le.fit_transform(processed_df.Embarked)
    processed_df = processed_df.drop(['Name','Ticket'],axis=1)
    return processed_df

processed_df = preprocess_df(df)
X = processed_df.drop(['Survived'], axis=1).values
y = processed_df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

clf_dt = tree.DecisionTreeClassifier(max_depth=10)
clf_dt.fit (X_train, y_train)
# clf_dt.score (X_test, y_test)
url = ds.push("titanic/decision_tree", clf_dt)
print("Decision tree ", url)

shuffle_validator = ShuffleSplit(len(X), test_size=0.2, random_state=0)
def test_classifier(clf):
    scores = cross_val_score(clf, X, y, cv=shuffle_validator)
    return  scores.mean()
print(f"Decision Tree Acc: {test_classifier(clf_dt)}\n")

clf_rf = ske.RandomForestClassifier(n_estimators=50)
clf_rf.fit (X_train, y_train)
url = ds.push("titanic/random_forest", clf_rf)
print("Random Forest ", url)
print(f"Random Forest Acc: {test_classifier(clf_rf)}\n")

clf_gb = ske.GradientBoostingClassifier(n_estimators=50)
clf_gb.fit (X_train, y_train)
url = ds.push("titanic/gradient_boosting", clf_gb)
print("Gradient Boosting ", url)
print(f"Gradient Boosting Acc: {test_classifier(clf_gb)}\n")
```

* We start by importing dependencies.
* Make a method to read CSV a file from local storage. Cache that method because we might call that method frequently.
* Drop NULL data, and some non-numeric column names like Cabin, Name, Ticket.
* Preprocess our data a little bit to make it trainable.
* Perform split of data into training set and testing set.
* Train a Decision Tree and push it to `titanic/decision_tree` and print its URL.
* Train a Random Forest and push it to `titanic/random_forest` and print its URL.
* Train a Gradient Boosting and push it to `titanic/gradient_boosting` and print its URL.

You should be able to see the following in your terminal.
![img]({{site.url}}/assets/dstack_blog/URLS.PNG)

Once you click on the URL whicl leads you to ML Models tab on the left navigation panel, we can see the following:
![img]({{site.url}}/assets/dstack_blog/ml_models.png)

Note that you can see more than one model as I already pushed some models into the registry prior to writing this blog.

If you open the model `titanic/gradient_boosting`, then you should be able to see your model as below.
![img]({{site.url}}/assets/dstack_blog/gb_model.png)

`dstack` offers a simple documentation feature by allowing to add a readme file on the pushed model. Here we can write about the performance of our model or the use case of our models. I find this feature very useful because I can write about the property of my model in plain text.

## File `titanic_pull.py`
Now that I have a model in the registry, I can pull this model from a python file instead of pulling it from a remote area.

```python
import dstack.controls as ctrl
import dstack as ds
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn import datasets, svm, tree, preprocessing, metrics
import sklearn.ensemble as ske
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score

@ds.cache()
def get_data():
    filename = "F:/Desktop/learning/dstack/blog/data/titanic_data.csv"
    return pd.read_csv(filename)

df = get_data()
titanic_df=df.copy()
titanic_df = titanic_df.drop(['Cabin'], axis=1)
titanic_df = titanic_df.dropna()

def preprocess_titanic_df(df):
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df.Sex = le.fit_transform(processed_df.Sex)
    processed_df.Embarked = le.fit_transform(processed_df.Embarked)
    processed_df = processed_df.drop(['Name','Ticket'],axis=1)
    return processed_df

processed_df = preprocess_titanic_df(titanic_df)
X = processed_df.drop(['Survived'], axis=1).values
y = processed_df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

def get_decision_tree():
    return ds.pull("titanic/decision_tree")
def get_random_forest():
    return ds.pull("titanic/random_forest")
def get_gradient_boosting():
    return ds.pull("titanic/gradient_boosting")

def dt_pred():
    dt = get_decision_tree()
    p = dt.predict(X)
    pdf = processed_df.copy()
    pdf["DT Pred"] = p
    return pdf

def rf_pred():
    rf = get_random_forest()
    p = rf.predict(X)
    pdf = processed_df.copy()
    pdf["RF Pred"] = p
    return pdf

def gb_pred():
    gb = get_gradient_boosting()
    p = gb.predict(X)
    pdf = processed_df.copy()
    pdf["GB Pred"] = p
    return pdf

dt_app = ds.app(dt_pred)
rf_app = ds.app(rf_pred)
gb_app = ds.app(gb_pred)

url = ds.push("titanic/dt_pred", dt_app)
print(f"Decision Tree: {url}\n")

url = ds.push("titanic/rf_pred", rf_app)
print(f"Random Forest: {url}\n")

url = ds.push("titanic/gb_pred", gb_app)
print(f"Gradient Boosting: {url}\n")
```


**What is happening above?**
* Same as `pushing` code, our `pulling` code starts by importing dependencies.
* Read the data from local storage and preprocess it because we will be using this dataset to find out the prediction of our models.
* Make a function to pull each model and return it.
* Make a function to do prediction using the pulled function and then stacking that prediction to a new column of the data frame and return that data frame.
* Make a data app for each of these applications(decision tree, random forest, and gradient boosting).
* Push each application and print its URL.

If everything is right, we can have URLs for each application. You can find the application `titanic/dt_pred` in the dstack UI. 
![img]({{site.url}}/assets/dstack_blog/dt_pred_pull.png)

## Finally
I first trained my model, then pushed/pulled some simple classifiers for Titanic Survival Dataset, and  stacked their prediction to the new column. If we want to share this model with someone, then we simply can go to share and choose whether we want it to be public or not. 

In the next part, I will be writing about training a Tensorflow model and then reusing it. Also, I have not figured out all the cool UI tools that dstack provides, so in the next part, I will try to use them and make a more cool project.

If you reached this line then please leave some comments so that I can improve myself. Also if you have any queries then ping me on LinkedIn as [Ramkrishna Acharya](https://www.linkedin.com/in/qramkrishna).

### Why not read more?
* [Gesture Based Visually Writing System Using OpenCV and Python]({{site.url}}/2020/08/01/gesture-based-visually-writing-system-using-opencv-and-python/)
* [Gesture Based Visually Writing System: Adding Visual User Interface]({{site.url}}/2020/08/11/gesture-based-visually-writing-system-make-a-visual-user-interface/)
* [Gesture Based Visually Writing System: Adding Virtual Animationn, New Mode and New VUI]({{site.url}}/2020/08/14/gesture-based-visually-writing-system-adding-virtual-animation-new-mode-and-new-vui/)
* [Gesture Based Visually Writing System: Add Slider, More Colors and Optimized OOP code]({{site.url}}/2020/08/21/gesture-based-visually-writing-system-add-slider-more-colors-and-optimized-code/)
* [Gesture Based Visually Writing System: A Web App]({{site.url}}/2020/08/29/gesture-based-visually-writing-system-web-app/)
* [Contour Based Game: Break The Bricks]({{site.url}}/2020/08/16/contour-based-game-break-the-bricks/)
* [Linear Regression from Scratch]({{site.url}}/2020/08/07/writing-a-linear-regression-class-from-scratch-using-python/)
* [Writing Popular ML Optimizers from Scratch]({{site.url}}/2020/06/05/writing-popular-machine-learning-optimizers-from-scratch-on-python/)
* [Feed Forward Neural Network from Scratch]({{site.url}}/2020/05/30/writing-a-deep-neural-network-from-scratch-on-python/)
* [Convolutional Neural Networks from Scratch]({{site.url}}/2020/06/05/convolutional-neural-networks-from-scratch-on-python/)
* [Writing a Simple Image Processing Class from Scratch]({{site.url}}/2020/05/30/image-processing-class-from-scratch-on-python/)
* [Deploying a RASA Chatbot on Android using Unity3d]({{site.url}}/2020/08/04/deploying-a-simple-rasa-chatbot-on-unity3d-project-to-make-a-chatbot-for-android-devices/)
* [Naive Bayes for text classifications: Scratch to Framework]({{site.url}}/2020/03/04/text-classification-using-naive-bayes-scratch-to-the-framework/)
* [Simple OCR for Devanagari Handwritten Text]({{site.url}}/2020/02/25/building-ocr-for-devanagari-handwritten-character/)



