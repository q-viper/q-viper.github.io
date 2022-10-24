---
title:  Intro to CNN with Keras
date:   2022-09-18 01:29:17 +0545
categories:
    - Python
    - CNN
tags:
    - CNN
    - keras
    - image
header:
  teaser: assets/cnn/2dconv.gif
---

CNN with Keras is easiest way to create CNN (Convolutional Neural Networks. But Why? Images I used in this blog are not owned by me and the full credit goes to the author of images. This blog was written in 2019 but I forgot to publish it that time and does not contain everything.

## Why keras?
* Because its like drag and drop for Neural Networks.
* No more headache of <b>mathematical complexity</b>
* Can customize layers and do training easily.
* Has great preprocessing APIs as well as data generators.


## What is CNN
The answer to this question has been already given by me in following blogs:
* [Writing Image Processing Codes from Scratch in Python](): Where I have written some codes to do read, write, convolve images.
* [Convolutional Neural Networks from Scratch in Python](): Made somewhat like keras class. This blog is first page of google search with query `cnn from scratch`.

In simple words, CNN is mainly a image processing neural networks but can be used for other purposes also.

## Why CNN?
* CNN has filters or kernel as learning parameters. Unlike simple neural nets, CNN uses very few parameters. In fact, parameters are shared!
* CNN handles feature extracting from images very efficiently.

## Basic Terms in CNN
* <b>No. channels</b>: ex RGB, Grayscale
* <b>Filters</b>: each filter gives a filtered image ie. feature map
* <b>Size of filters</b>: it is a square matrix which shape is less than image
* <b>Stride</b>: value with which we will skip the convolution
* <b>Padding</b>: concept of working with edges


Lets see the visualization:
![]({{site.url}}/assets/cnn/simple.png)


## What does normal CNN include?
Normal CNN includes below layers:

1. One input layer
2. One output layer
3. One or more convolutional layer
4. One or more Max-pooling layer
5. Some dropout layer
6. One flatten layer
7. One or more dense layer
8. Some normalizing layers

* <b>Convolutional layer</b> is the layer where convolution operation happens. Convolution here is same as on image processing where features are extracted. A filter of same row and column or square size is taken and multiplied across the window that fits filter. The element-wise product is done and summed all. We generally use stride, as how much pixel shift after doing one convolution. Also zero padding is done sometime to add zeros. The convolution layer gives number of filters with same properties. Here more the number of filter, more the accurate model can get but computational complexity increases.

> Each CNN layer manipulates image using kernels ex.
![]({{site.url}}/assets/cnn/conv out.png)

> What actually happens inside CNN layer?
![]({{site.url}}/assets/cnn/2dconv.gif)
<center> Inside Grayscale image </center>

![]({{site.url}}/assets/cnn/3dconv.gif)
![]({{site.url}}/assets/cnn/sum.gif)
![]({{site.url}}/assets/cnn/bias.gif)

<center> Inside RGB image </center>
![]({{site.url}}/assets/cnn/padding.gif)
<center> Padding of pixels</center>

* <b>Max-Pooling layer</b> is a layer where we take only few pixels from previous layers. We must provide a pool size and then that pool size is used on input pixels. The pool window is moved over entire input and max value within the overlaped input is taken. For example, pool of size (2, 2) give half of input data.

![]({{site.url}}/assets/cnn/stride.gif)
<center> Maxpooling</center>

![]({{site.url}}/assets/cnn/maxpool.gif)
<center> Summing up</center>


* <b> Dropout</b> is a layer which is actually used to avoid overfitting. This layer randomly cuts the connection between two neurons of different layers. For example, dropout of value 0.5 will cut the half of input's connection. Due to this effect, a NN couldn't memorize input sequence.
* <b>Flatten layer</b> is one where multiple sized input is converted into 1d vector
* <b>Dense layer</b> on CNN this is mostly used to do classification after doing whole convolution thing.


## What do we need?

* <b>Keras</b> this is main library we need to import.
* <b>Sequential</b> we use to create model and add layers.
* <b>Layers</b> we mainly use convolution, maxpool, dense layer and dropout layer.
* <b>Optimizers</b> we use various optimizers like SGD, Adam, Adagrad etc.

Sequential method will use create a simplest model of NN. 

<code>
    import keras 
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from keras.optimizers import SGD
    import numpy as np
    import matplotlib.pyplot as plt
    import time
</code>



## Do you know?

When a machine learning model has high training accuracy and very low validation then this case is known as over-fitting. The reasons for this can be as follows:

1. The hypothesis function you are using is too complex that your model perfectly fits the training data but fails to do on test/validation data.
2. The number of learning parameters in your model is way too big that instead of generalizing the examples , your model learns those examples and hence the model performs badly on test/validation data.

To solve the above problems a number of solutions can be tried depending on your dataset:

1. Use a simple cost and loss function.
2. Use regulation which helps in reducing over-fitting i.e Dropout.
3. Reduce the number of learning parameters in your model.

These are the 3 solutions that are most likely to improve the validation accuracy of your model and still if these don't work check your inputs whether they have right normalization or not.



```python

```
