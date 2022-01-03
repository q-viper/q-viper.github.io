---
title:  "Messing Up With Convolutional Neural Nets: Autoencoder for Grayscale to RGB Image"
date:   2021-03-11 01:29:17 +0545
categories:
  - Computer Vision
  - Image Processing
  - Project
  - Programming
tags:
  - Computer Vision
  - cnn
  - auto encoder
  - python 
  - web development
header:
  teaser: assets/messing-with-cnn/l_ab/output_32_31.png
---
# Messing Up with CNN
I might stop to write new blogs in this site so please visit [dataqoil.com](https://dataqoil.com) for more cool stuffs.


CNN has been so famous in last few years and these days many state of the art techniques are here to do amazing things on computer vision. One can not stop by listing the names of those researchs. In first few notebooks, I am thinking about Grayscale to RGB conversion. My work might not be complete but I intend to update it and add more concepts too. **I am only messing up with CNN**. 


# Conversion of Grayscale Image to RGB
Here in this blog, I am not doing anything amazing other than making a CNN and feed it with Grayscale image as input and compare it with its corresponding RGB image, and check how well it will perform well as the time passes. 

As we all know, RGB to Grayscale is irreversal process and once it is done then no way we can colorize it with as original. But in recent years, lots of state of the art features has made this possible. 

For this technique, I tried 2 approaches. 
* Input as Grayscale and output as RGB image
* Input as LAB's L value and output as AB value of LAB.
    * LAB stands for Lightness(which is only grayscale as we know), AB represents Red/Green and Blue/Yellow respectively. [More Here](https://www.xrite.com/blog/lab-color-space).

## Preliminary Steps
### Dummy Show Function
Just to visualize our image on large figure.
```python
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def show(img, fig_size=(10, 10)):
  figure = plt.figure(figsize=fig_size)
  plt.imshow(img)
  plt.xticks([])
  plt.yticks([])
  plt.show()

img = np.random.randint(0, 255, (100, 100))  
show(img)

```

![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_4_0.png)

### Prepare Dataset
For this problem, we need plenty of dataset and I am going to use dataset available publicly on Kaggle and other sources. For ease of hardware requirements, I am using google colab because I can have large storage and RAM along with GPU for training. Only main thing I am concerned about is the architecture and weight of the model file. So all the images downloaded on COLAB kernel are not necessarily saved on drive.

#### Dataset: Cat vs Dog 
Who doesn't remembers this dataset?

```python
import zipfile
import os

!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
    -O /tmp/cats_and_dogs_filtered.zip
local_zip = '/tmp/cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()
```

    --2021-02-28 13:11:55--  https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
    Resolving storage.googleapis.com (storage.googleapis.com)... 74.125.197.128, 74.125.142.128, 74.125.195.128, ...
    Connecting to storage.googleapis.com (storage.googleapis.com)|74.125.197.128|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 68606236 (65M) [application/zip]
    Saving to: ‘/tmp/cats_and_dogs_filtered.zip’
    
    /tmp/cats_and_dogs_ 100%[===================>]  65.43M   170MB/s    in 0.4s    
    
    2021-02-28 13:11:56 (170 MB/s) - ‘/tmp/cats_and_dogs_filtered.zip’ saved [68606236/68606236]
    

```python
base_dir = '/tmp/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_cat_fnames = os.listdir(train_cats_dir)
print(train_cat_fnames[:10])

train_dog_fnames = os.listdir(train_dogs_dir)
train_dog_fnames.sort()
print(train_dog_fnames[:10])
```

    ['cat.922.jpg', 'cat.994.jpg', 'cat.965.jpg', 'cat.721.jpg', 'cat.856.jpg', 'cat.941.jpg', 'cat.419.jpg', 'cat.974.jpg', 'cat.466.jpg', 'cat.259.jpg']
    ['dog.0.jpg', 'dog.1.jpg', 'dog.10.jpg', 'dog.100.jpg', 'dog.101.jpg', 'dog.102.jpg', 'dog.103.jpg', 'dog.104.jpg', 'dog.105.jpg', 'dog.106.jpg']
    

#### MIT Places Dataset
Contains variety of scenes.


```python
# mit places data http://places.csail.mit.edu/
!wget http://data.csail.mit.edu/places/places205/testSetPlaces205_resize.tar.gz
!tar -xzf testSetPlaces205_resize.tar.gz
```

    --2021-02-28 13:12:05--  http://data.csail.mit.edu/places/places205/testSetPlaces205_resize.tar.gz
    Resolving data.csail.mit.edu (data.csail.mit.edu)... 128.52.129.40
    Connecting to data.csail.mit.edu (data.csail.mit.edu)|128.52.129.40|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 2341250899 (2.2G) [application/octet-stream]
    Saving to: ‘testSetPlaces205_resize.tar.gz’
    
    testSetPlaces205_re 100%[===================>]   2.18G  15.9MB/s    in 1m 46s  
    
    2021-02-28 13:13:51 (21.1 MB/s) - ‘testSetPlaces205_resize.tar.gz’ saved [2341250899/2341250899]
    
    
#### Kaggle: API Key
In order to be able to use kaggle data on COLAB or anywhere, we need to have kaggle's API key. Which can be easily downloaded once our account is allowed.

```python
# to get kaggle dataset
!pip install kaggle
# uplaod kaggle.json firsst
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

    Requirement already satisfied: kaggle in /usr/local/lib/python3.7/dist-packages (1.5.10)
    Requirement already satisfied: python-dateutil in /usr/local/lib/python3.7/dist-packages (from kaggle) (2.8.1)
    Requirement already satisfied: python-slugify in /usr/local/lib/python3.7/dist-packages (from kaggle) (4.0.1)
    Requirement already satisfied: certifi in /usr/local/lib/python3.7/dist-packages (from kaggle) (2020.12.5)
    Requirement already satisfied: six>=1.10 in /usr/local/lib/python3.7/dist-packages (from kaggle) (1.15.0)
    Requirement already satisfied: urllib3 in /usr/local/lib/python3.7/dist-packages (from kaggle) (1.24.3)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from kaggle) (4.41.1)
    Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from kaggle) (2.23.0)
    Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.7/dist-packages (from python-slugify->kaggle) (1.3)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->kaggle) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->kaggle) (2.10)
    

#### Kaggle: Datasets
Trick is, go to URL of the dataset and copy the characters after .com, . i.e  if my dataset URL is https://www.kaggle.com/qramkrishna/corn-leaf-infection-dataset, then I need only qramkrishna/corn-leaf-infection-dataset.

##### Flowers Dataset
Who doesn't like flowers? File will be downloaded on ZIP with same name as dataset. Extract it on our kernel workspace.


```python
!kaggle datasets download -d alxmamaev/flowers-recognition

local_zip = 'flowers-recognition.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('flowers-recognition')
zip_ref.close()

fdir = "flowers-recognition/flowers"
flowers_dir = [os.path.join(fdir, fd) for fd in os.listdir(fdir)]
flowers_dir
```

    Downloading flowers-recognition.zip to /content
     98% 441M/450M [00:04<00:00, 103MB/s] 
    100% 450M/450M [00:04<00:00, 102MB/s]
    

    ['flowers-recognition/flowers/tulip',
     'flowers-recognition/flowers/sunflower',
     'flowers-recognition/flowers/daisy',
     'flowers-recognition/flowers/flowers',
     'flowers-recognition/flowers/rose',
     'flowers-recognition/flowers/dandelion']



##### Fruits Dataset
Yummy!


```python
!kaggle datasets download -d moltean/fruits

local_zip = 'fruits.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('fruits')
zip_ref.close()
```

    Downloading fruits.zip to /content
     98% 748M/760M [00:19<00:00, 54.9MB/s]
    100% 760M/760M [00:19<00:00, 40.9MB/s]
    


```python
ftrain = "/content/fruits/fruits-360/Training/"
ftrain_dir = [os.path.join(ftrain, fd) for fd in os.listdir(ftrain)]

ftest = "/content/fruits/fruits-360/Test/"
ftest_dir = [os.path.join(ftest, fd) for fd in os.listdir(ftest)]
ftest_dir
```

##### Corn Infection Dataset
Why did I choose this?

```python
!kaggle datasets download -d qramkrishna/corn-leaf-infection-dataset

local_zip = 'corn-leaf-infection-dataset.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('corn-leaf-infection-dataset')
zip_ref.close()


```

    Downloading corn-leaf-infection-dataset.zip to /content
    100% 13.0G/13.0G [05:46<00:00, 39.6MB/s]
    100% 13.0G/13.0G [05:46<00:00, 40.2MB/s]
    


```python
ctrain = ["/content/corn-leaf-infection-dataset/Corn Disease detection/Healthy corn", 
          "/content/corn-leaf-infection-dataset/Corn Disease detection/Infected"]
```

##### Inria Dataset


```python
!wget ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar
!tar -xvf INRIAPerson.tar
!rm INRIAPerson.tar
```


```python
inria_train = ['/content/INRIAPerson/70X134H96/Test/pos',
               "/content/INRIAPerson/96X160H96/Train/pos",
               "/content/INRIAPerson/train_64x128_H96/neg",
               "/content/INRIAPerson/train_64x128_H96/pos",
               "/content/INRIAPerson/Train/neg", 
               "/content/INRIAPerson/Train/pos"]
inria_test = ["/content/INRIAPerson/test_64x128_H96/pos",
              "/content/INRIAPerson/test_64x128_H96/neg",
              "/content/INRIAPerson/Test/neg", 
              "/content/INRIAPerson/Test/pos"]
```


```python
# delete these huge zip file to prevent memory full
!rm corn-leaf-infection-dataset.zip
!rm testSetPlaces205_resize.tar.gz
```




## Input as Grayscale and Output as RGB Image
This is traditional encoder which tries to add new 2 layers on existing layer but it is not as simple as I think it is. 

### Custom Data Generator
Lets take the possible image's root directory and store the full path of each image on a list. Then shuffle that list to make some random effect.
This same generator can be used on both cases and we only have to edit it little bit.

```python
from tensorflow.keras.utils import Sequence

img_size = (224, 224)
class ImageGenerator(Sequence):
  def __init__(self, dirs=[], 
              target_size=(224,224), batch_size=32):
    self.batch_size=batch_size
    self.target_size = target_size
    self.dirs = dirs
    self.all_dirs = []
    for dir in self.dirs:
      self.all_dirs.extend([os.path.join(dir, fname) for fname in os.listdir(dir)])
    
    self.x = np.arange(len(self.all_dirs))
    np.random.shuffle(self.x)

  def __len__(self):
        return int(np.ceil(len(self.x)/float(self.batch_size)))
    
  def __getitem__(self, idx):
      batch_x = self.x[idx * self.batch_size:(idx+1) * self.batch_size]
      #batch_y = self.y[idx * self.batch_size:(idx+1) * self.batch_size]
      
      x, y = self.generate_image(batch_x)
      #print(batch.shape)
      return x, y
      
  def generate_image(self, ids):
    image_size = self.target_size
    batch_x = []
    batch_y = []
    for i in ids:
      try:
        bgr = cv2.imread(self.all_dirs[i], 1)
        bgr = cv2.resize(bgr, image_size)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        batch_x.append(gray)
        batch_y.append(rgb)
      except:
        pass
    batch_x = np.array(batch_x).reshape(len(batch_x), image_size[0], image_size[1], 1)/255
    batch_y = np.array(batch_y).reshape(len(batch_y), image_size[0], image_size[1], 3)/255
    
    return batch_x, batch_y

tdirs=['/tmp/cats_and_dogs_filtered/train/dogs',
      '/tmp/cats_and_dogs_filtered/train/cats',
       "/content/testSet_resize",
       ]

tdirs.extend(flowers_dir)
# tdirs.extend(ctrain)
tdirs.extend(inria_train)
# tdirs.extend(ftrain_dir)

vdirs = ['/tmp/cats_and_dogs_filtered/validation/dogs',
        '/tmp/cats_and_dogs_filtered/validation/cats']
vdirs.extend(inria_test)

# vdirs.extend(ftest_dir)


train_generator = ImageGenerator(tdirs, 
                           target_size=img_size, batch_size=64)

valid_generator = ImageGenerator(dirs=vdirs, 
                           target_size=img_size, batch_size=32)


for i in range(2):
  # print(generator.__getitem__(i)[1][0].shape)
  show(train_generator.__getitem__(i)[0][0].reshape(img_size))
  show(train_generator.__getitem__(i)[1][0].reshape(img_size[0], img_size[1], 3))
  
```


    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_25_0.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_25_1.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_25_2.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_25_3.png)
    

* Keras Datagenerator allows us to write our own custom data generator and do our own stuffs inside it. 
* We have just given the root for each image and then we read the names of image and make a list where all the path of images are on. Then shuffle it.
* Use the list on each epoch to generate batch using the indices.
* We read image in RGB and convert it to Grayscale.
* Normalize images and return batch.


### Create a Model
I am going to use pretrained model because they tend to have learned more features than the other models. I am using filters in such a way that model's output shape matches the label shape. If we intend to use different filters, then it is best idea to use `Resize` layer.


```python
from keras.applications.inception_resnet_v2 import preprocess_input, InceptionResNetV2
from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Lambda, Reshape, BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import tensorflow as tf

from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras import activations

def create_base_network(input_shape, output_shape):
    """Get the base network to do the feature extract for the latent embedding
    
    Args:
        input_shape (tuple): Shape of image tensor input
        
    Returns:
        keras.models.Model
    """
    # input = Input(shape=input_shape)
    img_input = Input(shape=input_shape, name = 'grayscale_input_layer')
    x = Conv2D(3, (3,3),  padding= 'same', name = 'grayscale_RGB_layer', activation="relu")(img_input)
    # x=Lambda(lambda v: tf.cast(tf.compat.v1.spectral.fft(tf.cast(v,dtype=tf.complex64)),tf.float32))(x)
    # inception = InceptionResNetV2(input_shape = (input_shape[0], input_shape[1], 3), include_top = False, weights = 'imagenet')
    inception = InceptionV3(weights='imagenet',include_top=False,input_shape=(input_shape[0], input_shape[1], 3))
    inception.layers.pop()  # Remove classification layer
    # inception.summary()
    inception = inception(x)
    
    inception = Conv2D(128, (3,3), padding= 'same')(inception)
    inception = BatchNormalization()(inception)
    inception = activations.relu(inception)
    upsample = UpSampling2D(2)(inception)

    upsample = Conv2D(128, (3,3))(upsample)
    upsample = BatchNormalization()(upsample)
    upsample = activations.relu(upsample)
    upsample = UpSampling2D(2)(upsample)

    upsample = Conv2D(64, (3,3))(upsample)
    upsample = BatchNormalization()(upsample)
    upsample = activations.relu(upsample)
    upsample = UpSampling2D(2)(upsample)

    upsample = Conv2D(64, (3,3), padding="same")(upsample)
    upsample = BatchNormalization()(upsample)
    upsample = activations.relu(upsample)
    upsample = UpSampling2D(2)(upsample)

    upsample = Conv2D(32, (3,3), padding="same")(upsample)
    upsample = BatchNormalization()(upsample)
    upsample = activations.relu(upsample)
    upsample = UpSampling2D(2)(upsample)

    upsample = Conv2D(2, (3,3), padding="same")(upsample)
    upsample = BatchNormalization()(upsample)
    upsample = activations.tanh(upsample)
    upsample = UpSampling2D(2)(upsample)

    upsample = Conv2D(3, (3,3), padding="same")(upsample)
    upsample = activations.sigmoid(upsample)

    # resize = Lambda(resize_layer, output_shape=resize_layer_shape, arguments={"shape":output_shape})(upsample)
    # this is resizing layer
    # resize = tf.keras.layers.experimental.preprocessing.Resizing(input_shape[0], input_shape[1])(upsample)
    
    # output = resize
    output=upsample
    model = Model(inputs=[img_input], outputs=[output])
    model.summary()
    return model
model = create_base_network((img_size[0],img_size[1], 1), (img_size[0], img_size[1], 3))
```

    Model: "model_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    grayscale_input_layer (Input [(None, 224, 224, 1)]     0         
    _________________________________________________________________
    grayscale_RGB_layer (Conv2D) (None, 224, 224, 3)       30        
    _________________________________________________________________
    inception_v3 (Functional)    (None, 5, 5, 2048)        21802784  
    _________________________________________________________________
    conv2d_195 (Conv2D)          (None, 5, 5, 128)         2359424   
    _________________________________________________________________
    batch_normalization_194 (Bat (None, 5, 5, 128)         512       
    _________________________________________________________________
    tf.nn.relu_5 (TFOpLambda)    (None, 5, 5, 128)         0         
    _________________________________________________________________
    up_sampling2d_6 (UpSampling2 (None, 10, 10, 128)       0         
    _________________________________________________________________
    conv2d_196 (Conv2D)          (None, 8, 8, 128)         147584    
    _________________________________________________________________
    batch_normalization_195 (Bat (None, 8, 8, 128)         512       
    _________________________________________________________________
    tf.nn.relu_6 (TFOpLambda)    (None, 8, 8, 128)         0         
    _________________________________________________________________
    up_sampling2d_7 (UpSampling2 (None, 16, 16, 128)       0         
    _________________________________________________________________
    conv2d_197 (Conv2D)          (None, 14, 14, 64)        73792     
    _________________________________________________________________
    batch_normalization_196 (Bat (None, 14, 14, 64)        256       
    _________________________________________________________________
    tf.nn.relu_7 (TFOpLambda)    (None, 14, 14, 64)        0         
    _________________________________________________________________
    up_sampling2d_8 (UpSampling2 (None, 28, 28, 64)        0         
    _________________________________________________________________
    conv2d_198 (Conv2D)          (None, 28, 28, 64)        36928     
    _________________________________________________________________
    batch_normalization_197 (Bat (None, 28, 28, 64)        256       
    _________________________________________________________________
    tf.nn.relu_8 (TFOpLambda)    (None, 28, 28, 64)        0         
    _________________________________________________________________
    up_sampling2d_9 (UpSampling2 (None, 56, 56, 64)        0         
    _________________________________________________________________
    conv2d_199 (Conv2D)          (None, 56, 56, 32)        18464     
    _________________________________________________________________
    batch_normalization_198 (Bat (None, 56, 56, 32)        128       
    _________________________________________________________________
    tf.nn.relu_9 (TFOpLambda)    (None, 56, 56, 32)        0         
    _________________________________________________________________
    up_sampling2d_10 (UpSampling (None, 112, 112, 32)      0         
    _________________________________________________________________
    conv2d_200 (Conv2D)          (None, 112, 112, 2)       578       
    _________________________________________________________________
    batch_normalization_199 (Bat (None, 112, 112, 2)       8         
    _________________________________________________________________
    tf.math.tanh_1 (TFOpLambda)  (None, 112, 112, 2)       0         
    _________________________________________________________________
    up_sampling2d_11 (UpSampling (None, 224, 224, 2)       0         
    _________________________________________________________________
    conv2d_201 (Conv2D)          (None, 224, 224, 3)       57        
    _________________________________________________________________
    tf.math.sigmoid_1 (TFOpLambd (None, 224, 224, 3)       0         
    =================================================================
    Total params: 24,441,313
    Trainable params: 24,406,045
    Non-trainable params: 35,268
    _________________________________________________________________
    

#### Train It
Lets define some optimizers and loss function to compile the model.


```python
from keras.optimizers import Adam
adam = Adam(lr=0.001)
model.compile(optimizer=adam, loss='mse')

```

Also using the Callbacks will help our model to generalize properly. `EarlyStopping` allows model training to be stopped if overfitting is occuring. `ReduceLROnPlateau` is used to reduce the learning rate by given factor until it reaches some minimum value. We also will save the model's weight on each epoch. 


```python
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback


class ShowImagesOnEpochEnd(Callback):
    """ 
    Inherit from keras.callbacks.Callback
    """
    def __init__(self, data=None, img_size=(224, 224)):
        """ 
        data: dataset to view from
        img_size: image size
        """
        self.data = data
        self.img_size = img_size

    def lab2RGB(self, X, Y):
      canvas = np.zeros((img_size[0], img_size[1], 3), dtype=np.float64)
      canvas[:,:,0] = X[0][:,:,0]
      canvas[:,:,1:] = Y[0]*128

      return lab2rgb(canvas)

    def on_epoch_end(self, epoch, logs={}):
        inds = np.random.randint(0, 20, 3)
        for i in inds:
          # print(logs)
          img=self.data.__getitem__(i)[0][0].reshape(1, self.img_size[0], self.img_size[1], 1)
          
          gimg=self.data.__getitem__(i)[0][0].reshape(1, self.img_size[0], self.img_size[1], 1)
          rgbimg = self.data.__getitem__(i)[1][0].reshape(self.img_size[0], self.img_size[1], 3)
          
          
          plt.figure(figsize=(20,20))
          plt.subplot(1,2,1)
          plt.imshow(rgbimg)
          plt.title("True RGB Image")
          plt.xticks([])
          plt.yticks([])

          out = self.model.predict(img).reshape(self.img_size[0], self.img_size[1], 3)
          # out = self.lab2RGB(img, out)
          plt.subplot(1,2,2)
          plt.imshow(out)
          plt.title("Output RGB Image")
          plt.xticks([])
          plt.yticks([])
          plt.show()

callbacks = [
    EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(factor=0.9, patience=5, min_lr=0.0000001, verbose=1),
    ModelCheckpoint("/content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_{epoch:02d}.h5", verbose=1, save_weights_only=True),
    ShowImagesOnEpochEnd(data=valid_generator)
]
```
* The custom callback, `ShowImagesOnEpochEnd` shows us the result of grayscale images on the epoch end.

```python
history = model.fit(train_generator, 
          epochs=30,
          validation_data=valid_generator,
          callbacks=callbacks)

```

    Epoch 1/30
    881/881 [==============================] - 324s 350ms/step - loss: 0.0196 - val_loss: 0.0080    
    Epoch 00001: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_01.h5

![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_1.png)
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_2.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_3.png)
    


    Epoch 2/30
    881/881 [==============================] - 302s 343ms/step - loss: 0.0087 - val_loss: 0.0079
    
    Epoch 00002: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_02.h5
    


    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_5.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_6.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_7.png)
    


    Epoch 3/30
    881/881 [==============================] - 304s 344ms/step - loss: 0.0085 - val_loss: 0.0078
    
    Epoch 00003: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_03.h5
    


    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_9.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_10.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_11.png)
    


    Epoch 4/30
    881/881 [==============================] - 309s 350ms/step - loss: 0.0084 - val_loss: 0.0077
    
    Epoch 00004: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_04.h5
    


    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_13.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_14.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_15.png)
    


    Epoch 5/30
    881/881 [==============================] - 315s 357ms/step - loss: 0.0083 - val_loss: 0.0079
    
    Epoch 00005: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_05.h5
    


    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_17.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_18.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_19.png)
    


    Epoch 6/30
    881/881 [==============================] - 330s 374ms/step - loss: 0.0082 - val_loss: 0.0079
    
    Epoch 00006: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_06.h5
    


    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_21.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_22.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_23.png)
    


    Epoch 7/30
    881/881 [==============================] - 342s 388ms/step - loss: 0.0083 - val_loss: 0.0078
    
    Epoch 00007: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_07.h5
    


    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_25.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_26.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_27.png)
    


    Epoch 8/30
    881/881 [==============================] - 356s 404ms/step - loss: 0.0081 - val_loss: 0.0076
    
    Epoch 00008: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_08.h5
    


    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_29.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_30.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_31.png)
    


    Epoch 9/30
    881/881 [==============================] - 368s 417ms/step - loss: 0.0081 - val_loss: 0.0077
    
    Epoch 00009: ReduceLROnPlateau reducing learning rate to 0.0009000000427477062.
    
    Epoch 00009: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_09.h5
    


    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_33.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_34.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_35.png)
    


    Epoch 10/30
    881/881 [==============================] - 383s 435ms/step - loss: 0.0080 - val_loss: 0.0075
    
    Epoch 00010: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_10.h5
    


    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_37.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_38.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_39.png)
    


    Epoch 11/30
    881/881 [==============================] - 396s 449ms/step - loss: 0.0080 - val_loss: 0.0078
    
    Epoch 00011: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_11.h5
    


    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_41.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_42.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_43.png)
    


    Epoch 12/30
    881/881 [==============================] - 408s 462ms/step - loss: 0.0079 - val_loss: 0.0075
    
    Epoch 00012: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_12.h5
    


    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_45.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_46.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_47.png)
    


    Epoch 13/30
    881/881 [==============================] - 415s 471ms/step - loss: 0.0079 - val_loss: 0.0076
    
    Epoch 00013: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_13.h5
    


    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_49.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_50.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_51.png)
    


    Epoch 14/30
    881/881 [==============================] - 424s 481ms/step - loss: 0.0078 - val_loss: 0.0075
    
    Epoch 00014: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_14.h5
    


    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_53.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_54.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_55.png)
    


    Epoch 15/30
    881/881 [==============================] - 439s 498ms/step - loss: 0.0078 - val_loss: 0.0076
    
    Epoch 00015: ReduceLROnPlateau reducing learning rate to 0.0008100000384729356.
    
    Epoch 00015: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_15.h5
    


    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_57.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_58.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_59.png)
    


    Epoch 16/30
    881/881 [==============================] - 451s 512ms/step - loss: 0.0078 - val_loss: 0.0075
    
    Epoch 00016: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_16.h5
    


    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_61.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_62.png)
    



    
![]({{site.url}}/assets/messing-with-cnn/gray_rgb/output_34_63.png)
    


    Epoch 17/30
    784/881 [=========================>....] - ETA: 46s - loss: 0.0077

The trainnning ended here because of my network interruption but we can clearly see that this technique disappointed. But this might work after we have huge dataset. 



## Input as L and Output as AB of LAB
I am writing this technique after reading [this](https://emilwallner.medium.com/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d) awesome blog.
Everything will be same as above except Generator and the Model.
### Image Generator
I have imported 2 methods from `skimage.color` to conversion of RGB to LAB and reverse.

```python
from tensorflow.keras.utils import Sequence
from skimage.color import rgb2lab, lab2rgb

img_size = (224, 224)
class ImageGenerator(Sequence):
  def __init__(self, dirs=[], 
              target_size=(224,224), batch_size=32):
    self.batch_size=batch_size
    self.target_size = target_size
    self.dirs = dirs
    self.all_dirs = []
    for dir in self.dirs:
      self.all_dirs.extend([os.path.join(dir, fname) for fname in os.listdir(dir)])
    
    self.x = np.arange(len(self.all_dirs))
    np.random.shuffle(self.x)

  def __len__(self):
        return int(np.ceil(len(self.x)/float(self.batch_size)))
    
  def __getitem__(self, idx):
      batch_x = self.x[idx * self.batch_size:(idx+1) * self.batch_size]
      #batch_y = self.y[idx * self.batch_size:(idx+1) * self.batch_size]
      
      x, y = self.generate_image(batch_x)
      #print(batch.shape)
      return x, y
      
  def generate_image(self, ids):
    image_size = self.target_size
    batch_x = []
    batch_y = []
    for i in ids:
      try:
        bgr = cv2.imread(self.all_dirs[i], 1)
        bgr = cv2.resize(bgr, image_size)
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        X = rgb2lab(1.0/255*image)[:,:,0]
        Y = rgb2lab(1.0/255*image)[:,:,1:]
        Y = Y / 128
        X = X.reshape(1, image_size[0], image_size[1], 1)
        Y = Y.reshape(1, image_size[0], image_size[1], 2)
        
        batch_x.append(X)
        batch_y.append(Y)

        # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # batch_x.append(gray)
        # batch_y.append(rgb)
      except:
        pass
    # batch_x = np.array(batch_x).reshape(len(batch_x), image_size[0], image_size[1], 1)/255
    # batch_y = np.array(batch_y).reshape(len(batch_y), image_size[0], image_size[1], 3)/255
    batch_x = np.array(batch_x).reshape(len(batch_x), image_size[0], image_size[1], 1)
    batch_y = np.array(batch_y).reshape(len(batch_y), image_size[0], image_size[1], 2)

    return batch_x, batch_y

# tdirs =["/content/flowers-recognition/flowers/daisy"]

tdirs=['/tmp/cats_and_dogs_filtered/train/dogs',
      '/tmp/cats_and_dogs_filtered/train/cats',
       "/content/testSet_resize",
       ]


tdirs.extend(flowers_dir)
# tdirs.extend(ctrain)
tdirs.extend(inria_train)
# tdirs.extend(ftrain_dir)

# vdirs=["/content/flowers-recognition/flowers/daisy"]
vdirs = ['/tmp/cats_and_dogs_filtered/validation/dogs',
        '/tmp/cats_and_dogs_filtered/validation/cats']

# vdirs.extend(ftest_dir)
vdirs.extend(inria_test)

train_generator = ImageGenerator(tdirs, 
                           target_size=img_size, batch_size=64)

valid_generator = ImageGenerator(dirs=vdirs, 
                           target_size=img_size, batch_size=32)

def lab2RGB(X, Y):
  canvas = np.zeros((img_size[0], img_size[1], 3), dtype=np.float64)
  canvas[:,:,0] = X[0][:,:,0]
  canvas[:,:,1:] = Y[0]*128

  return lab2rgb(canvas)

for i in range(2):
  X = train_generator.__getitem__(i)[0][0].reshape(1, img_size[0], img_size[1], 1)
  Y = train_generator.__getitem__(i)[1][0].reshape(1, img_size[0], img_size[1], 2)

  # output = model.predict(X)
  show(lab2RGB(X, Y))
  show(X.reshape(img_size))
  
```


    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_24_0.png)
    



    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_24_1.png)
    



    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_24_2.png)
    



    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_24_3.png)
    
* The maximum value in LAB is 128 and hence we divide by that value for normalization.

### Create Model
```python
def create_base_network(input_shape, output_shape):
    img_input = Input(shape=input_shape, name = 'grayscale_input_layer')
    x = Conv2D(3, (3,3),  padding= 'same', name = 'grayscale_RGB_layer', activation="relu")(img_input)
    # x=Lambda(lambda v: tf.cast(tf.compat.v1.spectral.fft(tf.cast(v,dtype=tf.complex64)),tf.float32))(x)
    # inception = InceptionResNetV2(input_shape = (input_shape[0], input_shape[1], 3), include_top = False, weights = 'imagenet')
    inception = InceptionV3(weights='imagenet',include_top=False,input_shape=(input_shape[0], input_shape[1], 3))
    inception.layers.pop()  # Remove classification layer
    # inception.summary()
    for layer in inception.layers:
      layer.trainnable=False
    inception = inception(x)
    
    inception = Conv2D(128, (3,3), padding= 'same')(inception)
    inception = BatchNormalization()(inception)
    inception = activations.relu(inception)
    upsample = UpSampling2D(2)(inception)

    upsample = Conv2D(128, (3,3))(upsample)
    upsample = BatchNormalization()(upsample)
    upsample = activations.relu(upsample)
    upsample = UpSampling2D(2)(upsample)

    upsample = Conv2D(64, (3,3))(upsample)
    upsample = BatchNormalization()(upsample)
    upsample = activations.relu(upsample)
    upsample = UpSampling2D(2)(upsample)

    upsample = Conv2D(64, (3,3), padding="same")(upsample)
    upsample = BatchNormalization()(upsample)
    upsample = activations.relu(upsample)
    upsample = UpSampling2D(2)(upsample)

    upsample = Conv2D(32, (3,3), padding="same")(upsample)
    upsample = BatchNormalization()(upsample)
    upsample = activations.relu(upsample)
    upsample = UpSampling2D(2)(upsample)

    upsample = Conv2D(2, (3,3), padding="same")(upsample)
    upsample = BatchNormalization()(upsample)
    upsample = activations.tanh(upsample)
    upsample = UpSampling2D(2)(upsample)

    # upsample = Conv2D(3, (3,3), padding="same")(upsample)
    # upsample = activations.relu(upsample)

    # resize = Lambda(resize_layer, output_shape=resize_layer_shape, arguments={"shape":output_shape})(upsample)
    # this is resizing layer
    # resize = tf.keras.layers.experimental.preprocessing.Resizing(input_shape[0], input_shape[1])(upsample)
    
    # output = resize
    output=upsample
    model = Model(inputs=[img_input], outputs=[output])
    model.summary()
    return model
model = create_base_network((img_size[0],img_size[1], 1), (img_size[0], img_size[1], 3))
```

    Model: "model_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    grayscale_input_layer (Input [(None, 224, 224, 1)]     0         
    _________________________________________________________________
    grayscale_RGB_layer (Conv2D) (None, 224, 224, 3)       30        
    _________________________________________________________________
    inception_v3 (Functional)    (None, 5, 5, 2048)        21802784  
    _________________________________________________________________
    conv2d_194 (Conv2D)          (None, 5, 5, 128)         2359424   
    _________________________________________________________________
    batch_normalization_194 (Bat (None, 5, 5, 128)         512       
    _________________________________________________________________
    tf.nn.relu_5 (TFOpLambda)    (None, 5, 5, 128)         0         
    _________________________________________________________________
    up_sampling2d_6 (UpSampling2 (None, 10, 10, 128)       0         
    _________________________________________________________________
    conv2d_195 (Conv2D)          (None, 8, 8, 128)         147584    
    _________________________________________________________________
    batch_normalization_195 (Bat (None, 8, 8, 128)         512       
    _________________________________________________________________
    tf.nn.relu_6 (TFOpLambda)    (None, 8, 8, 128)         0         
    _________________________________________________________________
    up_sampling2d_7 (UpSampling2 (None, 16, 16, 128)       0         
    _________________________________________________________________
    conv2d_196 (Conv2D)          (None, 14, 14, 64)        73792     
    _________________________________________________________________
    batch_normalization_196 (Bat (None, 14, 14, 64)        256       
    _________________________________________________________________
    tf.nn.relu_7 (TFOpLambda)    (None, 14, 14, 64)        0         
    _________________________________________________________________
    up_sampling2d_8 (UpSampling2 (None, 28, 28, 64)        0         
    _________________________________________________________________
    conv2d_197 (Conv2D)          (None, 28, 28, 64)        36928     
    _________________________________________________________________
    batch_normalization_197 (Bat (None, 28, 28, 64)        256       
    _________________________________________________________________
    tf.nn.relu_8 (TFOpLambda)    (None, 28, 28, 64)        0         
    _________________________________________________________________
    up_sampling2d_9 (UpSampling2 (None, 56, 56, 64)        0         
    _________________________________________________________________
    conv2d_198 (Conv2D)          (None, 56, 56, 32)        18464     
    _________________________________________________________________
    batch_normalization_198 (Bat (None, 56, 56, 32)        128       
    _________________________________________________________________
    tf.nn.relu_9 (TFOpLambda)    (None, 56, 56, 32)        0         
    _________________________________________________________________
    up_sampling2d_10 (UpSampling (None, 112, 112, 32)      0         
    _________________________________________________________________
    conv2d_199 (Conv2D)          (None, 112, 112, 2)       578       
    _________________________________________________________________
    batch_normalization_199 (Bat (None, 112, 112, 2)       8         
    _________________________________________________________________
    tf.math.tanh_1 (TFOpLambda)  (None, 112, 112, 2)       0         
    _________________________________________________________________
    up_sampling2d_11 (UpSampling (None, 224, 224, 2)       0         
    =================================================================
    Total params: 24,441,256
    Trainable params: 24,405,988
    Non-trainable params: 35,268
    _________________________________________________________________
    

### Train It

#### Compile

```python
from keras.optimizers import Adam
adam = Adam(lr=0.001)
model.compile(optimizer=adam, loss='mse')

```

#### Callbacks
As above but we will use LAB to RGB now.
```python

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback

class ShowImagesOnEpochEnd(Callback):
    """ 
    Inherit from keras.callbacks.Callback
    """
    def __init__(self, data=None, img_size=(224, 224)):
        """ 
        data: dataset to view from
        img_size: image size
        """
        self.data = data
        self.img_size = img_size

    def lab2RGB(self, X, Y):
      canvas = np.zeros((img_size[0], img_size[1], 3), dtype=np.float64)
      canvas[:,:,0] = X[0][:,:,0]
      canvas[:,:,1:] = Y[0]*128

      return lab2rgb(canvas)

    def on_epoch_end(self, epoch, logs={}):
        inds = np.random.randint(0, 20, 3)
        for i in inds:
          # print(logs)
          gimg=self.data.__getitem__(i)[0][0].reshape(1, self.img_size[0], self.img_size[1], 1)
          abimg = self.data.__getitem__(i)[1][0].reshape(1, self.img_size[0], self.img_size[1], 2)
          
          inp_img = self.lab2RGB(gimg, abimg)

          plt.figure(figsize=(20,20))
          plt.subplot(1,2,1)
          plt.imshow(inp_img.reshape(self.img_size[0], self.img_size[1], 3), cmap="gray")
          plt.title("Ture RGB Image")
          plt.xticks([])
          plt.yticks([])

          out = self.model.predict(gimg)
          output = self.lab2RGB(gimg, out)
          plt.subplot(1,2,2)
          plt.imshow(output)
          plt.title("Predicted RGB Image")
          plt.xticks([])
          plt.yticks([])
          plt.show()

callbacks = [
    EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(factor=0.9, patience=5, min_lr=0.0000001, verbose=1),
    ModelCheckpoint("/content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_LAB_{epoch:02d}.h5", verbose=1, save_weights_only=True),
    ShowImagesOnEpochEnd(data=valid_generator)
    ]

# model.load_weights("/content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_LAB_02.h5")

```

```python
history = model.fit(train_generator, 
          epochs=30,
          validation_data=valid_generator,
          callbacks=callbacks)
```

    Epoch 1/30
    881/881 [==============================] - 2151s 2s/step - loss: 0.0212 - val_loss: 0.0075
    
    Epoch 00001: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_LAB_01.h5
    


    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_1.png)
    



    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_2.png)
    



    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_3.png)
    


    Epoch 2/30
    881/881 [==============================] - 2133s 2s/step - loss: 0.0117 - val_loss: 0.0080
    
    Epoch 00002: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_LAB_02.h5
    


    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_5.png)
    



    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_6.png)
    



    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_7.png)
    


    Epoch 3/30
    881/881 [==============================] - 2137s 2s/step - loss: 0.0108 - val_loss: 0.2515
    
    Epoch 00003: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_LAB_03.h5
    


    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_9.png)
    



    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_10.png)
    



    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_11.png)
    


    Epoch 4/30
    881/881 [==============================] - 2150s 2s/step - loss: 0.0109 - val_loss: 0.0069
    
    Epoch 00004: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_LAB_04.h5
    


    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_13.png)
    



    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_14.png)
    



    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_15.png)
    


    Epoch 5/30
    881/881 [==============================] - 2174s 2s/step - loss: 0.0114 - val_loss: 0.0134
    
    Epoch 00005: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_LAB_05.h5
    


    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_17.png)
    



    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_18.png)
    



    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_19.png)
    


    Epoch 6/30
    881/881 [==============================] - 2196s 2s/step - loss: 0.0111 - val_loss: 0.0276
    
    Epoch 00006: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_LAB_06.h5
    


    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_21.png)
    



    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_22.png)
    



    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_23.png)
    


    Epoch 7/30
    881/881 [==============================] - 2197s 2s/step - loss: 0.0111 - val_loss: 0.0064
    
    Epoch 00007: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_LAB_07.h5
    


    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_25.png)
    



    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_26.png)
    



    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_27.png)
    


    Epoch 8/30
    881/881 [==============================] - 2202s 2s/step - loss: 0.0110 - val_loss: 0.0067
    
    Epoch 00008: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_LAB_08.h5
    


    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_29.png)
    



    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_30.png)
    



    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_31.png)
    


    Epoch 9/30
    881/881 [==============================] - 2219s 3s/step - loss: 0.0112 - val_loss: 0.0072
    
    Epoch 00009: saving model to /content/drive/MyDrive/Messing Up With CNN/GRAY2RGB_LAB_09.h5
    


    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_33.png)
    



    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_34.png)
    



    
![png]({{site.url}}/assets/messing-with-cnn/l_ab/output_32_35.png)
    


    Epoch 10/30
    531/881 [=================>............] - ETA: 13:53 - loss: 0.0108


## Conclusions
* Using Grayscale to RGB seems to be non progressive because we will be messing up with entire pixels of image and hence it becomes harder to colorize after finding the valuable features.
* Using LAB seems to be more promising because, we will be only tuning the AB value of the image and the original grayscale image remains same. Also it is worth noting that, we only apply colorization on grayscae image.
* Image colorization is difficult task because it requires huge domain of images. If we trained a model which have many humans and we tried to colorize natural image then it surely will fail. 

## Ideas
Instead of using these dataset, If we read some movie's frame for some thousands and then train using it, then we might get rid of data requirement and also might get better training result.



## References
* [How to colorize black & white photos with just 100 lines of neural network code](https://emilwallner.medium.com/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d)



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


