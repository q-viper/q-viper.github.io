---
title: Gesture Based Visually Writing System Using OpenCV and Python
date: 2020-08-01T20:18:30+05:45
header:
  teaser: assets/wp-content/uploads/2020/08/gesture.png
categories:
  - Computer Vision
  - Programming
  - Project
tags:
  - computer vision
  - gesture recognition
  - image processing
  - opencv
  - python
---
# Gesture Based Writing/Drawing Using OpenCv Python
**Contents**
* TOC
{:toc}

# Theory Part

This is the first part of [Gesture Based Visually Writing System](https://github.com/q-viper/Contour-Based-Writing/).
* [Gesture Based Visually Writing System Using OpenCV and Python]({{site.url}}/2020/08/01/gesture-based-visually-writing-system-using-opencv-and-python/)
* [Gesture Based Visually Writing System: Adding Visual User Interface]({{site.url}}/2020/08/11/gesture-based-visually-writing-system-make-a-visual-user-interface/)
* [Gesture Based Visually Writing System: Adding Virtual Animationn, New Mode and New VUI]({{site.url}}/2020/08/14/gesture-based-visually-writing-system-adding-virtual-animation-new-mode-and-new-vui/)
* [Gesture Based Visually Writing System: Add Slider, More Colors and Optimized OOP code]({{site.url}}/2020/08/21/gesture-based-visually-writing-system-add-slider-more-colors-and-optimized-code/)
* [Gesture Based Visually Writing System: Building a Web App Using Flask]({{site.url}}/2020/08/28/gesture-based-visually-writing-system-web-app/)

## Before Starting
Huge credit goes to the pandemic COVID19 and without it, I would have not been another unemployed and the thoughts of writing this blog won't have appear. Also the climate change, due to which I usually have blackouts.

## Motivations
We have seen decades of AI revolutions and from Deep Blue chess to Dota playing agents. We are on the competition of achievements. Years ago, there was HOG methods and there was nothing like YOLO or RCNN and the world of Computer Vision today is hungry to find new methods. There are many researchers and academics working continously to make more AI based systems and the world of Cloud Services has granted huge advantages. The world of Open Source has been so great that we can almost get everything's code. Now the program can complete our sentence, drawings, can itself talk and many more. The list goes on.

## How come?
<strong>Lets describe how am I writing this article now?</strong>

I was stuck at village with no internet and electricity and I was planning on Region Based method for Corn Infection detection then I started to write using paper and pen. Please follow below links about how am I doing(or will be did?) it. The thought of <b>what if?</b> came. What if we can just move our hand on air and we can actually be able to write something? Then rest is on this blog.

I am doing below tasks currently but some unpredicted electricity problems are affecting me. 
* [Corn Infection Classification](#): Took 3k+ corn images and doing classification using some standard models.
* [Corn Infection Detection](#): Using Region Based Methods to succesfully detect infected parts of corn leaves.
* [A Chatbot Applications using Unity and RASA](#): Using Chatbot Framework [RASA](https://www.rasa.com) and [Unity3d](https://www.unity3d.com) to make GUI based application.
* [RNN from Scratch](#): Writing own RNN library from scratch.
* [A Simple Autoencoder From Scratch](#): Using FFNN from scratch to create Autoencoder.

## Inspiration
This blog post is inspired by the following articles and I have copied some contents also, so I am giving credits to the authors. 
* [Hand Gesture Recognition using Python and OpenCv](https://gogul.dev/software/hand-gesture-recognition-p1)
    The author of this article has made gesture recognition so simple that made me complete the drawing system within few hours. Honestly the part of hand contour extraction is inspired from this blog.


## Prerequisites
For understanding the things going one here, basic Image Processing concepts can work well. For coding part, understanding of NumPy, Matplotlib and OpenCv will be enough. For NumPy and Matplotlib, you can always checkout [this awesome repo](#https://www.github.com/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials) of Mr. Tarry Singh. And for OpenCv, documentation is great and don't forget to checkout [PyImagesearch](#https://www.pyimagesearch.com). Also the book <b>Image Operators Image Processing in Python by Jason M. Kinser</b> is great book to understand Image Processing from scratch. This book takes guide from basic image formation to advanced topics like segmentation.

## Introduction
Computer Vision has been the area of interest from years. Different competitions and has given us new concepts and ideas. Also there has been lots of work on Gesture Recognition and Sign Language Recognition. The Gesture Recognition is not a easy task because of the complexity of image formation and the noise on image. The extraction of hand from the image is another problem. There has been various researchs to do segmentation of image and one method to do so is `selective search`(Used on RCNN Family and I using this method to do Region Proposal on another blogs.)

## Problem Statement
We need to extract the hand gesture from the image first and then checkout for the movement of the hand. (For simplicity I am making this blog work for any object movements. But for advanced and more robustness, we must recognise hand and do draw only then.) The problem is so simple that it can be divided onto below steps:-
1. Read frame from image and extract hand region. 
2. Draw pointer on the top of hand region and draw the pointer on canvas along with the pixels.
3. When the character is drawn on the canvas, send the character to classification model.

The objective of this blog is to succesfully create a `Gesture Based Writing System` while explaining all major tasks on the way.
Also checkout for the bonus topics(there is always some interesting going on).

# Segmentation of Hand Region
For our this particular problem, the hand will be the foreground and the other remaining things are background. Our task will be to find the exact position of the hand. But how can one do that? The simple but tricky concept here is known as <b>Background Subtraction</b>. OpenCV's fast and powerful functionalities allows us to do this operation on realistic scenario. The background subtraction is done for each sequence of frames.


# Background Subraction
The concept of background subtraction is so simple, we take a image which have only background, then take a image with hand over same background image. Then subtract those two images, what will you get? Lets see by code in action.
* First define a good show method.
* And run camera, capture foreground/background images
* Show difference.

```python
def show(img, figsize=(10, 10)):
    figure = plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

cap = cv2.VideoCapture(0) #run camera with device 
imgs = [] #list to hold frames
while len(imgs) < 2: #while we captured one foreground and one background image
    ret, frame = cap.read() #read each frame
    frame = cv2.flip(frame, 1) #flip the frame for anti-mirror effect
    key = cv2.waitKey(2) #wait for 2sec to check user key strike    
    if key == 32: #if key is space then show the frame, append it to our list
        show(frame)
        imgs.append(frame)
    cv2.imshow('Running Frame', frame) #keep showing our current frame 
    if key == 27 & 0xFF: #when escape key is pressed, exit
        break
cap.release() #release the camera and destroy windows
cv2.destroyAllWindows()
show(imgs[0]-imgs[1]) #background subtraction
```

The result of this code will not be great but lets see them on image.
![image-title-here]({{site.url}}/assets/wp-content/uploads/2020/08/background-1024x418.png){:class="img-responsive"}


The only difference I am using on these image is the pamplet on the wall. The final difference image doesn't look like any difference but some noises. This is because of the image quality and noise effect, we can always remove these noise by some factor later. OpenCV reads image on BGR format and it still is on BGR. The difference on image shows that the object on the wall was not on background image but is present on new image hence its color will be different than others and it will be easy to recognize them. Hence the object is foreground and the result will be called the mask.

For better performance, we will use concept of running average. We will run our system idle for some frames and then take average from it. We will make an average background. Then will do the absolute subtract. Also using RGB/BGR image on background subtraction only increase the complexity, so we will use the Grayscale image or ROI.



# Mask Generation
The mask is something that holds the foreground. In general, the difference image on above image also contains some sort of mask on the place where the object is present. The mask is generated by doing absolute subtraction of current frame with average background and then doing thresholding. The actual mask will contain black on unwanted region and white on our hand region. This is pretty simple idea, we are subtracting background with background image that contains hand, the difference will only be on hand region and all other parts will be black(RGB (0, 0, 0)).
> Thresholding is the concept where pixels are grouped onto one of two groups by comparing pixels with thresholding value.
> There are various thresholding, most simple is Binary and others are Otsu, Inverse Binary etc.
Lets see the code on action.

```python
diff = np.abs(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)-cv2.cvtColor(imgs[1], cv2.COLOR_BGR2GRAY))
_, th = cv2.threshold(diff, 150, 255, cv2.THRESH_BINARY)
show(th)
```

Thresholding the difference of gray images will give us mask. The value will be 0 if it is less than 150 and will be 255 if it is greater than 150. The result is great but not as we expect it to be.

![image-title-here]({{site.url}}/assets/wp-content/uploads/2020/08/difference_gray.png){:class="img-responsive"}


The mask is generated for each and every frame(except averaging frames).

# Contour Extraction
The mask on the above image looks weird because it is full of noise and my room is not that lighty. Lesser the noise, easier the task becomes. The contour is the outer region where a object is located. On our case the contour must be around the hand. But because of the noise, we might have multiple contours and also can we hold our still? Also the shadows can effect. The contour with largest area is our hand region.
>Contour is the outline or bounding curves around hte boundary of an object located on an image.

Lets find the contour on above image, can we find the region where the poster is placed?

```python
(cnts, _) = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
segmented = max(cnts, key=cv2.contourArea)
cv2.drawContours(imgs[0], [segmented], -1, (0, 0, 255))
show(imgs[0])
```

![image-title-here]({{site.url}}/assets/wp-content/uploads/2020/08/contour.png){:class="img-responsive"}

Well, so far so goood. Please ignore the color channel on this image because I am using BGR color space. But we did pretty great job or OpenCV did?


# Writing Canvas Generation
The canvas is something where we can write here. So for simplicity, it will be like a whiteboard. But in image processing, it will be black image. 

<b>Why black image? </b>

The pixel values of black is `[0, 0, 0]` so we can do addition without affecting any pixels. The addition here will be what we call writing. Complete black image is like `addition identity` image. Thanks to NumPy we can make complete black image into white just by adding 255.

The basic idea behind writing is to find the top of our contour. Draw a pointer on that position on frame(it will be like we are seeing where our hand is moving). And on the canvas, also make a pointer on same position, change the color on the place where pointer is present. The image we will use for operations is black but what we will show is white so we will do some weird operations.  Also using OpenCV we can control camera with our keys, so we can make different modes of pointer. For simplicity, I am making some modes:
* <b>Write</b>:- It will add pixels on the canvas.
* <b>Erase</b>:- It will remove the written pixels from the canvas. It is tricky but works fine. Since our background is white, we will make every pixel white where the pointer resides on this mode. 
* <b>Idle</b>:- Neither write nor erase.
* <b>Erase All</b>:- Erases the canvas to make is clean.
* <b>Restart All</b>:- Erases the canvas along with the average image. New Average is taken.


# Implementation

## Background Preparation
We will start by assuming what the background will be for our case. We will do running average for up to some frames and only then do other processes.

```python
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
def running_average(bg_img, image, aweight):
    if bg_img is None:
        bg_img = image.copy().astype("float")
    else:
        cv2.accumulateWeighted(image, bg_img, aweight)    
    return bg_img
```

A mathematical formula is:-
\begin{equation}\label{eq:}
dst(x, y) = (1-\alpha) * dst(x,y) + \alpha *  src(x, y)
\end{equation}

* `dst` is destination image or accumulator
* `src` is source image
* `𝛼` is the weight of the input image and decides the speed of updating. If set small, average is taken over large number of frames.

Here source is `image` and destination is `bg_img`.

## Preparing Parameters and Variables
```python
aweight = 0.5 #accumulate weight variable
cam = cv2.VideoCapture(0) # strat the device camera
top, right, bottom, left = 250, 400, 480, 640 # ROI box
num_frames=0 # count frame
canvas = None # writing canvas
t=3 # thickness
draw_color = (0, 255, 0) # draw color(ink color)
pointer_color = (255, 0, 0) # pointer color
erase = False # mode flag
take_average=True # flag to indicate take average
bg_img=None #bg image
```

Please follow the comment written right to the code line for explanation.

## Running the Camera
```python
while True: # loop while everything is true
    (ret, frame) = cam.read() # read the camera result    
    if ret: # if camera has read frame        
        key = cv2.waitKey(1) & 0xFF # wait for 1ms to key press
        frame = imutils.resize(frame, width=700)        
        frame = cv2.flip(frame, 1) # flip to remove mirror effect        
        clone = frame.copy() # clone it to not mess with real frame
        h, w = frame.shape[:2]        
        roi = frame[top:bottom, right:left] # take roi, to send it onto contour/average extraction        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) # roi to grayscale        
        gray = cv2.GaussianBlur(gray, (7, 7), 0) # add GaussianBlur to eliminate some noise
```

Since we have chosed our device camera from `cv2.VideoCapture(0)`, we will read the frames next. When the camera is not working properly, the `ret` becomes `False`. Also we want our system to continously check for user input, so we will do `cv2.waitKey(1) & 0xFF`. The original frames will have to be flipped in order to make it realistic. We can do that by flipping on x axis using `cv2.flip(frame, 1)`. Then we crop the ROI from given box and convert it into Grayscale. Grayscale is much more efficient on doing thresholding processes. Then we added some Gaussian blur of shape (7, 7). Some kind of Convolution Operation happens inside that function. If you want to learn about Convolution operation from Scratch then follow below link.
* [Convolutional Neural Networks from Scratch on Python]({{site.url}}/2020/06/05/convolutional-neural-networks-from-scratch-on-python/)



## Take Average of ROI Background

```python
        if num_frames < 100 and take_average == True: # if to take average and num frames on average taking is lesser 
            bg_img = running_average(bg_img, gray, aweight) # perform running average            
            cv2.putText(clone, str(num_frames), (100, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 5) # put frame number on frame
            num_frames += 1        
        else: # if not to take average
            num_frames = 0                                    
            hand = get_contours(bg_img, gray) # take our segmented hand
            take_average=False        
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2) # draw a ROI        
        cv2.imshow("Feed", clone) # show live feed        
        if key==27: # if pressed  escape, loop out and stop processing
            break
cam.release()
cv2.destroyAllWindows()
```

Next, we check the flag if we had to take average or not, if we do want to find average still, then put frame number on the frame. Else we get contours from our image. On final, we draw a rectangle as ROI on live feed and if user enters escape key then stop the process.

## Get Hand Contours
Next we do find our contours on the current frame, this is done by taking absolute difference between average background image and current frame, then doing thresholding.

```python
def get_contours(bg_img, image, threshold=10):   
    diff = cv2.absdiff(bg_img.astype("uint8"), image) # abs diff betn img and bg
    _, th = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    (cnts, _) = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    if len(cnts) == 0:
        return None
    else:
        max_cnt = max(cnts, key=cv2.contourArea)
        return th, max_cnt
```

The function takes the background image, current frame and the thresholding value. It returns the thresholded image and the largest contour. 


## Drawing Contours and Showing Thresholded image
```python
    ........................
    ........................    
    else: # if not to take average
        num_frames = 0                               
        hand = get_contours(bg_img, gray) # take our segmented hand
        take_average=False
        if hand is not None:
            thresholded, segmented = hand
            cv2.drawContours(clone, [segmented+(right,top)], -1, (0, 0, 255))    
            cv2.imshow("mask", thresholded)
```

The contours returned is from the ROI image which is the cropped version of original frame. Hence we have to add the starting position of ROI box to correctly draw contours on original frame. I hope the image just looks like below's.

![image-title-here]({{site.url}}/assets/wp-content/uploads/2020/08/contour-drawn-1024x706.png){:class="img-responsive"}


## Draw Pointer on Canvas
```python
    .......................
    .......................
    else: # if not to take average
        num_frames = 0                        
        hand = get_contours(bg_img, gray) # take our segmented hand
        take_average=False
        if hand is not None:
            thresholded, segmented = hand
            cv2.drawContours(clone, [segmented+(right,top)], -1, (0, 0, 255))    
            cv2.imshow("mask", thresholded)
            tshape = thresholded.shape
            sshape = segmented.shape
            new_segmented = segmented.reshape(sshape[0], sshape[-1])
            m = new_segmented.min(axis=0)
            if type(canvas) == type(None):
                canvas = np.zeros((tshape[0], tshape[1], 3))+255
            c = np.zeros(canvas.shape, dtype=np.uint8)
            cv2.circle(c, (m[0], m[1]), 5, pointer_color, -3)
            cv2.drawContours(clone, [segmented+(right,top)], -1, (0, 0, 255))        
            cv2.circle(clone, (right+m[0], top+m[1]), 5, pointer_color, -3)
```

We want to draw a pointer only inside the ROI box and not outside. And also the contours are found on the corpped ROI image hence we need to add right and top values. The writing `canvas` is first initialzed with zeros and then added 255, thanks to NumPy array broadcasting, hence canvas became complete white. The variable `c` here is very important one and it is responsible to show the pointer. The pointer window is equal to the canvas, and pointer image is 0 on every place except where the minimum position of contour is. 

### Lets recap:-
* The frame is read and the contours are calculated by taking absolute difference of current frame with average background image.
* The contour is drawn on live feed. 
* Thresholed image is show.
* The pointer(circle) is drawn on live feed. 
* The pointer image, canvas image is prepared.

The pointer should move along with fingers. Please check the code below if your system is not running well.

```python
def running_average(bg_img, image, aweight):
    if bg_img is None:
        bg_img = image.copy().astype("float")
    else:
        cv2.accumulateWeighted(image, bg_img, aweight)    
    return bg_img
def get_contours(bg_img, image, threshold=10):    
    diff = cv2.absdiff(bg_img.astype("uint8"), image)    # abs diff betn img and bg    
    _, th = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    (cnts, _) = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    if len(cnts) == 0:
        return None
    else:
        max_cnt = max(cnts, key=cv2.contourArea)
        return th, max_cnt
aweight = 0.5 #accumulate weight variable
cam = cv2.VideoCapture(0) # strat the camera
top, right, bottom, left = 250, 400, 480, 640 # ROI box
num_frames=0 # count frame
canvas = None # writing canvas
t=3 # thickness
draw_color = (0, 255, 0) # draw color(ink color)
pointer_color = (255, 0, 0) # pointer color
erase = False # mode flag
take_average=True # flag to indicate take average
bg_img=None #bg image
while True: # loop while everything is true  
    (ret, frame) = cam.read() # read the camera result    
    if ret: # if camera has read frame        
        key = cv2.waitKey(1) & 0xFF # wait for 1ms to key press
        frame = imutils.resize(frame, width=700)        
        frame = cv2.flip(frame, 1) # flip to remove mirror effect        
        clone = frame.copy() # clone it to not mess with real frame
        h, w = frame.shape[:2]        
        roi = frame[top:bottom, right:left] # take roi, to send it onto contour/average extraction       
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # roi to grayscale        
        gray = cv2.GaussianBlur(gray, (7, 7), 0) # add GaussianBlur to eliminate some noise  
    
        if num_frames < 100 and take_average == True: # if to take average and num frames on average taking is lesser 
            bg_img = running_average(bg_img, gray, aweight) # perform running average            
            cv2.putText(clone, str(num_frames), (100, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 5) # put frame number on frame
            num_frames += 1        
        else: # if not to take average
            num_frames=0                                   
            hand = get_contours(bg_img, gray)  # take our segmented hand
            take_average=False
            if hand is not None:
                thresholded, segmented = hand
                cv2.drawContours(clone, [segmented+(right,top)], -1, (0, 0, 255))                   
                tshape = thresholded.shape
                sshape = segmented.shape
                new_segmented = segmented.reshape(sshape[0], sshape[-1])
                m = new_segmented.min(axis=0)  

                if type(canvas) == type(None):
                    canvas = np.zeros((tshape[0], tshape[1], 3))+255                
                c = np.zeros(canvas.shape, dtype=np.uint8)
                cv2.circle(c, (m[0], m[1]), 5, pointer_color, -3)
                cv2.circle(clone, (right+m[0], top+m[1]), 5, pointer_color, -3)   
                cv2.imshow("mask", thresholded)
                cv2.imshow("c", c)        
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2) # draw a ROI        
        cv2.imshow("Feed", clone) # show live feed
        if key == 32:
            cv2.imwrite("Captured.png", clone)        
        if key==27: # if pressed  escape, loop out and stop processing
            break
cam.release()
cv2.destroyAllWindows()
```


## Draw on Canvas
```python
                ...................................................
                ...................................................
                cv2.circle(c, (m[0], m[1]), 5, pointer_color, -3)
                cv2.circle(clone, (right+m[0], top+m[1]), 5, pointer_color, -3)   
                cv2.imshow("mask", thresholded)
                cv2.imshow("c", c)                
                cv2.circle(canvas, (m[0], m[1]), 5, draw_color, -3)
                e = cv2.erode(canvas, (3, 3), iterations=5)
                drawn = cv2.resize(e, (clone.shape[1], clone.shape[0]))
                c = cv2.resize(c, (clone.shape[1], clone.shape[0]))
                drawn_new = drawn+c
                cv2.imshow("Drawing", drawn+c)
```

<b>What is happenning here?</b>

Well, on canvas, on the position where the minimum position of contour is present, we draw a circle of given thickness with different color than pointer. The canvas, pointer, thresholded, ROI image all will be of same size but we want it to be just same as our live feed, hence we resize it. But why doing erosion? Erosion is used here to make image lesser noisy or more thick around drawn circle. The canvas here is original canvas. And we don't do anything than writing on it. But we resize the erosion result. Then to show the drawn color and pointer at the same time on canvas, we will sum the canvas with pointer image. The beauty of black image summing with any image is so awesome here.

![image-title-here]({{site.url}}/assets/wp-content/uploads/2020/08/writing-pointer-1-1024x706.png){:class="img-responsive"}


# Adding Modes
Befor going onto modes, lets first define our keys to do modes and other operations.
* `escape`:- Exit from system.
* `space`:- Save current drawing(excluding pointer).
* `z`:- Idle mode(move only pointer).
* `x`:- Erase mode.
* `c`:- Color mode.
* `e`:- Erase current canvas but leave the average image.
* `r`:- Restart the average and canvas.

Lets make basic assumptions:-
* If we want to erase, then the position where current pointer lies must be made white on canvas. 
* If we want to move, then the position where current pointer lies must be left unchanged on canvas.
* Also make the pointer color change on ever modes.

## Preparing Keys

```python
            ...........................
            ..........................
            if hand is not None:
                if chr(key) == "x": # if pressed x, erase
                    draw_color = (255, 255, 255)
                    pointer_color = (0, 0, 255)
                    erase = True
                if chr(key) == "c":
                    draw_color = (0, 255, 0)
                    pointer_color = (255, 0, 0)
                    erase = False
                if chr(key) == "z": #idle
                    erase = None
                    pointer_color = (0, 0, 0)                   
                if chr(key) == "r": # restart system
                    take_average=True
                    canvas = None
                if chr(key) == "e":
                    canvas = None
                    drawn = np.zeros(drawn.shape)+255
```

On above, we are using flag `erase`, to do 3 tasks. The pointer color must also be changed per mode.



## Erase mode
```python
                ..........................................
                ...........................................
                c = np.zeros(canvas.shape, dtype=np.uint8)
                cv2.circle(c, (m[0], m[1]), 5, pointer_color, -3)
                cv2.circle(clone, (right+m[0], top+m[1]), 5, pointer_color, -3)
                if erase==True:
                    cv2.circle(canvas, (m[0], m[1]), 5, draw_color, -3)
                    erimg=cv2.circle(canvas.copy(), (m[0], m[1]), 5, (0, 0, 0), -3)            
                    cv2.circle(c, (m[0], m[1]), 5, (0, 0, 255), -3)
                    e = cv2.erode(erimg, (3, 3), iterations=5)
                    drawn = cv2.resize(e, (clone.shape[1], clone.shape[0]))
                    c = cv2.resize(c, (clone.shape[1], clone.shape[0]))
                    drawn_new = drawn+c
                    cv2.imshow("Drawing", drawn+c)
                    erimg=cv2.circle(canvas.copy(), (m[0], m[1]), 5, (255, 255, 255), -3)            
                    cv2.circle(c, (m[0], m[1]), 5, (0, 0, 255), -3)
                    e = cv2.erode(erimg, (3, 3), iterations=5)
                    drawn = cv2.resize(e, (clone.shape[1], clone.shape[0]))                    
```

I have scratched my head for hours to make this work right. I hope somebody finds the better way to do this. 
* First draw a white color on pointer position on canvas.
* Draw a black color on pointer position on copy of current canvas.
* Take copy image and erode it, resize it and sum it with pointer image.(if the black color is not used then pointer will not be visible)
* And undo the color operation else the pointer position remains on new images.


## Idle Mode
```python
                .....................................................
                .....................................................
                if erase==True:
                    cv2.circle(canvas, (m[0], m[1]), 5, draw_color, -3)
                    erimg=cv2.circle(canvas.copy(), (m[0], m[1]), 5, (0, 0, 0), -3)            
                    cv2.circle(c, (m[0], m[1]), 5, (0, 0, 255), -3)
                    e = cv2.erode(erimg, (3, 3), iterations=5)
                    drawn = cv2.resize(e, (clone.shape[1], clone.shape[0]))
                    c = cv2.resize(c, (clone.shape[1], clone.shape[0]))
                    drawn_new = drawn+c
                    cv2.imshow("Drawing", drawn+c)
                    erimg=cv2.circle(canvas.copy(), (m[0], m[1]), 5, (255, 255, 255), -3)            
                    cv2.circle(c, (m[0], m[1]), 5, (0, 0, 255), -3)
                    e = cv2.erode(erimg, (3, 3), iterations=5)
                    drawn = cv2.resize(e, (clone.shape[1], clone.shape[0]))                    
                elif erase==False:
                    cv2.circle(canvas, (m[0], m[1]), 5, draw_color, -3)
                    e = cv2.erode(canvas, (3, 3), iterations=5)
                    drawn = cv2.resize(e, (clone.shape[1], clone.shape[0]))
                    c = cv2.resize(c, (clone.shape[1], clone.shape[0]))
                    drawn_new = drawn+c
                    cv2.imshow("Drawing", drawn+c)
                elif erase == None:
                    canvas_shape = canvas.shape
                    clone_shape = clone.shape
                    eshape = (clone_shape[0]/canvas_shape[0], clone_shape[1]/canvas_shape[1])
                    m[0] = int(eshape[1]*m[0])
                    m[1] = int(eshape[0]*m[1])
                    drawn = cv2.resize(drawn, (clone.shape[1], clone.shape[0]))
                    dc = drawn.copy()  
                    cv2.circle(dc, (m[0], m[1]), 10, pointer_color, -3)
                    cv2.imshow("Drawing", dc)
```

* When `x` key is pressed, erase is true and first condition runs.
* When `c` key is pressed, erase is fale and second condition runs.
* When `z` key is pressed, erase is None and 3rd condition runs.

On Idle mode,  we take the previous `drawn` image, which is the resized version. we also did some resizing there. On simple way, we take the copy of current version of the canvas and draw a pointer above it just to show movement like feature. The showing of movement is just a fooling idea. In fact this overall operation is fooling idea.


## Restart mode
When the key `r` is pressed, `take_average` becomes true and the averaging strats.

## Erase All mode
When the key `e` is pressed, `drawn` is made white. And canvas is restarted.

# Bonus Part
I am going to do Character Recognition on the characters written on the canvas. For that, I will be using Devanagari Handwritten dataset. For simple Devanagari Handwritten character, I have wrote a Detection/Localization or OCR like system years ago using NumPy, Keras and OpenCv. Please follow the bellow blogpost for full contents. The codes and algorithms are not good and I have not got time to improve it. But I used imagination on the coding so years after I still feel proud of that work.
* [Devanagari Handwritten Characters/word Detection](#)
* [GitHub Repository](#)

On that repository, contatins a file `recognition.py`, which is responsible for lots of things. So, here I am also using same that file, and it contains the module recognition inside it. Importantly, I am naming that folder as dcr. While doing some works, I am encountering some errors. Those errors are newbie errors telling about not finding saved model directory. Looking into `predictor.py` file and doing some path correction solves the problem. In my case, I did:-

```python
.......................
.......................
def prediction(img):
    curr_dir = os.getcwd()+"/dcr/"
    json_file = open(curr_dir+'cnn2/cnn2.json', 'r')
    loaded_model_json = json_file.read()     # load json and create model
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)  
    loaded_model.load_weights(curr_dir+"cnn2/cnn2.h5") # load weights into new model
    ..................................
    ..................................
```

Next let our system fire recognition while pressing `s` key. 

```python
...................................
...................................
                    cv2.imshow("Drawing", dc)
        if chr(key) == "s":
            show(drawn)
            d = drawn.copy().astype(np.uint8) 
            r = recognition(cv2.cvtColor(d, cv2.COLOR_BGR2GRAY), 'show')
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2) # draw a ROI
        cv2.imshow("Feed", clone) # show live feed 
```

![image-title-here]({{site.url}}/assets/wp-content/uploads/2020/08/detection-1024x385.png){:class="img-responsive"}

The porblem with combination of DCR(Devanagari Character Recognition) and Gesture Writing is only that, it doesn't quite works with words. The reason is the thickness of our writing. So in near future I will work on that.

# Where From Here?
* What about combining this with some LSTMS or RNN models to predict what is user's next draw might be?
* Adding a GUI like functionalities where we do hand extraction and then whenever we click on particular region on frame then fire some actions. Like when we show some GUI like area on camera, when a hand is taken above that area then we fire some events, like changing a color, drawing a line or playing a music etc.

Leave me comment for the entire codes.

# Lets Connect
* [Twitter](https://www.twitter.com/QuassarianViper)


