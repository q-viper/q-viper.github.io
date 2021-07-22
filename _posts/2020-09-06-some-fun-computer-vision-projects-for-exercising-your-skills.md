---
title: Some Fun Computer Vision Projects For Exercising Your Skills
date: 2020-09-06T22:34:15+05:45
header:
  teaser: assets/wp-content/uploads/2020/02/20170625_184639-scaled.jpg
categories:
  - Computer Vision
  - Game Development
  - Programming
  - Project
tags:
  - computer vision
  - deep learning
  - Game Development
  - image processing
  - keras
  - ocr
  - projects
  - web development
---
**Contents**
* TOC
{:toc}

## Introduction
Hello everyone, once again due to the lockdown and threat of COVID-19 has allowed me to write a new blog because I am free. Anyone with little of Image Processing and some Deep Learning Concepts can understand this blog. I am trying to coverup some cool projects that can be fun to work, proudful to share, and insightful to the recruiters. Many have lost job, and many were about to hunt the job but all of a sudden we are holding ourselves back. The good thing is we have plenty of time so why not use it to make our career more strong? 

<i>
Are you a tech student trying to do something cool and share it to your friends? Are you an AI enthusiast trying to do some awesome projects? Are you a person who wants to enhance skills by doing projects? Are you trying to do something simple but brain chilling and awesome at the same time? Well you are here at the right place.
</i>

><b>So how can one find a job oppertunity that has never existed?</b>

This is possible to find a job oppertunity that has never existed by finding a right audience of your work. Do anything, any project and if you are able to share it to right audience then there is high chance that people will be interested to your work. I am still officially unemployed but I have got some offers on projects to work, I also gave one interview (waiting their response) after I published a blog [Gesture Based Visually Writing System: Adding Virtual Animationn, New Mode and New VUI]({{site.url}}/2020/08/14/gesture-based-visually-writing-system-adding-virtual-animation-new-mode-and-new-vui/). [This post](https://www.linkedin.com/posts/ramkrishna-acharya-91a217183_opencv-computervision-python-activity-6699919193124548608-bJI-) on LinkedIn has also gained huge popularity that I myself is still on shock.

### Credits
I would like to give credits to everyone who supported me on social media about my projects and readers like you.

### Motivation
The motivation due to which I am writing this blog is nothing more than thought of '**what if I did this**'.

## What now?
I am going to write down my ideas onto this blogs and I will also try to explain a simple procedure about completion of this projects. I will try to minimise the use of high requirements. 

## Contour Based Game: Break The Bricks
When I posted a blog [Gesture Based Visually Writing System: Adding Virtual Animationn, New Mode and New VUI]({{site.url}}/2020/08/14/gesture-based-visually-writing-system-adding-virtual-animation-new-mode-and-new-vui/), I got high attention from lots of tech people. Then I did something more cool. I was learning Game Development using Unity and C#, then I had to pause learning because of pandemic and now I wrote a simple game on Python. It is not that cool but you can check it on below link.
* [Contour Based Game: Break The Bricks]({{site.url}}/2020/08/16/contour-based-game-break-the-bricks/)

I have introuduced a simple way of making a game that can be played by moving finger in front of the camera. Please follow the link above to know more about how to write a python code for this.

### What next?
The system I wrote by publish this day has very little features. You can do anything from below to make it your own customized and awesome project.
* Add a UI menu to do new game, restart, pause etc.
* Add a score saving system using some file.
* Add a powers, like when hitting some bricks, randowmly drop the powers and when a pad hits those powers, activate them.
    * We need to take care of multiple objects at the same time, it will be tricky.

## Contour Based Writing
This is the system where I have been working since month. I have started from a simple idea and then features and then the deployment on the latest version. View them serially so that you can learn how an idea was implemented onto the system.
* [Gesture Based Visually Writing System Using OpenCV and Python]({{site.url}}/2020/08/01/gesture-based-visually-writing-system-using-opencv-and-python/)
* [Gesture Based Visually Writing System: Adding Visual User Interface]({{site.url}}/2020/08/11/gesture-based-visually-writing-system-make-a-visual-user-interface/)
* [Gesture Based Visually Writing System: Adding Virtual Animationn, New Mode and New VUI]({{site.url}}/2020/08/14/gesture-based-visually-writing-system-adding-virtual-animation-new-mode-and-new-vui/)
* [Gesture Based Visually Writing System: Add Slider, More Colors and Optimized OOP code]({{site.url}}/2020/08/21/gesture-based-visually-writing-system-add-slider-more-colors-and-optimized-code/)
* [Gesture Based Visually Writing System: A Web App]({{site.url}}/2020/08/29/gesture-based-visually-writing-system-web-app/)

### What next?
* Try to add a different mode, a write and a draw. 
    * On Draw mode, use a different model to perform auto completion of the drawing using some state of the art model.
    * On Write mode, use a exsisting system.
* Try to add a dialouge box for some operations like restart, exit.
    * Creating a simple but new Dialouge UI might come to aid. Then we can inherit it or attach it to other classes.
* Try to use sign language. 
    * Currently, the system is runnable by anything but using a simple classifier to recognise hand on the contour region and only then perform the operation. 
    * A simple idea can be, use the fist to move mode, use the open hand to erase and a pointing finger to write, thumb up to save, thumb down to clear all etc.

## Devanagari Handwritten Character Detection
I did this project with my friends while I was on 7th semester of my BSC. This system is very old and I have not much time to improve it. But good thing is I have tried to write a blog where I have explained a system. Also my supportive friends made a documentation awesome. Follow the blow link for that project.
* [Simple OCR for Devanagari Handwritten Text]({{site.url}}/2020/02/25/building-ocr-for-devanagari-handwritten-character/)

### What next?
* I have used keras for CNN and gianed 99.29% accuracy, but model itself is not trained by generators so training a model using generators will help to improve the overfitting of model.
* The system is not runnable on RealTime because of the overhead of performing a detection task. One best solution to decrease a overhead will be to use the OOP and refactor the code.
* The system only detects the simple words hence <b>some new idea</b> must be used.
* Combine model with different NLP models to make another awesome system.
* Deploy it!

## Convolve Me
This is just an idea I am presenting and I have also started to work on this one one Unity3d. The idea is to write an image editor for those who wants to play with filters. 
* User can choose image or capture it using system camera.
* User can use own customized filters like Sobel filters. 
* user can view, save this filtered image.

Customized filters means we can use any real values on the filter. And filter shape can also vary. If you are new to convolution or want to learn more about it then please follow my blogpost where I described how can one make a simple image processing class to perform convolution and other stuffs.
* [Writing a Simple Image Processing Class from Scratch]({{site.url}}/2020/05/30/image-processing-class-from-scratch-on-python/)

Deploying this system on phone might be a great attraction. 

## Face Swap
I think there are already some systems that can perform this task. But why not try it on our own? Use face harcascades to find faces on the image and then if there are multiple faces, swap them randomly. This won't be any useful system but will surely be fun to work with. On addition we can also do eyes swap, mouth swap etc. But problem lies when we are swapping faces of variable shape. On those case resizing an swapping image according to place where to swap will be best idea.

I am working on it on few days. Because I might want to show pranks to my sister.

## Virtual Move
Can I move something on live frame using hand gestures? A simple idea can be to place an object in front of an camera for a while and then move it when user grabs it and moves.


### How might it be possible?
* Take an running average of steady and clear frames for some seconds. 
* Make a model that classifies the hand region.
* When a hand is moved, take contours and find classification results.
* Place an object on frame. Now its contours is also shown.
* With positive classification work below:
    * Classify hand's mode (grab, release, move).
    * On grab mode:
        * Take a contour of object and crop that portion of frame. And replace that region by background image.
        * Whenever a hand is moved, move that object accordingly but leave the object's original place as background color.
    * On move mode:
        * Replace the object on position where hand is moving. It will make look like moving effect.
        
I think this project might be highly fun to work with. I am surely giving it a try on next blog.

## Contour Based File System
Few days ago while I was cooking food on fire, suddenly I got an idea to do something crazy using Contour. The idea for this system is to make a simple file system that can be controlled by Contours. As you have found that I made a VUI on [previous systems](#Contour-Based-Writing), it will be helpful to take ideas from there too. This task is challanging and time consuming. It might take week for me to make a simple system and I am bored to do this but I will share a simple concept to make it.

* Make some icons, for folder, for file, for back, for exit button.
* Make an initial empty main and files window and attach exit button and some rectangular spaces to show navigation bars.
* Set current directory to current working directory on back. Set previous directory to the root directory.
* When count of mode is maximum on :
    * Back icon: Erase current files window and make current directory to previous directory. And previous to root.
    * Exit icon: Exit the system.
    * Folder icon: Set current directory to directory above which pointer lies.
    * File icon: Do show on seperate window if it is image.


The ideas comes and I can not sleep but when I start to code it, pain starts. I am not going to work on this system soon. So if someone tries to make this system I would praise work and support from my heart.

## Contour Based Image Editor
Image Editor using Contour? Yes if I can write something using contour, if I can play games using it, if can make a file system using it then why not image editor? For initial stage I am assumming that image is read by default. But for further improvement, [Contour Based File System](#Contour-Based-File-System) will be useful.
* Initial modes are: crop, move, paint.
* Prepare VUI Just like [Contour Based Method]()
* Initially make a mode ROI on frame where we will use Gestures to change mode. 
    * Scissor like hand for crop pointer.
    * Rock like hand to move pointer above image.
    * Paper like hand for paint over image.
    * Single finger pointer to do everything.

## Face Recognition Based Login System
There are already plenty of codes on Github to perform this task but I am sharing something different and easier to use. I thing flask will be the best one to use here. A simple idea can be like below.
* On Signup, take 3 photos of user's face. Front, left and right side.
* Save those images on local storage and name it userid_front for front of current userid.
* While sign up, make a region where a live camera feed is shown and user's face is rendered. 
* When user clicks on capture button, take that photo from frontend to backend and take images of this userid. 
    * If this userid is not present, then show error.
    * If the face didn't match, then show error else proceed to next page and do additional things.
    
For face comparision and other face recognition operations, we can use `face_recognition` library and it is available on PyPI.

## Finally
Thank you very much for reading this blog and I hope to get feedbacks. Find me on Twitter as [QuassarianViper](https://twitter.com/QuassarianViper) and on LinkedIn as [Ramkrishna Acharya](https://www.linkedin.com/in/ramkrishna-acharya-91a217183/). On the mean time why not read more of my blogs?

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


