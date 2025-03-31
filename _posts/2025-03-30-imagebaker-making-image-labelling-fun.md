---
title:  ImageBaker - Making Image Labelling Fun
date:   2025-03-21 01:29:17 +0545
categories:
    - Python
    - ImageBaker
tags:
    - Computer Vision
    - PySide6
    - Image Annotation
header:
  teaser: assets/data_annotation/android-chrome-512x512.png
---

# ImageBaker - Making Image Labelling Fun

What is the most boring task in machine learning? As a software engineer focusing on computer vision who has shipped more than a dozen computer vision models in production, I would say image labeling. Yet, a machine learning engineer can't escape this essential task—their models depend on it. So how do we make it bearable? For me, I try to make it fun. And that's how **[ImageBaker](https://github.com/q-viper/image-baker/)** was born.

## The Data Labeling Challenge

When clients request an anomaly detection system based on camera feeds, we need curated, labeled datasets to train our algorithms. However, without examples of anomalies, we can't effectively detect them. After repeatedly facing the challenge of developing anomaly detection systems without adequate datasets, I created smaller tools to generate synthetic anomaly data. These experiments eventually led to ImageBaker—a comprehensive solution that I hope to use long-term and improve with community feedback.

## Garbage In, Garbage Out

The performance of any machine learning model depends heavily on the quality and quantity of the labeled data it's trained on. This is especially true for computer vision tasks. The process typically involves multiple cycles of labeling, training, and evaluation, making it time-consuming and often tedious.

What if we could generate multiple realistic labeled datasets from a single image? This approach could significantly reduce the time spent on manual labeling while maintaining the quality needed for effective model training.

## What's in a Name?

The concept behind ImageBaker involves extracting portions of an image (such as objects of interest) using tools like polygons or models such as Segment Anything. These extractions are treated as layers that can be copied, pasted, and manipulated to create multiple instances of the desired object.

By combining these layers step by step, we create new labeled images with annotations in JSON format. The term "baking" refers to the process of merging these layers into a single cohesive image—similar to how ingredients are combined and baked to create something new.

## Key Features of ImageBaker

![Demo of ImageBaker in action](https://github.com/q-viper/image-baker/blob/main/assets/demo.gif?raw=true)

*An example of baked images. (Each object is a layer, and annotations are automatically extracted for all layers.)*

### 1. Annotate with Ease
Load a folder of images and annotate them using bounding boxes or polygons. The intuitive interface makes labeling faster and more precise.

![](https://github.com/q-viper/image-baker/blob/main/assets/demo/annotation_page.png?raw=true)

*Annotation page*

### 2. Model Testing
Define models for detection, segmentation, and prompts (e.g., points or rectangles) by following the base model structure. This allows you to test how your models perform on different data variations.

In the same page shown in above, we can see on the bottom right corner `DummyDetectionModel` that is where we select the models we define and running in the backed. Upon hitting the **Predict** button, the prompts will be passed to the selected model along with the loaded image. Then the result from model will be annotated back to the application.

### 3. Layerify Images
Crop images based on annotations to create reusable layers. Each cropped image represents a single object that can be manipulated independently.

![](https://github.com/q-viper/image-baker/blob/main/assets/demo/baker_page.png?raw=true)

*A sample baker page.*

### 4. Bake Custom States
Arrange layers to create image variations by dragging, rotating, adjusting opacity, and more. Save these arrangements as states with a simple button click or keyboard shortcut.

Those states could be save by Save button or with shortcut **Control + S**.

We can also draw to a selected layer with brush.

![](https://github.com/q-viper/image-baker/blob/main/assets/demo/drawing.png?raw=true)

### 5. Export for Training
Export the final annotated JSON and baked multilayer images for use in training your computer vision models.

![](https://github.com/q-viper/image-baker/blob/main/assets/demo/annotated_veg_smiley.png?raw=true)

**A sample exported annotated image with fake leaves.**

## Powerful Shortcuts for Productivity

* **Ctrl + C**: Copy selected annotation/layer.
* **Ctrl + V**: Paste copied annotation/layer in its parent image/layer if it is currently open.
* **Delete**: Delete selected annotation/layer.
* **Left Click**: Select an annotation/layer on mouse position.
* **Left Click + Drag**: Drag a selected annotation/layer.
* **Double Left Click**: When using polygon annotation, completes the polygon.
* **Right Click**: Deselect an annotation/layer. While annotating the polygon, undo the last point.
* **Ctrl + Mouse Wheel**: Zoom In/Out on the mouse position, i.e., resize the viewport.
* **Ctrl + Drag**: If done on the background, the viewport is panned.
* **Ctrl + S**: Save State on Baker Tab.
* **Ctrl + D**: Draw Mode on Baker Tab. Drawing can happen on a selected or main layer.
* **Ctrl + E**: Erase Mode on Baker Tab.
* **Wheel**: Change the size of the drawing pointer.

The custom image generated can be tested within the application as well and we can see the performance of the model. If the model does not predict better results, we can retrain the model with that image.

I have also made a video about the project in YouTube and can be viewed below.
<iframe width="560" height="315" src="https://www.youtube.com/embed/WckMT0r-2Lc" title="ImageBaker - Making Image Labelling Fun" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

I made this application in a weekend and hence could still contain bugs. To make it ease of use, the project is Open Source and I am hoping that more people will find it usefule and the app can be more stable.