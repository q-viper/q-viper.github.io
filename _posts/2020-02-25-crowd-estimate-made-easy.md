---
title: Crowd Counting Made Easy
date: 2020-02-25T05:45:43+05:45
comments_id: 8
header:
  teaser: assets/wp-content/uploads/2020/02/download.jpg
categories:
  - Artificial Intelligence
  - Computer Vision
  - Machine Learning
tags:
  - artificial intelligence
  - computer vision
  - crowd counting
  - people counting
---

**Contents**
* TOC
{:toc}


<!-- wp:paragraph -->
<p><strong>Experience Of Being Udacity Scholar:</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>As a scholar of <strong>Udacity’s Secure and Private AI Challenge</strong> course by Facebook, I have not only learned about the Pytorch and Pysyft from the scratch but I also learned how it feels to be a Udacity scholar. First of all, the Slack channel is awesome and everyone is eager to learn and help each other. I was engaged on multiple study groups and every group has very helpful members and I learned how to work on group and contribute on project. I would like to give huge thank to Udacity and all the Slack members for this wonderful opportunity. While attending this course, I also started the popular trend of programming <em>#100DaysOfCode</em>. And I have created an entire collection of GitHub repository of this. Here is a <a href="https://github.com/q-viper/SPAIC/blob/master/README.md" rel="noreferrer noopener" target="_blank">link</a>.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3><strong>Crowd Counting Made&nbsp;Easy</strong></h3>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>The problem of crowd counting has been studied for research purpose for quite some time. For too long, countless precious lives have been lost in stampedes. In countries with huge festivals, crowd management is a serious issue. One popular festival is ‘Kumbh Mela’, in India. It may be almost impossible to control the crowds in such scenarios, but the respective authorities can always take the precautionary controls before risk occurs. Human beings do not have the ability to count people in crowds, but we can make intelligent machines and programs that can do it easily. Artificial Intelligence can achieve this easily. There is nothing like magic in the field of AI, but we have various algorithms and approaches which solve our problems mathematically. The most important thing we need to create a crowd estimation system is data- a good dataset with proper information always helps developers to create a model. Another important thing is the choice of mode. We have various models which provide different level of accuracy for different dataset. We will cover some of the best approaches here.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3><strong>Contents:</strong></h3>
<!-- /wp:heading -->

<!-- wp:list -->
<ul><li>Prerequisites</li><li>Datasets</li><li>Previous Achievement History</li><li>Best Models</li><li>Useful Links</li></ul>
<!-- /wp:list -->

<!-- wp:heading {"level":3} -->
<h3><strong>Prerequisites</strong></h3>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>The requirements for understanding this blog are not high. A beginner to AI/ ML can easily understand contents inside this article.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3><strong>Datasets</strong></h3>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Crowd Counting has traditionally been employed on several classic datasets. Many of these are publicly available on the web.</p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul><li><a href="http://pascal.inrialpes.fr/data/human/" rel="noreferrer noopener" target="_blank">INRIA Person datasets</a></li><li><a href="http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html" rel="noreferrer noopener" target="_blank">Mall Dataset</a></li><li><a href="http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/" rel="noreferrer noopener" target="_blank">Caltech Pedestrian Dataset</a></li><li><a href="https://svip-lab.github.io/datasets.html" rel="noreferrer noopener" target="_blank">ShanghaiTech Dataset</a></li><li><a href="http://www.ee.cuhk.edu.hk/~xgwang/datasets.html" rel="noreferrer noopener" target="_blank">World Expo Dataset</a></li><li><a href="http://www.ee.cuhk.edu.hk/~xgwang/grandcentral.html" rel="noreferrer noopener" target="_blank">Grand Central Station Dataset</a></li><li><a href="https://www.crcv.ucf.edu/data/ucf-cc-50/" rel="noreferrer noopener" target="_blank">UCF CC 50</a></li><li><a href="https://www.crcv.ucf.edu/data/ucf-qnrf/" rel="noreferrer noopener" target="_blank">UCF-QNRF — A Large Crowd Counting Data Set</a></li></ul>
<!-- /wp:list -->

<!-- wp:heading {"level":4} -->
<h4><strong>* INRIA Person datasets:</strong></h4>
<!-- /wp:heading -->

<!-- wp:list -->
<ul><li>Includes large set of marked up images of standing or walking people.</li><li>The dataset is divided in two formats:</li></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>a. original images with corresponding annotation files, and</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>b. positive images in normalized 64x128 pixel format (as used in the CVPR paper) with original negative images.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><em>Annotations are the bounding box( a rectangular box around the detected object) with the corresponding class label. Normalization of image is done in order to get proper histogram of image. During Computer Vision problems, images are generally normalized in the range of (0, 1) by doing this the error/ loss can be decreased rapidly hence faster convergence of model. But in general we can transform image into desired range.</em></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Sample from the dataset is given below:</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://cdn-images-1.medium.com/max/800/0*orZg-8RwCajYPlMf" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p><a href="https://www.semanticscholar.org/paper/An-Improved-Labelling-for-the-INRIA-Person-Data-Set-Taiana-Nascimento/3b304585d5af0afe98a85d6e0559315fbf3a7807" rel="noreferrer noopener" target="_blank">Source</a></p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":4} -->
<h4><strong>* Mall Datasets</strong>:</h4>
<!-- /wp:heading -->

<!-- wp:list -->
<ul><li>Composed by RGB images of frames in a video (as inputs) and the object counting on every frame, this is the number of pedestrians (object) in the image</li><li>Images are 480x640 pixels at 3 channels of the same spot recorded by a webcam in a mall, but there is a different number of persons on every frame, which is a problem in crowd counting.</li><li>This dataset can be used for regression.</li><li>The properties of the dataset are:</li><li><em>Video length: 2000 frames</em></li><li><em>Frame size: 640x480</em></li><li><em>Frame rate: &lt; 2 Hz</em></li><li>In dataset, over 60,000 pedestrians were labelled in 2000 video frames.</li></ul>
<!-- /wp:list -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://cdn-images-1.medium.com/max/800/0*4f8il1ZF95PCM5RH" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>Example frame</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":4} -->
<h4><strong>* Caltech Pedestrian Datasets:</strong></h4>
<!-- /wp:heading -->

<!-- wp:list -->
<ul><li>Consists of approximately 10 hours of 640x480 30Hz video taken from a vehicle driving through regular traffic in an urban environment.</li><li>About 250,000 frames (in 137 approximately minute long segments) with a total of 350,000 bounding boxes and 2300 unique pedestrians were annotated.</li><li>Annotation includes temporal correspondence between bounding boxes and detailed occlusion labels.</li></ul>
<!-- /wp:list -->

<!-- wp:quote -->
<blockquote class="wp-block-quote"><p><em>Occlusion on the field of Computer Vision is the condition when the object we’re tracking is hidden by scene or other objects.</em></p></blockquote>
<!-- /wp:quote -->

<!-- wp:paragraph -->
<p>For more information please follow the <a href="http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/files/PAMI12pedestrians.pdf" rel="noreferrer noopener" target="_blank">link</a>.</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://cdn-images-1.medium.com/max/800/0*oEOfDKuhbGCX8dvl" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>Sample example</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":4} -->
<h4><strong>* Shanghai Tech&nbsp;Dataset:</strong></h4>
<!-- /wp:heading -->

<!-- wp:list -->
<ul><li>Appeared in CVPR 2016 paper <a href="https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf" rel="noreferrer noopener" target="_blank">Single Image Crowd Counting via Multi Column Convolutional Neural Network</a>.</li><li>Authors of the dataset have made two parts Part A and Part B.</li><li>Both datasets have unique data images.</li><li>In each dataset&nbsp;, there are 3 folders:</li><li><em>images: the jpg image file</em></li><li><em>ground-truth: matlab file contain annotated head (coordinate x, y)</em></li><li><em>ground-truth-h5: people density map</em></li></ul>
<!-- /wp:list -->

<!-- wp:quote -->
<blockquote class="wp-block-quote"><p><em>Ground truth in the field of Computer Vision is the location of object the model will predict if it is working fine. In general we can use it to compare with the result of model.</em></p></blockquote>
<!-- /wp:quote -->

<!-- wp:heading {"level":4} -->
<h4><strong>* World Expo&nbsp;Dataset:</strong></h4>
<!-- /wp:heading -->

<!-- wp:list -->
<ul><li>Split into two parts</li><li>1,127 one-minute long video sequences out of 103 scenes are treated as training and validation sets.</li><li>3 labeled frames in each training video and the interval between two labeled frames is 15 seconds.</li></ul>
<!-- /wp:list -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://cdn-images-1.medium.com/max/800/0*ue6Xd2V_ojNIEuhT" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p><a href="http://www.ee.cuhk.edu.hk/~xgwang/expo.html" rel="noreferrer noopener" target="_blank">Source</a></p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":4} -->
<h4><strong>* Grand Central Station&nbsp;Dataset:</strong></h4>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Description:</p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul><li>Length: 33:20 minutes</li><li>Frame No.: 50010 frames</li><li>Frame Rate: 25fps, 720x480</li></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>It includes 3 files:</p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul><li>[Video] contains the video compressed into 1.1 GB AVI file by ffmpeg for download convenience.</li><li>[Trajectories] contains the KLT keypoint trajectories extracted from the video, which are used in our CVPR2012 paper.</li><li>[TrajectoriesNew] contains new bunch of KLT keypoint trajectories extracted from the video with KLT tracker slightly modified.</li></ul>
<!-- /wp:list -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://cdn-images-1.medium.com/max/800/0*T7CDal65suAFjkqk" alt=""/><figcaption>Sample Image</figcaption></figure>
<!-- /wp:image -->

<!-- wp:heading {"level":4} -->
<h4><strong>* UCF CC 50&nbsp;Dataset:</strong></h4>
<!-- /wp:heading -->

<!-- wp:list -->
<ul><li>Contains images of extremely dense crowds.</li><li>The images are collected mainly from the FLICKR.</li><li>This dataset was used on <a href="https://www.crcv.ucf.edu/papers/cvpr2013/Counting_V3o.pdf" rel="noreferrer noopener" target="_blank">Multi-Source Multi-Scale Counting in Extremely Dense Crowd Images</a>.</li></ul>
<!-- /wp:list -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://cdn-images-1.medium.com/max/800/0*27KKWqOSGwKsJv3G" alt=""/><figcaption>Sample Image</figcaption></figure>
<!-- /wp:image -->

<!-- wp:heading {"level":4} -->
<h4><strong>* UCF-QNRF — A Large Crowd Counting Data&nbsp;Set:</strong></h4>
<!-- /wp:heading -->

<!-- wp:list -->
<ul><li>Contains 1535 images which are divided into train and test sets of 1201 and 334 images respectively.</li><li>According to authors, this dataset is most suitable for training very deep Convolutional Neural Networks (CNNs) since it contains order of magnitude more annotated humans in dense crowd scenes than any other available crowd counting dataset.</li><li>Authors have even provided the comparison of dataset with other dataset.</li><li>This dataset was used on <a href="https://www.crcv.ucf.edu/papers/eccv2018/2324.pdf" rel="noreferrer noopener" target="_blank">Composition Loss for Counting, Density Map Estimation and Localization in Dense Crowds</a>.</li></ul>
<!-- /wp:list -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://cdn-images-1.medium.com/max/800/0*y4gFfBosrmk6WWyi" alt=""/><figcaption>Sample Images</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p><strong>Previous Achievement History</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>There have been various implementations to estimate the number of people in crowds but there always will be many challenges and limitations on the developed model. Some key challenges:</p>
<!-- /wp:paragraph -->

<!-- wp:list {"ordered":true} -->
<ol><li>One of the main challenges is the quality of the image/video. Taking higher resolution of image always helps to gain greater information but in dark side it requires lots of resources.</li><li>Detecting the relative distance between any two people from a live visual is a tough task(a key measure that would be really useful to help measure density)</li></ol>
<!-- /wp:list -->

<!-- wp:heading {"level":3} -->
<h3><strong>Best Models</strong></h3>
<!-- /wp:heading -->

<!-- wp:heading {"level":4} -->
<h4><strong>* C³ Framework (Crowd Counting&nbsp;Code)</strong></h4>
<!-- /wp:heading -->

<!-- wp:list -->
<ul><li>A lot of research have been done on the field of object tracking/ object counting, but as time moves on, newer and faster techniques come along with new problems and solutions. Most of the research has been done for specific dataset only. But currently we have <a href="https://arxiv.org/pdf/1907.02724" rel="noreferrer noopener" target="_blank"><strong>C³ Framework</strong></a></li><li>This framework works with 6 main datasets UCF CC 50, WorldExpo’10, SHT A, SHT B, UCF-QNRF, and GCC.</li></ul>
<!-- /wp:list -->

<!-- wp:image {"linkDestination":"custom"} -->
<figure class="wp-block-image"><img src="https://cdn-images-1.medium.com/max/800/0*u4NQfwPzE2d2TEHp" alt=""/><figcaption>Source Original&nbsp;<a href="https://arxiv.org/pdf/1907.02724.pdf" rel="noreferrer noopener" target="_blank">Paper</a></figcaption></figure>
<!-- /wp:image -->

<!-- wp:list -->
<ul><li>Authors have used various models(AlexNet, VGG Series, ResNet Series etc.) for doing classification.</li><li>MAE(Mean Absolute Error) and MSE(Mean Squared Error) are used for error calculation.</li><li>And this model have relatively low MAE and MSE than other. Pytorch code of C³ Framework can be found on <a href="https://github.com/gjy3035/C-3-Framework" rel="noreferrer noopener" target="_blank">this GitHub link.</a></li></ul>
<!-- /wp:list -->

<!-- wp:heading {"level":4} -->
<h4><strong>* CSRNet&nbsp;: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes</strong></h4>
<!-- /wp:heading -->

<!-- wp:list -->
<ul><li>According to authors,</li><li>“We propose a network for Congested Scene Recognition called CSRNet to provide a data-driven and deep learning method that can understand highly congested scenes and perform accurate count estimation as well as present high-quality density maps.</li><li>The proposed CSRNet is composed of two major components: a convolutional neural network (CNN) as the front-end for 2D feature extraction and a dilated CNN for the back-end, which uses dilated kernels to deliver larger reception fields and to replace pooling operations.</li><li>CSRNet is an easy-trained model because of its pure convolutional structure. We demonstrate CSRNet on four datasets (ShanghaiTech dataset, the UCF_CC_50 dataset, the WorldEXPO’10 dataset, and the UCSD dataset) and we deliver the state-of-the-art performance.</li><li>In the ShanghaiTech Part_B dataset, CSRNet achieves 47.3% lower Mean Absolute Error (MAE) than the previous state-of-the-art method.</li><li>We extend the targeted applications for counting other objects, such as the vehicle in TRANCOS dataset.</li><li>Results show that CSRNet significantly improves the output quality with 15.4% lower MAE than the previous state-of-the-art approach.”</li><li>The paper is publicly available <a href="https://arxiv.org/pdf/1802.10062v4.pdf" rel="noreferrer noopener" target="_blank">here.</a></li></ul>
<!-- /wp:list -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://cdn-images-1.medium.com/max/800/0*7-Imjua0BC_7ymvm" alt=""/><figcaption>Source Original&nbsp;Paper</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>Pytorch Code can be found on this <a href="https://github.com/leeyeehoo/CSRNet-pytorch" rel="noreferrer noopener" target="_blank">GitHub Link.</a></p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":4} -->
<h4><strong>* PCC Net: Perspective Crowd Counting via Spatial Convolutional Network</strong></h4>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>This is another CNN based crowd counting technique by generating density. According to authors, It consists of:</p>
<!-- /wp:paragraph -->

<!-- wp:list {"ordered":true} -->
<ol><li>Density Map Estimation (DME) focuses on learning very local features for density map estimation;</li><li>Random High-level Density Classification (R-HDC) extracts global features to predict the coarse density labels of random patches in images;</li><li>Fore- /Background Segmentation (FBS) encodes mid-level features to segments the foreground and background</li></ol>
<!-- /wp:list -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://cdn-images-1.medium.com/max/800/0*-ip1tXKXe2R9vzlE" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://cdn-images-1.medium.com/max/800/0*FBrCW4gCoVyJFaZd" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>Pytorch code of PCCNet can be found on <a href="https://github.com/gjy3035/PCC-Net" rel="noreferrer noopener" target="_blank">Github Link.</a></p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3>Conclusion</h3>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Now we have various algorithms and the frameworks to do crowd counting, we can choose any one of them and do experiment. There are still numerous research and frameworks not listed here but here is a <a href="https://github.com/gjy3035/Awesome-Crowd-Counting" rel="noreferrer noopener" target="_blank">link</a> which have most of their links.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>During writing this article I got lots of valuable suggestions from study group members(#sg_wonder_vision) and especially <em>Pooja Vinod </em>who have helped me complete this. She have made this <a href="https://feedback.wix.com/?sharedId=e7735d4b-1e1b-4eb3-a771-2c444c2db981&amp;iFrameUrl=http%3A%2F%2Feditor.wix.com%2Fhtml%2Feditor%2Freview%3Ft%3DJWE.eyJhbGciOiJBMTI4S1ciLCJlbmMiOiJBMTI4Q0JDLUhTMjU2Iiwia2lkIjoiQkFZSGZGVFYifQ.OoyweZd_nKZEzIYH_OCYdlKt3xqL8D-sKLQK-819Dnemjzjf9CoxMg.9L19hl2qzvpTAhoqjnzC-Q.ii14eISjcU_KB3UxgQRUQq0ORX5Lv_AaRSYNmsfzG7oUBd0r8HsosxDINa3CSGX3dzwIM4WPH1TEVU62q98-WsyYKfv05pol10Zoa8Cm1mB-U6sa52mLZyg7n1qgSWRLaP08_ADmchWHBB87RzHHpUYqNZObdyMDkbTVXdDBNDOSGtblrYeLqyutIhtR_1AprfbN8paW3cZkS0imBDazcVKMDdYD3Mbes99_umKasOW_c-HKWSD8I0ik1c6YAz7ex9czZhmfBJ8EgVc9CISHvC3WHUv1IVHWL0ml2FNYCWn3ibwdf-vb8ZKfZ8PcjPh7.bR0eGU0UVsx2PILuO4DJwA" rel="noreferrer noopener" target="_blank">awesome website</a> and she have included our implementations.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Useful Links</strong></p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul><li><a href="https://www.analyticsvidhya.com/blog/2019/02/building-crowd-counting-model-python/" rel="noreferrer noopener" target="_blank">Analytics Vidhya</a></li><li><a href="http://papers.nips.cc/paper/4043-learning-to-count-objects-in-images" rel="noreferrer noopener" target="_blank">Learning To Count Objects in Images</a></li><li><a href="https://blog.algorithmia.com/algorithm-spotlight-crowd-counter/" rel="noreferrer noopener" target="_blank">Algorithm Spotlight: Crowd Counter</a></li></ul>
<!-- /wp:list -->