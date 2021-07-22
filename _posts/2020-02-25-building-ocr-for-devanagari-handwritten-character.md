---
title: Building OCR For Devanagari Handwritten Character
date: 2020-02-25T07:46:33+05:45
comments_id: 2
header:
    teaser: assets/wp-content/uploads/2020/02/dcr.png
categories:
  - Artificial Intelligence
  - Computer Vision
  - Machine Learning
  - Project
tags:
  - computer vision
  - devanagari character recognition
  - image processing
  - ocr
  - python computer vision
---
<!-- wp:heading {"level":4} -->
<h4>Using Keras, OpenCv, Numpy build a simple&nbsp;OCR.</h4>
<!-- /wp:heading -->

**Contents**
* TOC
{:toc}

<!-- wp:heading {"level":4} -->
<h4>Inspiration</h4>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Devanagari is popular across the India and Nepal. It is also a National font of Nepal so back on 2018 I thought of doing OCR for our font as project. I had no clue how to do it but I knew some basics of Machine Learning. But I started doing it on 2019 February and it ended on 3 months. At the end it became as my school project.</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://cdn-images-1.medium.com/max/800/1*gvAMsk8_o_I_FDP5UaMraw.png" alt=""/><figcaption>Recoginiton Of Devanagari Character</figcaption></figure>
<!-- /wp:image -->

<!-- wp:heading {"level":4} -->
<h4>Requirements</h4>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Some basic knowledge on Machine Learning. And for coding, you might need keras 2.X, open-cv 4.X, Numpy and Matplotlib.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":4} -->
<h4>Introduction</h4>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Devanagari is the national font of Nepal and is used widely throughout the India also. It contains 10 numerals(०, १, २, ३, ४, ५, ६, ७, ८, ९) and 36 consonants (क, ख, ग, घ, ङ, च, छ, ज, झ, ञ, ट, ठ, ड, ढ, ण, त, थ, द, ध,न, प,फ, ब, भ, म, य, र, ल, व, श, ष, स, ह, क्ष, त्र, ज्ञ). Some consonants are complex and made by combine some other. However, throughout this project I considered them as single character.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>The required dataset is publicly available on the <a href="https://web.archive.org/web/20160307001701/http:/cvresearchnepal.com/wordpress/dhcd/" rel="noreferrer noopener" target="_blank">link.</a> Huge credit goes to the team who collected the dataset and made it public.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":4} -->
<h4>Dataset Preparation</h4>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>We could create own version of dataset but why to take lot time rather than working with already collected dataset. The format of image was Grayscale with 2 pixels margin on each side. I didn’t knew that much about ‘Image Datagenerator’ of Keras then so I converted all the image files to CSV file with first column as label and remaining 1024 as pixel values. But now I highly recommend to use ‘<em>Image Datagenerator</em>’.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":4} -->
<h4>Model Preparation</h4>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>For model, I picked up a simple CNN. Keras Summary of model is given below.</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://cdn-images-1.medium.com/max/800/0*PG712MaX4_yuJEaL" alt=""/><figcaption>Keras Model&nbsp;Summary</figcaption></figure>
<!-- /wp:image -->

<!-- wp:heading {"level":4} -->
<h4>Model Training</h4>
<!-- /wp:heading -->

<!-- wp:list -->
<ul><li>Loss: Categorical Cross Entropy</li><li>Optimizer: SGD</li><li>Batch size: 32</li><li>Epochs: 100</li><li>Validation split: 0.2</li><li>Train time: 37.86 minutes on Google Colab</li><li>Test accuracy: 99.29%</li></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>I have used various models and some week to train a fine model and ended up with the best one by using above parameters. Here is the image about how model was tuning.</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://cdn-images-1.medium.com/max/800/0*LQRwtgIfmOdVszPb" alt=""/><figcaption>Model Accuracy</figcaption></figure>
<!-- /wp:image -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://cdn-images-1.medium.com/max/800/0*X5rDROv99etqRjnl" alt=""/><figcaption>Model Loss</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>My Github Repository consists of all top 3 models(cnn0, cnn1, cnn2) and their code please follow through this <a href="https://github.com/q-viper/final-devanagari-word-char-detector" rel="noreferrer noopener" target="_blank">link</a>.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":4} -->
<h4>Image Processing</h4>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Training a model alone will not create a OCR. And we can’t use real world image on the model without doing pre-processing. Here is a complete image processing model code.</p>
<!-- /wp:paragraph -->

<!-- wp:html -->
<script src="https://gist.github.com/q-viper/a9651d5449c52ff66cc7d5503dd3b3fc.js"></script>
<!-- <figure><iframe width="700" height="250" src="/media/da77729b842075c17e47eaa65080e5a9" allowfullscreen=""></iframe><figcaption>Image Processing Method</figcaption></figure> -->
<!-- /wp:html -->

<!-- wp:paragraph -->
<p>First we will copy the BGR(OpenCv reads on BGR) image and then convert it to Binary Image using OpenCv’s threshold function. Thresholding the image always reduces the complexity of tasks because we will be working on only 2 pixel values 0 and 255. Another important thing is we have to find the background and foreground pixels. This is really tricky part because there will be a case where our text will be white and background be black and vice versa so we need to do some trick to always find foreground and background pixels. Here I checked only 5 pixels from the top left corner. This idea will not always work but for some time it will be good approach.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Next we need to find the ROI. Because the text might be situated on any side of the image. So we must find the exact image position and crop it to do further processing. Next We will do segmentation. Here I used only <strong>Numpy for image cropping and segmenting. </strong>This sounds funny but it is <em>true.</em></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Finding ROI</strong></p>
<!-- /wp:paragraph -->

<!-- wp:html -->
<script src="https://gist.github.com/q-viper/253bdba4fd4e622cbc2b1d1318b5cf27.js"></script>
<!-- /wp:html -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://cdn-images-1.medium.com/max/800/1*u2E1Mid2L5sxkcdpUXZBoQ.png" alt=""/><figcaption>Working of&nbsp;Cropping</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>Finding the real position of text is another problem. Because on realtime, image can consists lot of noise and there is always a chance of finding the false shapes. To handle this I have wrote some formula. Here I set the some pixel values to be noise and neglect them. Then we keep checking from the top of the image. Whenever we find the foreground pixels more than the noise value, we crop the image from the position <em>current_row — noise_value.</em> We do same for other 3 sides also.</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://cdn-images-1.medium.com/max/800/1*TXA_wGVxr3bcsOKOSUGIqQ.png" alt=""/><figcaption>Cropped Image</figcaption></figure>
<!-- /wp:image -->

<!-- wp:heading {"level":4} -->
<h4>Segmentation</h4>
<!-- /wp:heading -->

<!-- wp:html -->
<script src="https://gist.github.com/q-viper/2f158f5a5bbca55e1a913810e0ef51fd.js"></script>
<!-- /wp:html -->

<!-- wp:paragraph -->
<p>Now more crazy part is image segmentation using Numpy. We take the copy of cropped image and remove the top most part of the text. On our Nepali, we call it ‘Dika’. By doing this we can actually get some space between characters. So I wrote a general formula which will work for all images.</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://cdn-images-1.medium.com/max/800/1*0M055AAKqeXN85MOSfpgOw.png" alt=""/><figcaption>Removed ‘Dika’</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>Next we will run a loop through each column and see if any column is entire a background pixel. If so it might be the right place to do segmentation but there is always a risk of having large margin. So we send them again to bordered function to remove the unwanted background spaces. After doing so, we will get the exact column number from which we can slice our original cropped image. I’ve called them <em>segmented_templates </em>here.</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://cdn-images-1.medium.com/max/800/1*gX2lSeJFUAQm2MIRRy7fsw.png" alt=""/><figcaption>Image segmentation</figcaption></figure>
<!-- /wp:image -->

<!-- wp:heading {"level":4} -->
<h4>Localization</h4>
<!-- /wp:heading -->

<!-- wp:html -->
<script src="https://gist.github.com/q-viper/aa94429d356def9fb476020d8c206b6e.js"></script>
<!-- /wp:html -->

<!-- wp:paragraph -->
<p>Localization is the concept of finding the exact image position and showing the border. We use the previous segments and pass them as template and using OpenCv’s template matching method we find the exact position where that segment matched. Of course the template will match 100% but i’ve set the threshold value to 0.8 here. Whenever a template matches, i’ve drawn a rectangle around the matched portion on original image. For this, I have used OpenCv’s rectangle drawing.</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://cdn-images-1.medium.com/max/800/1*8b5iepLSl33Upf6aEv3vYg.jpeg" alt=""/><figcaption>Localizing</figcaption></figure>
<!-- /wp:image -->

<!-- wp:heading {"level":4} -->
<h4>Add Border</h4>
<!-- /wp:heading -->

<!-- wp:html -->
<script src="https://gist.github.com/q-viper/5bca6014fad5489a9625e2b0b34e46a0.js"></script>
<!-- /wp:html -->

<!-- wp:paragraph -->
<p>Now our image must be converted to 32 by 32 size because our training data is also 32 by 32. But resizing the segments to that shape will cause our prediction fail mostly. The reason is, our train image have 2 pixels margin around it. And we need to do so here also.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":4} -->
<h4>Prediction With Trained&nbsp;Model</h4>
<!-- /wp:heading -->

<!-- wp:html -->
<script src="https://gist.github.com/q-viper/e280064574bdbf85d4010ebc5ecbd60c.js"></script>
<!-- /wp:html -->

<!-- wp:paragraph -->
<p>For each preprocessed segments passed to this function will return the accuracy and label of prediction. We will use these on the recognition method.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":4} -->
<h4>Recognition Of&nbsp;Segments</h4>
<!-- /wp:heading -->

<!-- wp:html -->
<script src="https://gist.github.com/q-viper/24ec4fdace54777e35e7731fd600d8c7.js"></script>
<!-- /wp:html -->

<!-- wp:paragraph -->
<p>No matter how hard we code there will be always a false positive predictions. But we can try to reduce them. Problem of false positive can happen when the image quality is low and entire text is taken as single character. On that case, we take that as true only if prediction is more than 80%. But on final code to prevent localization of false segments, I have done localization after finding if segment is true. If the prediction is less than 80%, the entire text will be treated as single character and done prediction.</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://cdn-images-1.medium.com/max/800/1*gvAMsk8_o_I_FDP5UaMraw.png" alt=""/><figcaption>Recoginiton and Localization of&nbsp;Word</figcaption></figure>
<!-- /wp:image -->

<!-- wp:heading {"level":4} -->
<h4>Camera For&nbsp;Realtime</h4>
<!-- /wp:heading -->

<!-- wp:html -->
<script src="https://gist.github.com/q-viper/725fddd9d810b477b894554e4c9a66c1.js"></script>
<!-- /wp:html -->

<!-- wp:paragraph -->
<p>OCR needs camera to work. So I used the OpenCv’s camera methods for doing realtime image capture. Here on above code, I wrote plenty of codes to do some interesting things. The camera will show a rectangular box and we can actually manipulate its shape also. The portion of image lying inside the box will be sent to the recognition process. Here I used some keys like spacebar for capture image, enter key for relatime video, etc.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":4} -->
<h4>Combining It&nbsp;all</h4>
<!-- /wp:heading -->

<!-- wp:html -->
<script src="https://gist.github.com/q-viper/361c977f65ed63346cc0e56cf3db9499.js"></script>
<!-- /wp:html -->

<!-- wp:paragraph -->
<p>Now is the time for integration of all the modules. User can pass the image location on local storage and if the image doesn’t exists, program runs the camera mode.</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://cdn-images-1.medium.com/max/800/1*xabnnrF-Qn2TwzYL-MKMew.png" alt=""/><figcaption>Overall System&nbsp;Process</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>I have tried to do Android App development by using TensorflowLite also but it is still paused. And I am planning to write a code for web app.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Thank you so much for reading this article. And please follow to this <a href="https://github.com/q-viper/final-devanagari-word-char-detector" rel="noreferrer noopener" target="_blank">Github Link</a> for the entire project and the documentation.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Find me on <a href="https://twitter.com/QuassarianViper" rel="noreferrer noopener" target="_blank">Twitter</a>.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Find me on <a href="https://www.linkedin.com/in/ramkrishna-acharya-91a217183/?lipi=urn%3Ali%3Apage%3Ad_flagship3_feed%3BFCZIE%2FfkS2usch6WJ6YCSg%3D%3D" rel="noreferrer noopener" target="_blank">LinkedIn</a>.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Find me on <a href="https://www.youtube.com/channel/UCR8bjIFUkmWMRntCNLsAuIg" rel="noreferrer noopener" target="_blank">Youtube.</a></p>
<!-- /wp:paragraph -->