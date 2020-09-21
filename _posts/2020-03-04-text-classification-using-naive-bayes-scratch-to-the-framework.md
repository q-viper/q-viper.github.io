---
title: Text Classification using Naive Bayes, Scratch to the Framework
date: 2020-03-04T04:50:23+05:45
header:
  teaser: assets/wp-content/uploads/2020/03/mlst_01051.png
categories:
  - Artificial Intelligence
  - Machine Learning
  - Programming
tags:
  - machine learning
  - naive bayes
  - spam classification
  - text classification
---
<!-- wp:paragraph -->
<p>So this is not a blog for introduction to naive bayes but implementation way for spam message classification. I have also created a YouTube video for this topic which is available on the link below.</p>
<!-- /wp:paragraph -->

<!-- wp:core-embed/youtube {"url":"https://www.youtube.com/watch?v=jlQPojZlX2Q","type":"video","providerNameSlug":"youtube","className":"wp-embed-aspect-16-9 wp-has-aspect-ratio"} -->
<figure class="wp-block-embed-youtube wp-block-embed is-type-video is-provider-youtube wp-embed-aspect-16-9 wp-has-aspect-ratio"><div class="wp-block-embed__wrapper">
https://www.youtube.com/watch?v=jlQPojZlX2Q
</div></figure>
<!-- /wp:core-embed/youtube -->

<!-- wp:paragraph -->
<p>And the resources are available on the link below.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><a href="https://github.com/q-viper/ML-from-Basics">https://github.com/q-viper/ML-from-Basics</a></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Steps to perform text classification are:-</p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul><li>Lower Casing  text</li><li>Remove Punctuation</li><li>Perform Bag of words</li><li>Frequency of words</li><li>Find Bag of Words</li><li>Probability of word on class p(w/c)</li><li>Probability of class given word p(c/w)</li></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>In order to perform our classifier we have to preprocess our input data. For text processing, our data will be on text format so we will convert that into vector form. In general we will find a data frame where index will be the example and columns will be all the unique words from our training set. Then each cell will be probability of word on class. Then for the part of prediction, we will find p(c/w) using simple bayes formula:-</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>p(c/w) = p(w/c) * p(c) / p(w)</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Please follow through the video for more information about the topic. </p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Thank you for reading the post and feel free to share it. :)</p>
<!-- /wp:paragraph -->