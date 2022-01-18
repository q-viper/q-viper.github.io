---
title:  "Image Compression In Python: Huffman Encoding"
date:   2021-12-24 09:29:17 +0545
categories:
  - Computer Vision
  - Image Processing
tags:
  - Computer Vision
  - Image Processing
  - huffman encoding
  - image compression
header:
  teaser: assets/huffman/huffman.png
---

## Huffman Coding
This blog will be completed with much more exercises on [dataqoil.com](https://dataqoil.com/2022/01/03/image-compression-in-python-huffman-encoding/).

Huffman coding is a popular lossless Variable Length Coding (VLC), based on the following principles:
1. Shorter code words are assigned to more probable symbols and longer code words are assigned to less probable symbols. 
2. No code word of a symbol is a prefix of another code word. This makes Huffman coding uniquely decodable.
3. Every source symbol must have a unique code word assigned to it.

In image compression systems, Huffman coding is performed on the quantized symbols. The first step is to create a series of source reductions by ordering the probabilities of the symbols under consideration and combining the lowest probability symbols into a single symbol that replace them in the next source reduction. The second step in Huffman’s procedure is to code each reduced source, starting with the smallest source and working back to the original source. Huffman’s procedure creates the optimal code for a set of symbols. It is uniquely decodable, because any string of code symbols can be decoded in only one way.


## Example

### Reduction

Lets assume that we have a following image where the probability of occurance of each symbol is given.

|Symbol|Probability|
|-----|------|
|s1|0.4|
|s2|0.3|
|s3|0.1|
|s4|0.1|
|s5|0.07|
|s6|0.03|

* Now in the first step, we select two symbols which have lowest probability. It is `s5` and `s6`. Then sum them up. We will have new table as:

|symbol|probability|Reduction 1|
|-----|------|----|
|s1|0.4|0.4|
|s2|0.3|0.3|
|s3|0.1|0.1|
|s4|0.1|0.1|
|s5|0.07|0.1|
|s6|0.03|-|

* Now in the second step, we repeat same process.

|symbol|probability|Reduction 1|Reduction 2|
|-----|------|----|---|
|s1|0.4|0.4|0.4|
|s2|0.3|0.3|0.3|
|s3|0.1|0.1|0.1|
|s4|0.1|0.1|0.2|
|s5|0.07|0.1|-|
|s6|0.03|-|-|

* Similarly, at the end we will get reduction table as:

|symbol|probability|Reduction 1|Reduction 2|Reduction 3|Reduction 4|
|-----|------|----|---|--|--|
|s1|0.4|0.4|0.4|0.4|0.4|
|s2|0.3|0.3|0.3|0.3|0.6|
|s3|0.1|0.1|0.1|0.3|-|
|s4|0.1|0.1|0.2|-|-|
|s5|0.07|0.1|-|-|-|
|s6|0.03|-|-|-|-|

Now only two probability values are left so we stop the reduction here and start the code assignment.

### Code Assignment
Now we start from the final reduction table where only two values are left. We give different bit values to these two. 

* Give 0 to 0.6 and 1 to 0.4.

|symbol|probability|Reduction 1|Reduction 2|Reduction 3|Reduction 4|
|-----|------|----|---|--|--|
|s1|0.4|0.4|0.4|0.4|0.4 (1)|
|s2|0.3|0.3|0.3|0.3|0.6 (0)|
|s3|0.1|0.1|0.1|0.3|-|
|s4|0.1|0.1|0.2|-|-|
|s5|0.07|0.1|-|-|-|
|s6|0.03|-|-|-|-|

* Now our value 0.6 was derived from 0.3 and 0.3 from 3rd reduction step. Thus, we get to that step and give 0 and 1 to individual 0.3 and 0.3. Now our table will be like below.

|symbol|probability|Reduction 1|Reduction 2|Reduction 3|Reduction 4|
|-----|------|----|---|--|--|
|s1|0.4|0.4|0.4|0.4 (1)|0.4 (1)|
|s2|0.3|0.3|0.3|0.3 (00)|0.6 (0)|
|s3|0.1|0.1|0.1|0.3 (01)|-|
|s4|0.1|0.1|0.2|-|-|
|s5|0.07|0.1|-|-|-|
|s6|0.03|-|-|-|-|

In above table, we took the code 0 from 0.6 and put 0, 1 respectively to get code for 0.3 and 0.3.

* Again repeating above process,
|symbol|probability|Reduction 1|Reduction 2|Reduction 3|Reduction 4|
|-----|------|----|---|--|--|
|s1|0.4|0.4|0.4 (1)|0.4 (1)|0.4 (1)|
|s2|0.3|0.3|0.3 (00)|0.3 (00)|0.6 (0)|
|s3|0.1|0.1|0.1 (010)|0.3 (01)|-|
|s4|0.1|0.1|0.2 (011)|-|-|
|s5|0.07|0.1|-|-|-|
|s6|0.03|-|-|-|-|

The code value increases by going leftward everytime.

* Finally,

|symbol|probability| Code|Reduction 1|Reduction 2|Reduction 3|Reduction 4|
|-----|---|---|----|---|--|--|
|s1|0.4 |1|0.4 (1)|0.4 (1)|0.4 (1)|0.4 (1)|
|s2|0.3 |00|0.3 (00)|0.3 (00)|0.3 (00)|0.6 (0)|
|s3|0.1 |010|0.1 (010)|0.1 (010)|0.3 (01)|-|
|s4|0.1 |0110|0.1 (0110)|0.2 (011)|-|-|
|s5|0.07 |01110|0.1 (0111)|-|-|-|
|s6|0.03 |01111|-|-|-|-|

In above example, we have given a unique code for each value and it is clear that the vlaue with high probability have least length of code.

### Average length

$$
L(z) = \sum_{i=1}^{n}{L(a_i)P(a_i)}
$$

Where L is length of each code and P is a probability.

In our example,

$$
L = 1*0.4 + 2*0.3 + 3*0.1+4*0.1+5*0.07+5*0.03 \\
=  2.2 bits/symbol
$$




**Coding Efficency**

$$
\nu = \frac{H(z)}{L(z)}
$$

Where, 

$$
H(z) = -\sum_{i=1}^{n}{P(a_i) * \log{P(a_i)}}
$$
It is also known as Information Entropy.

$$
H(z) = -(0.4 * log(0.4) + 0.3 * log(0.3) + 0.1 * log(0.1) + \\
0.1 * log(0.1) + 0.07 * log(0.07) + 0.03 * log(0.03)) \\
 = 1.8
$$

Now,

$$
\nu = \frac{1.8}{2.2} \\
= 0.81
$$

Which means that our Huffman coding is 81% efficent.

## Python Implementation


```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys, os

def show(img, figsize=(15,15), title="Image"):
    fg = plt.figure(figsize=figsize)
    plt.imshow(img, vmin=0, vmax=255, cmap="gray")
    plt.show()
    
show(np.zeros((10,10)))
```


    
![png]({{site.url}}/assets/huffman/output_4_0.png)
    


We will be using below image.

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Football_%28soccer_ball%29.svg/1200px-Football_%28soccer_ball%29.svg.png)
(From wikimedia)


```python
img = cv2.imread("football.png",0)
show(img)
```


    
![png]({{site.url}}/assets/huffman/output_6_0.png)
    


### Get size of image


```python
sys.getsizeof(img)/1024
```




    1494.25



Above size is of img array after it has been read as a gray image.


```python
def get_size(filename="football.png"):
    stat = os.stat(filename)
    size=stat.st_size
    return size

print(get_size())
```

    230554
    

Above size is of our image has taken in filesystem.

### Get Probability


```python
from collections import Counter


fimg = img.flatten().tolist()
pxs = len(fimg)
tbl = Counter(fimg)

ntbl = {k:v/pxs for k,v in tbl.items()}
ntbl = dict(sorted(ntbl.items(), key=lambda item: item[1]))
ntbl
```




    {92: 2.156862745098039e-05,
     95: 2.2875816993464052e-05,
     74: 2.2875816993464052e-05,
     71: 2.3529411764705884e-05,
     86: 2.4183006535947712e-05,
     72: 2.4183006535947712e-05,
     90: 2.4836601307189544e-05,
     93: 2.4836601307189544e-05,
     101: 2.5490196078431373e-05,
     84: 2.5490196078431373e-05,
     79: 2.5490196078431373e-05,
     87: 2.5490196078431373e-05,
     78: 2.61437908496732e-05,
     100: 2.61437908496732e-05,
     97: 2.745098039215686e-05,
     75: 2.745098039215686e-05,
     76: 2.8104575163398693e-05,
     80: 2.8758169934640522e-05,
     98: 2.8758169934640522e-05,
     83: 2.8758169934640522e-05,
     99: 2.9411764705882354e-05,
     96: 2.9411764705882354e-05,
     94: 2.9411764705882354e-05,
     81: 2.9411764705882354e-05,
     89: 3.0718954248366014e-05,
     77: 3.137254901960784e-05,
     82: 3.202614379084967e-05,
     91: 3.3333333333333335e-05,
     88: 3.3333333333333335e-05,
     73: 4.052287581699346e-05,
     151: 4.901960784313725e-05,
     137: 5.032679738562092e-05,
     189: 5.032679738562092e-05,
     172: 5.0980392156862745e-05,
     122: 5.0980392156862745e-05,
     168: 5.22875816993464e-05,
     149: 5.3594771241830066e-05,
     144: 5.4248366013071894e-05,
     184: 5.4248366013071894e-05,
     146: 5.490196078431372e-05,
     135: 5.555555555555556e-05,
     139: 5.620915032679739e-05,
     183: 5.620915032679739e-05,
     150: 5.620915032679739e-05,
     147: 5.620915032679739e-05,
     162: 5.620915032679739e-05,
     148: 5.6862745098039215e-05,
     129: 5.7516339869281044e-05,
     166: 5.7516339869281044e-05,
     119: 5.7516339869281044e-05,
     127: 5.7516339869281044e-05,
     118: 5.7516339869281044e-05,
     192: 5.816993464052288e-05,
     114: 5.816993464052288e-05,
     124: 5.882352941176471e-05,
     174: 5.882352941176471e-05,
     157: 5.882352941176471e-05,
     164: 5.882352941176471e-05,
     190: 5.882352941176471e-05,
     123: 5.882352941176471e-05,
     193: 5.882352941176471e-05,
     181: 5.9477124183006536e-05,
     182: 6.0130718954248365e-05,
     170: 6.0130718954248365e-05,
     175: 6.0130718954248365e-05,
     143: 6.0130718954248365e-05,
     116: 6.0130718954248365e-05,
     187: 6.078431372549019e-05,
     134: 6.078431372549019e-05,
     140: 6.078431372549019e-05,
     155: 6.209150326797386e-05,
     165: 6.209150326797386e-05,
     115: 6.274509803921569e-05,
     142: 6.274509803921569e-05,
     163: 6.274509803921569e-05,
     167: 6.274509803921569e-05,
     186: 6.274509803921569e-05,
     177: 6.274509803921569e-05,
     145: 6.274509803921569e-05,
     136: 6.274509803921569e-05,
     195: 6.274509803921569e-05,
     161: 6.339869281045751e-05,
     176: 6.339869281045751e-05,
     160: 6.339869281045751e-05,
     169: 6.339869281045751e-05,
     171: 6.339869281045751e-05,
     125: 6.470588235294117e-05,
     126: 6.470588235294117e-05,
     120: 6.470588235294117e-05,
     198: 6.535947712418301e-05,
     133: 6.601307189542484e-05,
     152: 6.601307189542484e-05,
     173: 6.666666666666667e-05,
     117: 6.666666666666667e-05,
     196: 6.73202614379085e-05,
     188: 6.73202614379085e-05,
     178: 6.73202614379085e-05,
     179: 6.797385620915033e-05,
     111: 6.797385620915033e-05,
     200: 6.797385620915033e-05,
     121: 6.797385620915033e-05,
     194: 6.862745098039216e-05,
     110: 6.862745098039216e-05,
     112: 6.928104575163398e-05,
     159: 6.928104575163398e-05,
     141: 6.928104575163398e-05,
     197: 6.928104575163398e-05,
     201: 6.928104575163398e-05,
     131: 7.124183006535948e-05,
     185: 7.124183006535948e-05,
     208: 7.38562091503268e-05,
     138: 7.516339869281045e-05,
     156: 7.516339869281045e-05,
     130: 7.516339869281045e-05,
     180: 7.516339869281045e-05,
     113: 7.581699346405228e-05,
     108: 7.581699346405228e-05,
     191: 7.647058823529411e-05,
     204: 7.712418300653595e-05,
     132: 7.777777777777778e-05,
     158: 7.908496732026144e-05,
     199: 8.431372549019608e-05,
     109: 8.69281045751634e-05,
     202: 8.758169934640522e-05,
     106: 8.888888888888889e-05,
     203: 9.019607843137255e-05,
     105: 9.215686274509804e-05,
     107: 9.215686274509804e-05,
     154: 9.281045751633986e-05,
     205: 9.411764705882353e-05,
     206: 9.607843137254902e-05,
     209: 9.738562091503268e-05,
     210: 9.934640522875818e-05,
     207: 0.0001,
     70: 0.00010980392156862745,
     212: 0.00011111111111111112,
     211: 0.00012026143790849673,
     104: 0.00012352941176470587,
     128: 0.00012679738562091503,
     213: 0.0001496732026143791,
     103: 0.00018431372549019607,
     85: 0.00020392156862745098,
     153: 0.000330718954248366,
     68: 0.0003784313725490196,
     69: 0.0003980392156862745,
     66: 0.0004699346405228758,
     63: 0.000603921568627451,
     67: 0.0006156862745098039,
     65: 0.0007398692810457517,
     1: 0.0009248366013071895,
     61: 0.000945751633986928,
     10: 0.0011209150326797386,
     58: 0.0011843137254901961,
     4: 0.0012196078431372548,
     62: 0.0012235294117647058,
     3: 0.0012254901960784314,
     11: 0.001292156862745098,
     9: 0.0013019607843137255,
     56: 0.001333986928104575,
     8: 0.0014764705882352942,
     2: 0.0015019607843137254,
     64: 0.0015045751633986928,
     7: 0.0015143790849673202,
     60: 0.0015359477124183007,
     5: 0.0015490196078431372,
     13: 0.0016352941176470588,
     6: 0.0016784313725490196,
     59: 0.0018673202614379084,
     15: 0.0018869281045751634,
     12: 0.0020169934640522874,
     17: 0.0020882352941176473,
     18: 0.002249673202614379,
     14: 0.0023019607843137255,
     44: 0.0023607843137254904,
     42: 0.002588888888888889,
     102: 0.0026281045751633987,
     20: 0.002669281045751634,
     16: 0.0026980392156862746,
     41: 0.002735947712418301,
     47: 0.0028359477124183007,
     39: 0.0028372549019607843,
     37: 0.002903267973856209,
     34: 0.0029633986928104573,
     35: 0.003010457516339869,
     32: 0.003108496732026144,
     22: 0.003141830065359477,
     29: 0.0032013071895424837,
     30: 0.003235294117647059,
     23: 0.003415686274509804,
     27: 0.0034477124183006536,
     57: 0.0035738562091503267,
     19: 0.0035875816993464053,
     46: 0.0036,
     45: 0.004003921568627451,
     21: 0.004015686274509804,
     25: 0.004024183006535948,
     40: 0.0040398692810457515,
     38: 0.004173202614379085,
     54: 0.004298692810457516,
     33: 0.004549673202614379,
     31: 0.004598039215686274,
     28: 0.004822875816993464,
     36: 0.004833333333333334,
     55: 0.004891503267973856,
     43: 0.0049673202614379085,
     26: 0.0050607843137254905,
     251: 0.005195424836601307,
     24: 0.005199346405228758,
     248: 0.005243137254901961,
     249: 0.0053,
     48: 0.005390196078431372,
     247: 0.005405228758169935,
     252: 0.005511764705882353,
     254: 0.005599346405228758,
     253: 0.005709803921568627,
     245: 0.005832679738562091,
     244: 0.005964705882352941,
     250: 0.00608562091503268,
     243: 0.006169281045751634,
     241: 0.006376470588235294,
     240: 0.006515686274509804,
     246: 0.006598692810457516,
     239: 0.006682352941176471,
     237: 0.007043790849673203,
     235: 0.0074124183006535945,
     215: 0.007559477124183006,
     216: 0.007843790849673203,
     238: 0.007973856209150327,
     217: 0.007988888888888889,
     233: 0.008234640522875817,
     242: 0.008459477124183006,
     236: 0.008605882352941177,
     232: 0.008952287581699346,
     234: 0.008958169934640523,
     219: 0.009194771241830066,
     220: 0.009209803921568628,
     221: 0.009210457516339868,
     49: 0.009449673202614379,
     223: 0.009605228758169935,
     53: 0.009791503267973857,
     231: 0.01001372549019608,
     224: 0.010115686274509804,
     225: 0.010215032679738563,
     227: 0.010351633986928104,
     218: 0.010497385620915033,
     228: 0.010667973856209151,
     222: 0.01108235294117647,
     229: 0.011607843137254902,
     226: 0.012026797385620914,
     230: 0.013360130718954248,
     52: 0.015488235294117646,
     50: 0.0173718954248366,
     51: 0.024058823529411764,
     214: 0.07023660130718955,
     255: 0.1365516339869281,
     0: 0.21423921568627452}



In above code, we have flattened our image to get a 1d vector then we counted the number of times it has occured. Then calculated the probability of occurence. And sorted it.

### Code Assigning


```python

```


```python

```


```python

```


```python

```
