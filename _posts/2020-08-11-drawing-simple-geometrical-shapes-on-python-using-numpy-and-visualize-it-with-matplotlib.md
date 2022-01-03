---
title: Drawing Simple Geometrical Shapes on Python Using NumPy and Visualize it with Matplotlib
date: 2020-08-11T00:50:26+05:45
header:
  teaser: assets/wp-content/uploads/2020/08/circle2.png
categories:
  - Computer Vision
  - Programming
  - Project
tags:
  - geometry
  - mathematics
  - numpy
  - opencv
  - python computer vision
---
**Contents**
* TOC
{:toc}


# Making Simple Geometrical Shapes on Python using NumPy and Matplotlib
I might stop to write new blogs in this site so please visit [dataqoil.com](https://dataqoil.com) for more awesome blogs about computer vision projects.

Now on this series of task I am going to tackle some of interesting image processing concepts from scratch using Python and then will compare it with popular OpenCV framework. Last time I did Convolution operation from Scratch and RGB to GrayScale conversion, etc. Now is the time to draw circle, rectangle, ellipse and get the flashback of childhood. I am highly inspired by the book named <b>Image Operators: Image Processing in Python by Jason M. Kinser.</b> In fact I am going to use some simple geometrical concepts to draw these basic shapes using only NumPy and Matplotlib.

Also I have to mention the awesome book named <b> The Journey of X: A Guided Tour of Mathematics by Steven Strogatz.</b> Author really have a great way of describing the mathematical terms and I have learned lot of concepts on Mathematics from there. And author also introduced to the awesome book named <b>The House Keeper and the Professor</b>.

The method I am including here will be added to the previous Image Processing Class (which is also given below) I have used to do Convolution and Colorspace changes. So it will be helpful to view that one also.
* [Writing a Image Processing Class from Scratch on Python]({{site.url}}/2020/05/30/image-processing-class-from-scratch-on-python/)

## What will I do here?
* Using primary grade mathematics, I will create a simple methods to draw/create a simple geometric shapes and compare them with OpenCV's own methods.


```python
import imageio
import warnings
import numpy as np
import matplotlib.pyplot as plt
import cv2
%matplotlib inline

```


```python
class ImageProcessing:
    def __init__(self):
        self.readmode = {1 : "RGB", 0 : "Grayscale"}
    
    def read_image(self, location = "", mode = 1):
        """
            Uses imageio on back.
            * location: Directory of image file.
            * mode: Image readmode{1 : RGB, 0 : Grayscale}.
        """
        
        img = imageio.imread(location)
        if mode == 1:
            img = img
        elif mode == 0:
            img = self.convert_color(img, 0)
        elif mode == 2:
            pass
        else:
            raise ValueError(f"Readmode not understood. Choose from {self.readmode}.")
        return img
    
    def show(self, image, figsize=(5, 5)):
        """
            Uses Matplotlib.pyplot.
            * image: A image to be shown.
            * figsize: How big image to show. From plt.figure()
            
        """
        fig = plt.figure(figsize=figsize)
        im = image
        plt.imshow(im, cmap='gray')
        plt.show()
    
    def convert_color(self, img, to=0):
        if to==0:
            return  0.21 * img[:,:,0] + 0.72 * img[:,:,1] + 0.07 * img[:,:,2]
        else:
            raise ValueError("Color conversion can not understood.")
    
    def convolve(self, image, kernel = None, padding = "zero", stride=(1, 1), show=False, bias = 0):
        """
            * image: A image to be convolved.
            * kernel: A filter/window of odd shape for convolution. Used Sobel(3, 3) default.
            * padding: Border operation. Available from zero, same, none. 
            * stride: How frequently do convolution? 
        """
        
        if len(image.shape) > 3:
            raise ValueError("Only 2 and 3 channel image supported.")
        if type(kernel) == type(None):
            warnings.warn("No kernel provided, trying to apply Sobel(3, 3).")
            kernel = np.array([[-1, 0, 1],
                              [-1, 0, 1],
                              [-1, 0, 1]])
            kernel += kernel.T
        kshape = kernel.shape
        if kshape[0] % 2 != 1 or kshape[1] % 2 != 1:
            raise ValueError("Please provide odd length of 2d kernel.")
        
        if type(stride) == int:
                 stride = (stride, stride)
        
        shape = image.shape
        
        if padding == "zero":
            zeros_h = np.zeros(shape[1]).reshape(-1, shape[1])
            zeros_v = np.zeros(shape[0]+2).reshape(shape[0]+2, -1)

            #zero padding
            padded_img = np.vstack((zeros_h, image, zeros_h)) # add rows
            # print(padded_img)
            padded_img = np.hstack((zeros_v, padded_img, zeros_v)) # add cols

            image = padded_img
            shape = image.shape
            
        elif padding == "same":
            h1 = image[0].reshape(-1, shape[1])
            h2 = image[-1].reshape(-1, shape[1])


            #zero padding
            padded_img = np.vstack((h1, image, h2)) # add rows

            v1 = padded_img[:, 0].reshape(padded_img.shape[0], -1)
            v2 = padded_img[:, -1].reshape(padded_img.shape[0], -1)

            padded_img = np.hstack((v1, padded_img, v2)) # add cols

            image = padded_img
            shape = image.shape
        elif padding == None:
            pass

        rv = 0
        cimg = []
        for r in range(kshape[0], shape[0]+1, stride[0]):
            cv = 0
            for c in range(kshape[1], shape[1]+1, stride[1]):
                chunk = image[rv:r, cv:c]
                soma = (np.multiply(chunk, kernel)+bias).sum()
                try:
                    chunk = int(soma)
                except:
                    chunk = int(0)
                if chunk < 0:
                    chunk = 0
                if chunk > 255:
                    chunk = 255
                cimg.append(chunk)
                cv+=stride[1]
            rv+=stride[0]
        cimg = np.array(cimg, dtype=np.uint8).reshape(int(rv/stride[0]), int(cv/stride[1]))
        if show:
            print(f"Image convolved with \nKernel:{kernel}, \nPadding: {padding}, \nStride: {stride}")
        return cimg

ip = ImageProcessing()
img = ip.read_image("../cb.jpg", mode=0)
cv = ip.convolve(img)
ip.show(cv)
```

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:52: UserWarning: No kernel provided, trying to apply Sobel(3, 3).
    


![png]({{site.url}}/assets/drawing-scratch/output_3_1.png)



```python
# lets read our new image that we are going to use for drawing simple shape
img = ip.read_image("../dog.jpg")
ip.show(img)
```


![png]({{site.url}}/assets/drawing-scratch/output_4_0.png)


## Circle
Everyone knows what is circle but only few cares how did it originated? Thanks to Euclid and his contribution to the modern Mathematics. Circle on simple term can be thought of as a shape where infinite points are present and the distance between two consecutive two points is infinitesimally small. The Computer Graphics doesn't cares about that what it needs is a number. So if we zoom the shapes we start to see the pixels crystal clear. Here I will be using a simple concept of drawing a circle. I will be using the concept of polar form. If I have to write it on steps then:-
* Read an input image, get a radius for circle, get a center point, get a border, get a smoothness value and get a color value for it.
* Prepare smoothness * 360 angles for circle (of course 0 to 360).
* For each angle:
    * Convert angle to radian from degree (NumPy geometrical functions takes radian).
    * Find the distance between two points on circumference by Pythagoras theorem.     
    * Find the new point on circumference and make that point's color on circle color if point is on first quadrant.

![png]({{site.url}}/assets/drawing-scratch/circle.png)

Lets take an example of above circle on 2d plane. Circle's center is on `(h, k)` its radius is `r`, and there are 2 points on circumference `p1, p2`, and third point is drawn on radius line joining `p2` and center. Additionally, `p3` is perpendicular to line joining `p2` and center. Here we know length of 2 lines but not the line `p1p3`. But when the points `p1` and `p2` are so near that the distance between them is nearly zero (or limit tends to 0) then the point `p3` will be at `p2`. At that time we can apply Pythagoras theorem. Below figure show a zoomed version of that situation.

![png]({{site.url}}/assets/drawing-scratch/circle2.png)

But what we need is the coordinate values of `p1`. We can do that by thinking as `c` is on origin. then the x coordinate of `p1` will be equal to x coordinate of `p3`. And to find x-coordinate of `p3` we can solve.


$$
cos(\theta) = \frac{b}{h}\\
x = b = cos(\theta) * h
$$

Similarly,

$$
sin(\theta) = \frac{p}{h}\\
y = p = sin(\theta) * h
$$

And on our case, when the circle is not on center then our `(x, y)` coordinate or `p1` will be `(h, k)` far from plane's center.

Hence, coordinate value for `p1` will be:

$$
x = h + cos(\theta) * r\\
y = k + sin(\theta) * r
$$

And on image plane, the coordinate starts from (0, 0) and we don't have -ve quadrant. Hence we ignore all (x, y) values that lies other than first quadrant. Enough of this theory, lets write that on code.

All the stories given above is already found on the polar form.

$$
x = cos(\theta) * r\\
y = sin(\theta) * r\\
and,\\
r = \sqrt{x^2 + y^2}\\
and,\\
\theta = tan^{-1}(\frac{y}{x})
$$



```python
# creating a circle
def circle(img=None, center=(0, 0), rad=10, border=4, color=[1], smooth=2):
    """
        A method to create a circle on a give image.
        img: Expects numpy ndarray of image. 
        center: center of a circle
        rad: radius of a circle
        border: border of the circle, if -ve, circle is filled
        color: color for circle
        smooth: how smooth should our circle be?(smooth * 360 angles in 0 to 360)
    """
    if type(img) == None:
        raise ValueError("Image can not be None. Provide numpy array instead.")
    ix = center[0]+rad
    angles = 360
    cvalue = np.array(color)
    if type(img) != type(None):
        shape = img.shape
        if len(shape) == 3:
            row, col, channels = shape
        else:
            row, col = shape
            channels = 1
        angles = np.linspace(0, 360, 360*smooth)
        for i in angles:
            a = i*np.pi/180
            y = center[1]+rad*np.sin(a) # it is p=h*sin(theta)
            x = center[0]+rad*np.cos(a)
            
            # since we are wroking on image, coordinate starts from (0, 0) onwards and we have to ignore -ve values
            
        
            if border >= 0:
                b = int(np.ceil(border/2))
                
                x1 = np.clip(x-b, 0, shape[0]).astype(np.int32)
                y1 = np.clip(y-b, 0, shape[1]).astype(np.int32)
                x2 = np.clip(x+b, 0, shape[0]).astype(np.int32)
                y2 = np.clip(y+b, 0, shape[1]).astype(np.int32)
            
                
                img[x1:x2, y1:y2] = cvalue

            else:
                x = np.clip(x, 0, shape[0])
                y = np.clip(y, 0, shape[1])
                r, c = int(x), int(y)
                
                if i > 270:
                    img[center[0]:r, c:center[1]] = cvalue
                elif i > 180:
                    img[r:center[0], c:center[1]] = cvalue
                elif i > 90:
                    img[r:center[0], center[1]:c] = cvalue
                elif i > 0:
                    img[center[0]:r, center[1]:c] = cvalue

        return img
        
img = ip.read_image("../dog.jpg")
#ip.show(img)
fig = plt.figure(figsize=(5,5))
mimg = circle(img, center=(400, 100), border=20, rad=500)
ip.show(mimg)
```


    <Figure size 360x360 with 0 Axes>



![png]({{site.url}}/assets/drawing-scratch/output_8_1.png)


Let me explain little bit of the code above.
* Check the inputs and loop for the angles.
* We will try to take as much as possible angles given by smoothness value.
* Take a coordinate value for a point to draw on.
* If the point lies on first quadrant:
    * If the border value is +ve then draw only within the pixel and border/2 neighbor pixels on each of 4 directions.
    * Else:
        * If angle is greater than 270 then fill 4th quadrant with color
        * If angle is greater than 190 then fill 3rd quadrant with color
        * If angle is greater than 90 then fill 2nd quadrant with color
        * If angle is greater than 0 then fill 1st quadrant with color

![png]({{site.url}}/assets/drawing-scratch/circle fill.png)

### Compare it with OpenCV's Circle
Before comparing with OpenCV, lets have a clear understanding of 2d graph plane and image plane. Image plane starts from the top left side but 2d graph plane starts from the center to upwards. Hence in order to compare our circle, we have to change the center value (in this case).
![png]({{site.url}}/assets/drawing-scratch/planes.png)

And on this case, I am just swapping center values (i.e. `(x, y)` for OpenCV and `(y, x)` for ours).

* Draw circle on image using OpenCV
* Draw circle on same image using our method.
* Subtract drawn images
* Show the difference.

The common parts on images are shown complete black and those which are not are shown comlete white.


```python
# read image
img = ip.read_image("../dog.jpg")
# draw using opencv
print("OpenCV")
cimg = cv2.circle(img.copy(), (400, 1000), 500, [0, 0, 0], -20)
ip.show(cimg)

# draw using our method (swap center)
print("Ours")
mimg = circle(img, center=(1000, 400), border=-20, rad=500, color=[0, 0, 0])
ip.show(mimg)

# difference
print("Difference")
r = mimg-cimg
r[r!=[0, 0, 0]] = 255
ip.show(r)

# count difference pixels
diff = np.sum(mimg!=cimg)
shape = mimg.shape

# what percentage is different?
diff * 100 / (shape[0] * shape[1])
```

    OpenCV
    


![png]({{site.url}}/assets/drawing-scratch/output_11_1.png)


    Ours
    


![png]({{site.url}}/assets/drawing-scratch/output_11_3.png)


    Difference
    


![png]({{site.url}}/assets/drawing-scratch/output_11_5.png)





    0.2912136361400096



It seems clear that only 0.29% of pixels were different than the result of OpenCV's circle. But the difference varies with the shape of circle.


```python
# read image
img = ip.read_image("../dog.jpg")
# draw using opencv
print("OpenCV")
cimg = cv2.circle(img.copy(), (900, 1000), 500, [0, 0, 0], 20)
ip.show(cimg)

# draw using our method (swap center)
print("Ours")
mimg = circle(img.copy(), center=(1000, 900), border=20, rad=500, color=[0, 0, 0])
ip.show(mimg)

# difference
print("Difference")
r = mimg-cimg
r[r!=[0, 0, 0]] = 255
ip.show(r)

# count difference pixels
diff = np.sum(mimg!=cimg)
shape = mimg.shape

# what percentage is different?
diff * 100 / (shape[0] * shape[1])
```

    OpenCV
    


![png]({{site.url}}/assets/drawing-scratch/output_13_1.png)


    Ours
    


![png]({{site.url}}/assets/drawing-scratch/output_13_3.png)


    Difference
    


![png]({{site.url}}/assets/drawing-scratch/output_13_5.png)





    1.064850381662056



## Rectangle
Drawing a Rectangle is very easy, in fact just an array indexing completes the task. We need coordinates of two opposite corners i.e. major diagonal. Top left and bottom right corner's coordinate is required on this case. And we will perform array indexing. Same as on above case, we will work with border and color values.


```python
def rectangle(img, pt1, pt2, border=2, color=[0]):
    """
        img: Input image where we want to draw rectangle:
        pt1: top left point (y, x)
        pt2: bottom right point
        border: border of line
        color: color of rectangle line,
        returns new image with rectangle.
        
    """
    p1 = pt1
    pt1 = (p1[1], p1[0])
    p2 = pt2
    pt2 = (p2[1], p2[0])
    b = int(np.ceil(border/2))
    cvalue = np.array(color)
    if border >= 0:
        # get x coordinates for each line(top, bottom) of each side
        # if -ve coordinates comes, then make that 0
        x11 = np.clip(pt1[0]-b, 0, pt2[0])
        x12 = np.clip(pt1[0]+b+1, 0, pt2[0])
        x21 = pt2[0]-b
        x22 = pt2[0]+b+1

        y11 = np.clip(pt1[1]-b, 0, pt2[1])            
        y12 = np.clip(pt1[1]+b+1, 0, pt2[1])   
        y21 = pt2[1]-b
        y22 = pt2[1]+b+1
        # right line
        img[x11:x22, y11:y12] = cvalue
        #left line
        img[x11:x22, y21:y22] = cvalue
        # top line
        img[x11:x12, y11:y22] = cvalue
        # bottom line
        img[x21:x22, y11:y22] = cvalue
        
    else:
        pt1 = np.clip(pt1, 0, pt2)
        img[pt1[0]:pt2[0]+1, pt1[1]:pt2[1]+1] = cvalue
        
    return img

mimg = rectangle(img, (100,500), (1000, 1000), border=-5, color=[20, 150, 20])
ip.show(mimg)
```


![png]({{site.url}}/assets/drawing-scratch/output_15_0.png)


Lets explain little bit of the code here:-
* Take an image where we want to draw, take coordinates of corners, take border of rectangle, take color of rectangle.
* Extract the coordinates where we want to draw (if the coordinates is out of image plane then perform clipping)
* If border is +ve:
    * Change pixels on topmost line(top coordinate along with its some neighbors)
    * Change pixels on bottommost line(top coordinate along with its some neighbors)
    * Follow same for other lines.
* Else:
    * Fill/Change color from top line to bottom from left to right line.

## Compare with OpenCV


```python
# read image
img = ip.read_image("../dog.jpg")
# draw using opencv
print("OpenCV")
cimg = cv2.rectangle(img.copy(), (100, 500), (1000, 1000), [0, 0, 0], -5)
ip.show(cimg)

# draw using our method (swap center)
print("Ours")
mimg = rectangle(img, (100,500), (1000, 1000), border=-5, color=[0, 0, 0])
ip.show(mimg)

# difference
print("Difference")
r = mimg-cimg
r[r!=[0, 0, 0]] = 255
ip.show(r)

# count difference pixels
diff = np.sum(mimg!=cimg)
shape = mimg.shape

# what percentage is different?
diff * 100 / (shape[0] * shape[1])
```

    OpenCV
    


![png]({{site.url}}/assets/drawing-scratch/output_18_1.png)


    Ours
    


![png]({{site.url}}/assets/drawing-scratch/output_18_3.png)


    Difference
    


![png]({{site.url}}/assets/drawing-scratch/output_18_5.png)





    0.0



The comparison with OpenCV seems to be great because we have 0 difference. You can try with different sizes of rectangle.

## Ellipse
Ellipse is a modified version of circle but it is well described as the portion that lies on a 2d plane when a plane is inclined inside a cone. Please search about this to see the bunch of images. I will again be using the polar form of ellipse. It is just as simple as the circle's except we use axis instead of radius. 

$$
x = h + cos(\theta) * a\\
y = k + sin(\theta) * b
$$

A simple example can be done using Matplotlib's plot.


```python
h = 2
k = 1
a = 3
b = 1

t = np.linspace(0, 2 * np.pi, 100)
plt.plot(h+a*np.cos(t), k+b*np.sin(t))
plt.plot()
```




    []




![png]({{site.url}}/assets/drawing-scratch/output_21_1.png)



```python
# creating a ellipse
def ellipse(img=None, center=(0, 0), a=3, b=1, border=4, color=[0], smooth=2):
    """
        A method to create a ellipse on a give image.
        img: Expects numpy ndarray of image. 
        center: center of a ellipse
        a: major axis
        b: minor axis
        border: border of the ellipse, if -ve, ellipse is filled
        color: color for ellipse
        smooth: how smooth should our ellipse be?(smooth * 360 angles in 0 to 360)
    """
    if type(img) == None:
        raise ValueError("Image can not be None. Provide numpy array instead.")
    angles = 360
    cvalue = np.array(color)
    if type(img) != type(None):
        shape = img.shape
        if len(shape) == 3:
            row, col, channels = shape
        else:
            row, col = shape
            channels = 1
        angles = np.linspace(0, 360, 360*smooth)
        for i in angles:
            angle = i*np.pi/180
            y = center[1]+b*np.sin(angle) 
            x = center[0]+a*np.cos(angle)
            
            
            # since we are wroking on image, coordinate starts from (0, 0) onwards and we have to ignore -ve values
            if border >= 0:
                r, c = int(x), int(y)
                bord = int(np.ceil(border/2))
                x1 = np.clip(x-bord, 0, img.shape[0]).astype(np.int32)
                y1 = np.clip(y-bord, 0, img.shape[1]).astype(np.int32)
                x2 = np.clip(x+bord, 0, img.shape[0]).astype(np.int32)
                y2 = np.clip(y+bord, 0, img.shape[1]).astype(np.int32)
                
                img[x1:x2, y1:y2] = cvalue
                
            else:
                x = np.clip(x, 0, img.shape[0])
                y = np.clip(y, 0, img.shape[1])
                r, c = int(x), int(y)
                if i > 270:
                    img[center[0]:r, c:center[1]] = cvalue
                elif i > 180:
                    img[r:center[0], c:center[1]] = cvalue
                elif i > 90:
                    img[r:center[0], center[1]:c] = cvalue
                elif i > 0:
                    img[center[0]:r, center[1]:c] = cvalue
                     
        return img
        
    

mimg = np.zeros((100, 100, 3), dtype=np.int32) + 255
eimg = ellipse(mimg.copy(), center=(10, 30), a = 10, b = 40, border=-2, color=[0, 0, 0])
ip.show(eimg)
```


![png]({{site.url}}/assets/drawing-scratch/output_22_0.png)


## Compare it with OpenCV's Ellipse
Again, the case is just like circle's, we have to swap center and the axes for the ellipse.


```python
cimg = cv2.ellipse(mimg.copy(), (30, 10), (40, 10), 0, 0, 360, [0, 0, 0], -2)
ip.show(cimg)
```


![png]({{site.url}}/assets/drawing-scratch/output_24_0.png)



```python
# difference on fill
diff = np.sum(cimg!=eimg)
shape = cimg.shape

# what percentage is different?
diff * 100 / (shape[0] * shape[1])
```




    4.47




```python
# difference on normal
mimg = np.zeros((100, 100, 3), dtype=np.int32) + 255
eimg = ellipse(mimg.copy(), center=(20, 30), a = 10, b = 40, border=2, color=[0, 0, 0])
print("Ours")
ip.show(eimg)

# opencv's
print("OpenCV's")
cimg = cv2.ellipse(mimg.copy(), (30, 20), (40, 10), 0, 0, 360, [0, 0, 0], 2)
ip.show(cimg)

# difference on fill
diff = np.sum(cimg!=eimg)
shape = cimg.shape

# what percentage is different?
diff * 100 / (shape[0] * shape[1])
```

    Ours
    


![png]({{site.url}}/assets/drawing-scratch/output_26_1.png)


    OpenCV's
    


![png]({{site.url}}/assets/drawing-scratch/output_26_3.png)





    8.55



The difference of OpenCV's and our method's output is not that bad. But as always, the difference depends on the size of the shape.

## Finally
We have written simple methods to perform basic geometric shapes drawing. Now on bonus topic I will add these methods on our Image Processing class.

## Bonus Topic



```python
class ImageProcessing:
    def __init__(self):
        self.readmode = {1 : "RGB", 0 : "Grayscale"}
    
    def read_image(self, location = "", mode = 1):
        """
            Uses imageio on back.
            * location: Directory of image file.
            * mode: Image readmode{1 : RGB, 0 : Grayscale}.
        """
        
        img = imageio.imread(location)
        if mode == 1:
            img = img
        elif mode == 0:
            img = self.convert_color(img, 0)
        elif mode == 2:
            pass
        else:
            raise ValueError(f"Readmode not understood. Choose from {self.readmode}.")
        return img
    
    def show(self, image, figsize=(5, 5)):
        """
            Uses Matplotlib.pyplot.
            * image: A image to be shown.
            * figsize: How big image to show. From plt.figure()
            
        """
        fig = plt.figure(figsize=figsize)
        im = image
        plt.imshow(im, cmap='gray')
        plt.show()
    
    def convert_color(self, img, to=0):
        if to==0:
            return  0.21 * img[:,:,0] + 0.72 * img[:,:,1] + 0.07 * img[:,:,2]
        else:
            raise ValueError("Color conversion can not understood.")
    
    # creating a circle
    def circle(self, img=None, center=(0, 0), rad=10, border=4, color=[1], smooth=2):
        """
            A method to create a circle on a give image.
            img: Expects numpy ndarray of image. 
            center: center of a circle
            rad: radius of a circle
            border: border of the circle, if -ve, circle is filled
            color: color for circle
            smooth: how smooth should our circle be?(smooth * 360 angles in 0 to 360)
        """
        if type(img) == None:
            raise ValueError("Image can not be None. Provide numpy array instead.")
        ix = center[0]+rad
        angles = 360
        cvalue = np.array(color)
        if type(img) != type(None):
            shape = img.shape
            if len(shape) == 3:
                row, col, channels = shape
            else:
                row, col = shape
                channels = 1
            angles = np.linspace(0, 360, 360*smooth)
            for i in angles:
                a = i*np.pi/180
                y = center[1]+rad*np.sin(a) # it is p=h*sin(theta)
                x = center[0]+rad*np.cos(a)

                # since we are wroking on image, coordinate starts from (0, 0) onwards and we have to ignore -ve values


                if border >= 0:
                    b = int(np.ceil(border/2))

                    x1 = np.clip(x-b, 0, shape[0]).astype(np.int32)
                    y1 = np.clip(y-b, 0, shape[1]).astype(np.int32)
                    x2 = np.clip(x+b, 0, shape[0]).astype(np.int32)
                    y2 = np.clip(y+b, 0, shape[1]).astype(np.int32)


                    img[x1:x2, y1:y2] = cvalue

                else:
                    x = np.clip(x, 0, shape[0])
                    y = np.clip(y, 0, shape[1])
                    r, c = int(x), int(y)

                    if i > 270:
                        img[center[0]:r, c:center[1]] = cvalue
                    elif i > 180:
                        img[r:center[0], c:center[1]] = cvalue
                    elif i > 90:
                        img[r:center[0], center[1]:c] = cvalue
                    elif i > 0:
                        img[center[0]:r, center[1]:c] = cvalue

            return img
        
    def rectangle(self, img, pt1, pt2, border=2, color=[0]):
        """
            img: Input image where we want to draw rectangle:
            pt1: top left point (y, x)
            pt2: bottom right point
            border: border of line
            color: color of rectangle line,
            returns new image with rectangle.

        """
        p1 = pt1
        pt1 = (p1[1], p1[0])
        p2 = pt2
        pt2 = (p2[1], p2[0])
        b = int(np.ceil(border/2))
        cvalue = np.array(color)
        if border >= 0:
            # get x coordinates for each line(top, bottom) of each side
            # if -ve coordinates comes, then make that 0
            x11 = np.clip(pt1[0]-b, 0, pt2[0])
            x12 = np.clip(pt1[0]+b+1, 0, pt2[0])
            x21 = pt2[0]-b
            x22 = pt2[0]+b+1

            y11 = np.clip(pt1[1]-b, 0, pt2[1])            
            y12 = np.clip(pt1[1]+b+1, 0, pt2[1])   
            y21 = pt2[1]-b
            y22 = pt2[1]+b+1
            # right line
            img[x11:x22, y11:y12] = cvalue
            #left line
            img[x11:x22, y21:y22] = cvalue
            # top line
            img[x11:x12, y11:y22] = cvalue
            # bottom line
            img[x21:x22, y11:y22] = cvalue

        else:
            pt1 = np.clip(pt1, 0, pt2)
            img[pt1[0]:pt2[0]+1, pt1[1]:pt2[1]+1] = cvalue

        return img

    # creating a ellipse
    def ellipse(self, img=None, center=(0, 0), a=3, b=1, border=4, color=[0], smooth=2):
        """
            A method to create a ellipse on a give image.
            img: Expects numpy ndarray of image. 
            center: center of a ellipse
            a: major axis
            b: minor axis
            border: border of the ellipse, if -ve, ellipse is filled
            color: color for ellipse
            smooth: how smooth should our ellipse be?(smooth * 360 angles in 0 to 360)
        """
        if type(img) == None:
            raise ValueError("Image can not be None. Provide numpy array instead.")
        angles = 360
        cvalue = np.array(color)
        if type(img) != type(None):
            shape = img.shape
            if len(shape) == 3:
                row, col, channels = shape
            else:
                row, col = shape
                channels = 1
            angles = np.linspace(0, 360, 360*smooth)
            for i in angles:
                angle = i*np.pi/180
                y = center[1]+b*np.sin(angle) 
                x = center[0]+a*np.cos(angle)


                # since we are wroking on image, coordinate starts from (0, 0) onwards and we have to ignore -ve values
                if border >= 0:
                    r, c = int(x), int(y)
                    bord = int(np.ceil(border/2))
                    x1 = np.clip(x-bord, 0, img.shape[0]).astype(np.int32)
                    y1 = np.clip(y-bord, 0, img.shape[1]).astype(np.int32)
                    x2 = np.clip(x+bord, 0, img.shape[0]).astype(np.int32)
                    y2 = np.clip(y+bord, 0, img.shape[1]).astype(np.int32)

                    img[x1:x2, y1:y2] = cvalue

                else:
                    x = np.clip(x, 0, img.shape[0])
                    y = np.clip(y, 0, img.shape[1])
                    r, c = int(x), int(y)
                    if i > 270:
                        img[center[0]:r, c:center[1]] = cvalue
                    elif i > 180:
                        img[r:center[0], c:center[1]] = cvalue
                    elif i > 90:
                        img[r:center[0], center[1]:c] = cvalue
                    elif i > 0:
                        img[center[0]:r, center[1]:c] = cvalue

            return img
        
    
    def convolve(self, image, kernel = None, padding = "zero", stride=(1, 1), show=False, bias = 0):
        """
            * image: A image to be convolved.
            * kernel: A filter/window of odd shape for convolution. Used Sobel(3, 3) default.
            * padding: Border operation. Available from zero, same, none. 
            * stride: How frequently do convolution? 
        """
        
        if len(image.shape) > 3:
            raise ValueError("Only 2 and 3 channel image supported.")
        if type(kernel) == type(None):
            warnings.warn("No kernel provided, trying to apply Sobel(3, 3).")
            kernel = np.array([[-1, 0, 1],
                              [-1, 0, 1],
                              [-1, 0, 1]])
            kernel += kernel.T
        kshape = kernel.shape
        if kshape[0] % 2 != 1 or kshape[1] % 2 != 1:
            raise ValueError("Please provide odd length of 2d kernel.")
        
        if type(stride) == int:
                 stride = (stride, stride)
        
        shape = image.shape
        
        if padding == "zero":
            zeros_h = np.zeros(shape[1]).reshape(-1, shape[1])
            zeros_v = np.zeros(shape[0]+2).reshape(shape[0]+2, -1)

            #zero padding
            padded_img = np.vstack((zeros_h, image, zeros_h)) # add rows
            # print(padded_img)
            padded_img = np.hstack((zeros_v, padded_img, zeros_v)) # add cols

            image = padded_img
            shape = image.shape
            
        elif padding == "same":
            h1 = image[0].reshape(-1, shape[1])
            h2 = image[-1].reshape(-1, shape[1])


            #zero padding
            padded_img = np.vstack((h1, image, h2)) # add rows

            v1 = padded_img[:, 0].reshape(padded_img.shape[0], -1)
            v2 = padded_img[:, -1].reshape(padded_img.shape[0], -1)

            padded_img = np.hstack((v1, padded_img, v2)) # add cols

            image = padded_img
            shape = image.shape
        elif padding == None:
            pass

        rv = 0
        cimg = []
        for r in range(kshape[0], shape[0]+1, stride[0]):
            cv = 0
            for c in range(kshape[1], shape[1]+1, stride[1]):
                chunk = image[rv:r, cv:c]
                soma = (np.multiply(chunk, kernel)+bias).sum()
                try:
                    chunk = int(soma)
                except:
                    chunk = int(0)
                if chunk < 0:
                    chunk = 0
                if chunk > 255:
                    chunk = 255
                cimg.append(chunk)
                cv+=stride[1]
            rv+=stride[0]
        cimg = np.array(cimg, dtype=np.uint8).reshape(int(rv/stride[0]), int(cv/stride[1]))
        if show:
            print(f"Image convolved with \nKernel:{kernel}, \nPadding: {padding}, \nStride: {stride}")
        return cimg
ip = ImageProcessing()
img = ip.read_image("../cb.jpg", mode=0)
cv = ip.convolve(img)
ip.show(cv)

img = ip.read_image("../dog.jpg")
#ip.show(img)
fig = plt.figure(figsize=(5,5))
mimg = ip.circle(img, center=(400, 100), border=20, rad=500)
ip.show(mimg)


mimg = ip.rectangle(img, (100,500), (1000, 1000), border=-5, color=[20, 150, 20])
ip.show(mimg)

mimg = np.zeros((100, 100, 3), dtype=np.int32) + 255
eimg = ip.ellipse(mimg.copy(), center=(10, 30), a = 10, b = 40, border=-2, color=[0, 0, 0])
ip.show(eimg)

```

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:211: UserWarning: No kernel provided, trying to apply Sobel(3, 3).
    


![png]({{site.url}}/assets/drawing-scratch/output_29_1.png)



    <Figure size 360x360 with 0 Axes>



![png]({{site.url}}/assets/drawing-scratch/output_29_3.png)



![png]({{site.url}}/assets/drawing-scratch/output_29_4.png)



![png]({{site.url}}/assets/drawing-scratch/output_29_5.png)


Thank you so much for reading this and if you find it interesting why not share it or leave the comments? If you have any queries then you can send me mail or find me at Twitter as @QuassarianViper.

## What next?
* Add functionality to do blurring, noise cancellation, sharpening etc
* Add functionality to do erosion, dilation etc operations.

In the meantime how about looking over some of mine works?
### Why not read more?
* [Linear Regression from Scratch]({{site.url}}/2020/08/07/writing-a-linear-regression-class-from-scratch-using-python/)
* [Writing Popular ML Optimizers from Scratch]({{site.url}}/2020/06/05/writing-popular-machine-learning-optimizers-from-scratch-on-python/)
* [Feed Forward Neural Network from Scratch]({{site.url}}/2020/05/30/writing-a-deep-neural-network-from-scratch-on-python/)
* [Convolutional Neural Networks from Scratch]({{site.url}}/2020/06/05/convolutional-neural-networks-from-scratch-on-python/)
* [Writing a Simple Image Processing Class from Scratch]({{site.url}}/2020/05/30/image-processing-class-from-scratch-on-python/)
* [Deploying a RASA Chatbot on Android using Unity3d]({{site.url}}/2020/08/04/deploying-a-simple-rasa-chatbot-on-unity3d-project-to-make-a-chatbot-for-android-devices/)
* [Gesture Based Visually Writing System Using OpenCV and Python]({{site.url}}/2020/08/01/gesture-based-visually-writing-system-using-opencv-and-python/)
* [Naive Bayes for text classifications: Scratch to Framework]({{site.url}}/2020/03/04/text-classification-using-naive-bayes-scratch-to-the-framework/)
* [Simple OCR for Devanagari Handwritten Text]({{site.url}}/2020/02/25/building-ocr-for-devanagari-handwritten-character/)
