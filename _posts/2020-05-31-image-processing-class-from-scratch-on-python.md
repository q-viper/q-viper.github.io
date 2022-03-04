---
title: Image Processing Class from Scratch on Python
date: 2020-05-30T01:45:00+05:45
comments_id: 5
header:
  teaser: assets/wp-content/uploads/2020/05/download-1.png
  # image: assets/wp-content/uploads/2020/05/download-1.png
categories:
  - Artificial Intelligence
  - Computer Vision
  - Machine Learning
  - Programming
tags:
  - image processing
---

**Contents**
* TOC
{:toc}


# 1 Writing a Image Processing Codes from Python on Scratch
I might stop to write new blogs in this site so please visit [dataqoil.com](https://dataqoil.com) for more cool stuffs.

What will you do when you suddenly think about `Convolutional Neural Networks from Scratch` while serving cows? For me, I wrote some codes for image processing before thinking about those codes. Once again I am not going to write another `OpenCV` here.

## 1.1 What am I using?
* `Numpy` for array operations
* `imageio` builtin library for reading image
* `warnings` to show warning
* `matplotlib` for visualizing

## 1.2 What this blog includes?
* Converting an image into Grayscale from RGB.
* Convolution of an image using different kernels.

# 2 Steps
* Initializing a `ImageProcessing` class.
* Adding a read method
* Adding a show method
* Adding color converison method
* Adding a convolution method

## Initializing a `ImageProcessing` class

``` python
class ImageProcessing:
    def __init__(self):
        self.readmode = {1 : "RGB", 0 : "Grayscale"} # this dictionary will hold readmode values
```  

## Adding a read method
``` python
    def read_image(self, location = "", mode = 1):
        """
            Uses imageio on back.
            location: Directory of image file.
            mode: Image readmode{1 : RGB, 0 : Grayscale}.
        """
        img = imageio.imread(location)
        if mode == 1:
            img = img
        elif mode == 0:
            img = 0.21 * img[:,:,0] + 0.72 * img[:,:,1] + 0.07 * img[:,:,2]
        elif mode == 2:
            pass
        else:
            raise ValueError(f"Readmode not understood. Choose from {self.readmode}.")
        return img
```

* This method only wraps the `imageio`, but I am applying a concept of `RGB` to `GRAYSCALE` conversion.
* By default, imageio reads on RGB format.
* A typical `RGB` to `GRAYSCALE` can be done on below concepts ([taken from](https://www.tutorialspoint.com/dip/grayscale_to_rgb_conversion.htm)):-
    * Average Method:
        \begin{equation}
        Grayscale = \frac{R + G + B}{3}
        \end{equation}
        All channels are given 33% contribution.
    * Weighted Method of luminosity method
        \begin{equation}
        Grayscale = 0.3*R + 0.59*G + 0.11*B
        \end{equation}
        Red channel have 30%, Green have 59 and Blue have 11% contribution.\
        But I am using different version of method ([taken from](https://www.johndcook.com)).
* If user enter different mode, then raise error.



## Adding a show method
``` python
def show(self, image, figsize=(5, 5)):
    """
        Uses Matplotlib.pyplot.
        image: A image to be shown.
        figsize: How big image to show. From plt.figure()
    """
    fig = plt.figure(figsize=figsize)
    im = image
    plt.imshow(im, cmap='gray')
    plt.show()
```
Nothing to say here, docstring is enough.

## Color conversion
``` python
    def convert_color(self, img, to=0):
        if to==0:
            return  0.21 * img[:,:,0] + 0.72 * img[:,:,1] + 0.07 * img[:,:,2]
        else:
            raise ValueError("Color conversion can not understood.")
```
I have still have not thought about grayscale to RGB conversion. But even using `OpenCV` `cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)`, we can not get complete BGR image.

## Adding a convolution method
``` python
    def convolve(self, image, kernel = None, padding = "zero", stride=(1, 1), show=False, bias = 0):
        """
            image: A image to be convolved.
            kernel: A filter/window of odd shape for convolution. Used Sobel(3, 3) default.
            padding: Border operation. Available from zero, same, none. 
            stride: How frequently do convolution?
            show: whether to show result
            bias: a bias term(used on Convolutional NN)
        """
        if len(image.shape) > 3:
            raise ValueError("Only 2 and 3 channel image supported.")
        if type(kernel) == type(None):
            warnings.warn("No kernel provided, trying to apply Sobel(3, 3).")
            kernel = np.array([[1, 0, -1],
                              [1, 0, -1],
                              [1, 0, -1]])
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
            padded_img = np.vstack((zeros_h, image, zeros_h)) # add rows
            padded_img = np.hstack((zeros_v, padded_img, zeros_v)) # add cols
            image = padded_img
            shape = image.shape   
        elif padding == "same":
            h1 = image[0].reshape(-1, shape[1])
            h2 = image[-1].reshape(-1, shape[1])
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
```

What is happening above?
* First the kernel is checked, if not given, used from sobel 3 by 3
* If the given kernel shape is not odd, error is raised.
* For padding, `numpy` stack methods are used.
* Initialize an empty list to store convoluted values
* For convolution, 
    * we loop through every rows in step of kernel's row upto total img rows
    * loop through every cols in step of kernel's col up to total img cols
    * get a current chunk of image and multiply its elements with the kernel's elements
    * if current sum is geater than 255, set it 255
    * append sum to list
* Finally convert the list into array then into right shape.

### Recall the mathematics of Convolution Operation
\begin{equation}
g(x, y) = f(x,y) * h(x,y)
\end{equation}
Where `f` is a image function and `h` is a kernel or mask or filter.

What happens on convolution can be clear from the matrix form of operation.
Lets take a image of `5X5` and kernel of `3X3` sobel y.

$$
\begin{equation*}
f(x, y) = 
\begin{pmatrix}
1 & 10 & 11 & 20 & 30\\12 & 200 & 152 & 223 & 60 \\100 & 190 & 11 & 20 & 10\\102 & 207 & 102 & 223 & 50 \\18 & 109 & 117 & 200 & 30\\\end{pmatrix}\ and\
h(x, y) = \begin{pmatrix}
-1 & 0 & 1\\-1 & 0 & 1\\-1 & 0 & 1\\\end{pmatrix}
\end{equation*}
$$


We have to move the kernel over the each and every pixels of the image from top left to bottom. Placing a kernel over a image and taking a elementwise matrix multiplication of the kernel and chunk of image of the kernel shape. For most cases, we use odd shaped kernel. By using odd shaped kernel, we can place a center of kernel to the center of image chunk.

Now we try to start from the top right pixel, but since our kernel is 3 by 3, we don't have any pixels that will be facing the 1st row of kernel. So we have to work with the concept of `padding` or we will loose those pixels of the border. For the sake of simplicity, lets take a `zero padding`.

$$
\begin{equation}
padded\ f(x, y) = 
\begin{pmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 10 & 11 & 20 & 30 & 0\\
0 & 12 & 200 & 152 & 223 & 60 & 0 \\
0 & 100 & 190 & 11 & 20 & 10 & 0\\
0 & 102 & 207 & 102 & 223 & 50  & 0\\
0 & 18 & 109 & 117 & 200 & 30 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 
\end{pmatrix}
\end{equation}
$$

Now the first chunk of image will be:

$$
\begin{pmatrix}
0 & 0 & 0 \\
0 & 1 & 10 \\
0 & 12 & 200 \\
\end{pmatrix}
$$

Now the convolution operation:

$$
\begin{equation}
\begin{pmatrix}
0 & 0 & 0 \\
0 & 1 & 10 \\
0 & 12 & 200 \\
\end{pmatrix} * \begin{pmatrix}
-1 & 0 & 1\\
-1 & 0 & 1\\
-1 & 0 & 1\\
\end{pmatrix}= 0*-1+0*0+0*1+0*-
1+1*0+10*1+0*-1+12*0+200*1
 = 210
\end{equation}
$$

Similarly, the final image will be like below after sliding through row then column:

$$
\begin{pmatrix}
210 & 150 & 213 & 0 & 0\\
400 & 61 & 43 & 0 & 0 \\
597 & 51 & 0 & 0 & 0\\
506 & 10 & 0 & 0 & 0 \\
316 & 99 & 107 & 0 & 0\\
\end{pmatrix}
$$

But we will set 255 to all values which exceeds 255.

A better visualisation of a convolution operation can be seen by below gif(i don't own this gif):-


Finally, visualizing our convolutated image:-
``` python
    ip = ImageProcessing()
    img = np.array([1, 10, 11, 200, 30, 12, 200, 152, 223, 60, 100, 
                190, 11, 20, 10, 102, 207, 102, 223, 50, 18, 109, 117, 200, 30]).reshape(5, 5)
    cv = ip.convolve(img)
    ip.show(cv)
```
If we printed the output of this code, i.e. `cv`, then we will see the array just like above.

