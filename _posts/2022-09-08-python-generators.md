---
title:  "Python Generators"
date:   2022-09-08 01:29:17 +0545
categories:
    - Python
tags:
    - Generators
header:
  teaser: assets/python/pygen.png
---
Python Generators are kind of `iterators` which allows us to iterate through the values returned through the function using `yield` keyword. In simple words, generators are the function with `yield` keyword instead of `return`. Some of benefits of using Python Generators are:
* They return data as a iterator rather than a whole sequence making it memory efficient when we are working with large data files. One application can be seen in Deep Learning where we have to read many images from the drive in the array form and pass that to our model for training. Putting all images at once in a variable will crash our memory but we will store only a batch of image at once to reduce it. Its done using Generators. [One can be found here by us](https://dataqoil.com/2020/10/20/corn-leaf-infection-detection-data-preprocessing-and-custom-datagenerator/).

## Definition
Its simple :)

```python
def my_gen(...):
    yield ..
```


## Lets Make One


```python
def my_pow_gen(x: int):
    for i in range(x):
        yield i*i
```


```python
def my_pow_fun(x: int):
    return [i*i for i in range(x)]
```

### Test It


```python
my_pow_fun(5), my_pow_gen(5)
```




    ([0, 1, 4, 9, 16], <generator object my_pow_gen at 0x0000023B5F56E970>)



In above example, there are two functions and one represents a generator and another is a function. Both are defined to do same thing i.e. return the power of a sequence up to the given number. While calling function, it gave us the values which is pretty clear but when calling generator, it gave us simple object info. Its not executed yet. To get the next item on the top of generator, we can do `next(generator)`. Its like opposite of the pop operation.


```python
# lets put generator object in a variable.
mpg = my_pow_gen(5)
```

### Getting Item


```python
next(mpg)
```




    0



It gave us 0th item. Now doing again next, it will give next item.


```python
next(mpg)
```




    1



Now doing again, it will move on to next item.


```python
next(mpg)
```




    4



We can loop into mpg too.


```python
for v in mpg:
    print(v)
```

    9
    16
    

But it started to run a loop from the next available item as all others are retrieved already. And trying to get next item now will return an error.


```python
next(mpg)
```


    ---------------------------------------------------------------------------

    StopIteration                             Traceback (most recent call last)

    <ipython-input-25-7428b49b111d> in <module>
    ----> 1 next(mpg)
    

    StopIteration: 


Which simply means that there is nothing left for the iteration.

### Easy Way to Retrieve items

Another way we could iterate over the generators easily is by wrapping them inside a `list()`.


```python
mpg = my_pow_gen(10)
```


```python
list(mpg)
```




    [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]



But doing this clearly violates the purpose of using generator to save memory in the first place.

Thats all for this short blog today, and will keep sharing some small blog in coming days too. Thanks.


```python

```
