---
title:  "Pyscript: Running Python in Webpages"
date:   2022-05-15 09:29:17 +0545
categories:
    - Python
    - Pyscript
    - Javascript
    
tags:
    - pyscript
    - python
header:
  teaser: assets/pyscript/hello.png
---

Pyscript is a JS library that allows us to write and run Python code on HTML web pages.  In this blog, we will explore PyScript for running Python codes inside our HTML files. It is quite easy to do so. How it works under the hood is not what is being focused on here but what can we do will be. For docs, please visit [here](https://github.com/pyscript/pyscript).

## First Program
* Create an HTML file and on the top, import packages inside the head section.
```html
<html>
    <head>
        <title>PyScript Test</title>
        <link rel="stylesheet" href="https://pyscript.net/alpha/pyscript.css" />
        <script defer src="https://pyscript.net/alpha/pyscript.js"></script> 
    </head>
    
    <body>
        <h1><py-script>print("Hello from PyScript")</py-script></h1>
    </body>
</html>
```

Opening above HTML in browser will show like below:

![]({{site.url}}/assets/pyscript/hello.png)

## Importing Default Package

Let's add below code just below above to print a random number generated between 100 and 200.

```html
<py-script>
import random
print(f"Random: {random.randint(100,200)}")
</py-script>
```

Output will be something like below:

![]({{site.url}}/assets/pyscript/random.png)

## Importing External Libraries
Let's import NumPy. It is not possible to import them by default as they are not installed. Hence we need to include them in the environment section like below.

```html
<html>
    <head>
        <title>PyScript Test</title>
        <link rel="stylesheet" href="https://pyscript.net/alpha/pyscript.css" />
        <script defer src="https://pyscript.net/alpha/pyscript.js"></script> 
    </head>
    
    <body>
        <py-env>
            - numpy
            - sympy
            - matplotlib
        </py-env>

        <py-script>print("Hello from PyScript")</py-script>
        <py-script>
import random
print(f"Random: {random.randint(100,200)}")
        </py-script>
        <py-script>
import numpy as np
import matplotlib.pyplot as plt

x = np.random.randint(100,200,1000)
y = np.random.randint(000,100,1000)
fig = plt.figure(figsize=(5,5))
plt.scatter(x,y)
fig
    </body>
</html>
```

It will take a little bit to see the result like below:

![]({{site.url}}/assets/pyscript/scatter.png)

### Pandas Dataframe
Just like we used NumPy above, we can include pandas too and let's see it in action.

```html
<html>
    <head>
        <title>PyScript Test</title>
        <link rel="stylesheet" href="https://pyscript.net/alpha/pyscript.css" />
        <script defer src="https://pyscript.net/alpha/pyscript.js"></script> 
    </head>
    
    <body>
        <py-env>
            - numpy
            - sympy
            - matplotlib
            - pandas
        </py-env>

        <py-script>print("Hello from PyScript")</py-script>
        <py-script>
import random
print(f"Random: {random.randint(100,200)}")
        </py-script>
        <py-script>
import numpy as np
import matplotlib.pyplot as plt

x = np.random.randint(100,200,1000)
y = np.random.randint(000,100,1000)
fig = plt.figure(figsize=(5,5))
plt.scatter(x,y)
fig
        </py-script>
        <py-script>
import pandas as pd
df = pd.DataFrame({"x":x,"y":y})
df
        </py-script>
    </body>
</html>
```

![]({{site.url}}/assets/pyscript/pandas.png)

## Live Code Editor
We can even have our own code editor in the browser using PyScript. It can be done by using the tag `<py-repl>`. Let's see it in action.

```html
<html>
    <head>
        <title>PyScript Test</title>
        <link rel="stylesheet" href="https://pyscript.net/alpha/pyscript.css" />
        <script defer src="https://pyscript.net/alpha/pyscript.js"></script> 
    </head>
    
    <body>
        <py-env>
            - numpy
            - sympy
            - matplotlib
            - pandas
        </py-env>
        <py-repl>
    </body>
</html>
```

We can see something like below in a browser:

![]({{site.url}}/assets/pyscript/repl.png)

What is interesting is that we can run code by shift+enter just like in Jupyter Notebook.

![]({{site.url}}/assets/pyscript/repl_hello.png)

For more content like this one, please subscribe to our [newsletter](https://dataqoil.com/newsletter/).