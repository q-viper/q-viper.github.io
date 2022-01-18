---
title:  "Data Analysis and Importance of Groupby in Pandas but not Just pd.groupby"
date:   2022-01-05 09:29:17 +0545
categories:
  - Data Analysis
  - Pandas
tags:
  - data project
  - pandas
  - resample
  - groupby
header:
  teaser: assets/groupby/output_21_0.png
---

# Data Analysis and Importance of Groupby in Pandas but not Just pd.groupby
This blog will be continously updated as I find new ways, tricks to make things work faster and easier.

## Updates
* January 5 2022
    * Started blog and written up to [**Rate of Views Change Per Month According to Category**](#Rate-of-Views-Change-Per-Month-According-to-Category).

What would you like to become in $y= mx+c$ ? Please don't say `+`.

## Introduction
I have been working with Pandas frequently and most of the time I have to do groupby. But I have noticed that `pd.groupby` is not always what I should do. Before diving into hands on experience, I would like to share some scenarios but first lets assume that you are working in a media company:
1. What if your manager asks you to find the trend of content reach/growth in monthly basis so that they could know whether the contents have desired effect or not? Where you have one datetime column in timestamp format.
2. What if your social media manager asks you to find the top 10 category of post with respect to profession of viewers so that they could make more focused and personalized contents dedicating to them and increase vies.
3. You see there is a chance of being promoted and you want to give some valuable insights? What if you to present a best time to post a particular type of content. For example, a comedy or funny content might get best views during the day, a nature or motivating content might get good views during morning and a loving or musical content might get good views during the night.

Above 3 examples are some high level problem statement but in the ground level, almost every analyst have to group the data. Here in this blog, I am going to create a dummy data and perform some of analysis using groupby with it.

## Creating a Dummy Data
> **The data will be generated randomly and thus it might not make any sense in the realworld but the goal of this blog is to explain/explore ways to do groupby in Pandas.**


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings


sns.set(rc={'figure.figsize':(40, 20),
                "axes.titlesize" : 24,
                "axes.labelsize" : 20,
                "xtick.labelsize" : 16,
                "ytick.labelsize" : 16})
plt.rc("figure", figsize=(16,8))
warnings.filterwarnings('ignore')
```

Lets suppose your number of contents per day ranges from 3 to 7. Your views from the date of publish to 1 week could range from 2k to 100k and it also grows by 0.1% after reaching 100 views. 


```python
dates = pd.date_range(pd.to_datetime("2020-01-01"), pd.to_datetime("2021-01-01"))
times = ["Morning", "Day", "Night"]
categories = ["Motivating", "Musical", "Career", "News", "Funny"]
posts = list(range(5, 11))

content_dict = {"post_id":[], "date":[], "dtime": [], "category":[], "views": []}
post_id = 0

month = []
rate = 0.1

for d in dates:
    post_count = posts[np.random.randint(len(posts))]
    
    for p in range(post_count):
        if len(content_dict["date"])%100==0:
            rate+=0.1
        dtime = times[np.random.randint(len(times))]
        category = categories[np.random.randint(len(categories))]
        views = np.random.randint(20000, 100000) * rate
        
        content_dict["post_id"].append(post_id)
        content_dict["date"].append(d)
        content_dict["dtime"].append(dtime)
        content_dict["category"].append(category)
        content_dict["views"].append(views)
        post_id+=1
        
df = pd.DataFrame(content_dict, columns=list(content_dict.keys()))
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>post_id</th>
      <th>date</th>
      <th>dtime</th>
      <th>category</th>
      <th>views</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2020-01-01</td>
      <td>Night</td>
      <td>Funny</td>
      <td>19536.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2020-01-01</td>
      <td>Day</td>
      <td>Funny</td>
      <td>5048.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2020-01-01</td>
      <td>Night</td>
      <td>News</td>
      <td>13165.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2020-01-01</td>
      <td>Day</td>
      <td>Career</td>
      <td>12326.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2020-01-01</td>
      <td>Morning</td>
      <td>News</td>
      <td>19512.8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2735</th>
      <td>2735</td>
      <td>2021-01-01</td>
      <td>Day</td>
      <td>Musical</td>
      <td>180803.4</td>
    </tr>
    <tr>
      <th>2736</th>
      <td>2736</td>
      <td>2021-01-01</td>
      <td>Morning</td>
      <td>Motivating</td>
      <td>203542.3</td>
    </tr>
    <tr>
      <th>2737</th>
      <td>2737</td>
      <td>2021-01-01</td>
      <td>Morning</td>
      <td>Musical</td>
      <td>161295.1</td>
    </tr>
    <tr>
      <th>2738</th>
      <td>2738</td>
      <td>2021-01-01</td>
      <td>Morning</td>
      <td>Motivating</td>
      <td>143900.9</td>
    </tr>
    <tr>
      <th>2739</th>
      <td>2739</td>
      <td>2021-01-01</td>
      <td>Night</td>
      <td>Career</td>
      <td>84401.6</td>
    </tr>
  </tbody>
</table>
<p>2740 rows × 5 columns</p>
</div>



The data is ready and now we could start our analysis.

## Number of Posts According to Category

Using normal groupby. [More at here](https://pandas.pydata.org/docs/reference/groupby.html).


```python
df.groupby("category").post_id.count().plot(kind="bar")
```




    <AxesSubplot:xlabel='category'>




    
![png]({{site.url}}/assets/groupby/output_8_1.png)
    


## Number of Views According to Category


```python
df.groupby("category").views.sum().plot(kind="bar")
```




    <AxesSubplot:xlabel='category'>




    
![png]({{site.url}}/assets/groupby/output_10_1.png)
    


Pretty easy right?

Lets try something more.

## Views and Count According to Day Time


```python
df.groupby("dtime").post_id.count().plot(kind="bar")
```




    <AxesSubplot:xlabel='dtime'>




    
![png]({{site.url}}/assets/groupby/output_12_1.png)
    



```python
df.groupby("dtime").views.sum().plot(kind="bar")
```




    <AxesSubplot:xlabel='dtime'>




    
![png]({{site.url}}/assets/groupby/output_13_1.png)
    


## Views According to Month

Using resample on date according to month. We could use week, quarter and also more flexible times to resample. [More at here](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html#pandas.DataFrame.resample)


```python
df.resample(rule='M', on='date')["views"].sum().plot(kind="bar")
```




    <AxesSubplot:xlabel='date'>




    
![png]({{site.url}}/assets/groupby/output_15_1.png)
    


In above step, we groupped the data according the month and took sum of views. But will it meet our next requirement?

## Number of Views Per Month According to Category

Using Grouper to groupby month inside a groupby. [More at here](https://pandas.pydata.org/docs/reference/api/pandas.Grouper.html).


```python
df.groupby(["category", pd.Grouper(key="date", freq="1M")]).views.sum().plot(kind="bar")
```




    <AxesSubplot:xlabel='category,date'>




    
![png]({{site.url}}/assets/groupby/output_17_1.png)
    


Why not make our plot little bit more awesome?


```python
vdf = df.groupby(["category", pd.Grouper(key="date", freq="1M")]).views.sum().rename("Views").reset_index()
vdf["date"] = vdf["date"].dt.date

def bar_plot(data, title="Views", xax=None,yax=None, hue=None):

    fig, ax = plt.subplots(figsize = (50, 30))   
    fig = sns.barplot(x = xax, y = yax, data = data, 
                 ci = None, ax=ax, hue=hue)
    plt.legend(fontsize=40)
    plt.yticks(fontsize=40)
    plt.xticks(fontsize=40, rotation=80)
    plt.title(title, fontsize=50)
    plt.xlabel(xax, fontsize=50)
    plt.ylabel(yax, fontsize=50)
    plt.show()
```

I love to make my own custom visualization function. That gives me more flexibility and less time to tune sizes.


```python
bar_plot(vdf, title="Views Plot", xax="date", yax="Views", hue="category")
```


    
![png]({{site.url}}/assets/groupby/output_21_0.png)
    


Can you find some insights or make some argument by looking over above data? Your result will definately be different than mine because of the random data used on above.

## Rate of Views Change Per Month


```python
df.groupby([pd.Grouper(key="date", freq="1M")]).views.sum().pct_change().plot(kind="line")
```




    <AxesSubplot:xlabel='date'>




    
![png]({{site.url}}/assets/groupby/output_24_1.png)
    


## Rate of Views Change Per Month According to Category

Using shift inside the groupby object.


```python
vdf = df.groupby(["category", pd.Grouper(key="date", freq="1M")]).views.sum().rename("Sums").reset_index()
lags = vdf.groupby("category").Sums.shift(1)
vdf["Rate"] = (vdf["Sums"]-lags)/lags
vdf
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>date</th>
      <th>Sums</th>
      <th>Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Career</td>
      <td>2020-01-31</td>
      <td>738533.2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Career</td>
      <td>2020-02-29</td>
      <td>1199889.7</td>
      <td>0.624693</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Career</td>
      <td>2020-03-31</td>
      <td>1913763.8</td>
      <td>0.594950</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Career</td>
      <td>2020-04-30</td>
      <td>3330908.0</td>
      <td>0.740501</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Career</td>
      <td>2020-05-31</td>
      <td>3390679.1</td>
      <td>0.017944</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>58</th>
      <td>News</td>
      <td>2020-08-31</td>
      <td>4054761.3</td>
      <td>0.221326</td>
    </tr>
    <tr>
      <th>59</th>
      <td>News</td>
      <td>2020-09-30</td>
      <td>4865650.4</td>
      <td>0.199984</td>
    </tr>
    <tr>
      <th>60</th>
      <td>News</td>
      <td>2020-10-31</td>
      <td>4976088.2</td>
      <td>0.022697</td>
    </tr>
    <tr>
      <th>61</th>
      <td>News</td>
      <td>2020-11-30</td>
      <td>5903048.6</td>
      <td>0.186283</td>
    </tr>
    <tr>
      <th>62</th>
      <td>News</td>
      <td>2020-12-31</td>
      <td>9044649.8</td>
      <td>0.532200</td>
    </tr>
  </tbody>
</table>
<p>63 rows × 4 columns</p>
</div>




```python
sns.lineplot(data=vdf,x="date", y="Rate", hue="category")
```




    <AxesSubplot:xlabel='date', ylabel='Rate'>




    
![png]({{site.url}}/assets/groupby/output_27_1.png)
    


Because of being random data, we can not find any valuable information but we can see that the views has been decreased up to negative values in months like June. Lets verify that.


```python
vdf[vdf.Rate<0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>date</th>
      <th>Sums</th>
      <th>Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Career</td>
      <td>2020-06-30</td>
      <td>3044816.8</td>
      <td>-0.102004</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Career</td>
      <td>2020-12-31</td>
      <td>8259240.0</td>
      <td>-0.183001</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Career</td>
      <td>2021-01-31</td>
      <td>84401.6</td>
      <td>-0.989781</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Funny</td>
      <td>2020-04-30</td>
      <td>2716427.3</td>
      <td>-0.029595</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Funny</td>
      <td>2020-09-30</td>
      <td>5526894.3</td>
      <td>-0.108572</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Funny</td>
      <td>2020-11-30</td>
      <td>5058885.1</td>
      <td>-0.180990</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Motivating</td>
      <td>2020-06-30</td>
      <td>3270395.5</td>
      <td>-0.029415</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Motivating</td>
      <td>2020-09-30</td>
      <td>4252837.6</td>
      <td>-0.032287</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Motivating</td>
      <td>2020-12-31</td>
      <td>6643837.5</td>
      <td>-0.075897</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Motivating</td>
      <td>2021-01-31</td>
      <td>347443.2</td>
      <td>-0.947704</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Musical</td>
      <td>2020-05-31</td>
      <td>2558136.9</td>
      <td>-0.110001</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Musical</td>
      <td>2020-10-31</td>
      <td>5345765.1</td>
      <td>-0.058601</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Musical</td>
      <td>2021-01-31</td>
      <td>621339.5</td>
      <td>-0.933231</td>
    </tr>
    <tr>
      <th>57</th>
      <td>News</td>
      <td>2020-07-31</td>
      <td>3319967.6</td>
      <td>-0.035456</td>
    </tr>
  </tbody>
</table>
</div>



More ways and ideas will be updated soon.
