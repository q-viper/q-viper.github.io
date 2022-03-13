---
title:  "Central Tendency vs Dispersion"
date:   2022-03-27 09:29:17 +0545
categories:
    - Statistics
    - Data Analysis
    - 
tags:
    - pandas
    - statistics
header:
  teaser: assets/statistical_analysis/output_17_1.png
---
Hello everyone, welcome back! In this blog we will again focus into the some of widely used central tendency techniques and then measure of spread in the Statistical Analysis of the EDA part. If you are looking for a brief walk-through of a [Statistical Data Analysis in Data Science please refer to this blog](https://dataqoil.com/2022/02/06/walkthrough-of-statistical-analysis-in-data-science/) of mine.

## Central Tendency
Central tendency in statistics is the property of a data which tries to explain the central portion of the data. There are different types of the central tendency and each have their advantage and disadvantages. Lets explore them below.

We will calculate these values in the the titanic dataset.

### Mean
Mean is the most common measure of a central tendency where we divide the sum of all data points by the number of the data points. This is often called as an average and is denoted by Greek symbol $\mu$.




```python
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
df.head()
```




<div><div id=8c5d51eb-0403-4cfe-b422-2fa34b4b0a74 style="display:none; background-color:#9D6CFF; color:white; width:200px; height:30px; padding-left:5px; border-radius:4px; flex-direction:row; justify-content:space-around; align-items:center;" onmouseover="this.style.backgroundColor='#BA9BF8'" onmouseout="this.style.backgroundColor='#9D6CFF'" onclick="window.commands?.execute('create-mitosheet-from-dataframe-output');">See Full Dataframe in Mito</div> <script> if (window.commands.hasCommand('create-mitosheet-from-dataframe-output')) document.getElementById('8c5d51eb-0403-4cfe-b422-2fa34b4b0a74').style.display = 'flex' </script> <table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Thayer)</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table></div>




```python
df.Age.mean()
```




    29.69911764705882



The mean age of the passenger seems to be 29.6.

### Median
Median in statistics is the point of data which is exactly in the center of the data when it is sorted. Median should cut the data points into two halves. Median is often called as second quartile and is calculated as (where N is number of data point):
* When N is odd,
$$
\text{median} = \left(\frac{N+1}{2} \right)^{th} \text{item}
$$

* When N is even,

$$
\text{median} = \frac{\left[ \left(\frac{N}{2} \right)^{th} \text{item} + \left(\frac{N}{2} +1 \right)^{th} \text{item} \right]}{2}
$$




```python
df.Age.median()
```




    28.0



But the median age of the customer seems to be 28.

### Mode
Mode is the value which is most repeated in the data.


```python
df.Age.mode()
```




    0    24.0
    dtype: float64



Age 24 seems to have repeated most. But is it true?


```python
df.Age.value_counts().sort_values(ascending=False)
```




    24.00    30
    22.00    27
    18.00    26
    28.00    25
    19.00    25
             ..
    0.67      1
    20.50     1
    66.00     1
    24.50     1
    12.00     1
    Name: Age, Length: 88, dtype: int64



Yes, it is. It seems that there are 30 passengers with age 24. Which is the most repeated.

### Quartiles
Besides mean, median and mode, quartiles are also sometimes taken along with median. Quartiles gives the values which lies in the 25, 50 and the 75 percentile of the data. But data has to be sorted first.


```python
df.Age.describe()
```




    count    714.000000
    mean      29.699118
    std       14.526497
    min        0.420000
    25%       20.125000
    50%       28.000000
    75%       38.000000
    max       80.000000
    Name: Age, dtype: float64



Looking over the quartile values above, we can see that the minimum value is 0.42, 25% is 20, median or 50% is 28, mean is 29 and 75% is 38 and max is 80. It seems that our data is slightly spread, as max value is far from the 3rd quartile.

### Box Plot


```python
sns.boxplot(data=df, x="Age")
```




    <AxesSubplot:xlabel='Age'>




    
![png]({{site.url}}/assets/statistical_analysis/output_17_1.png)
    


Box plot gives the values just like pandas describe. It seems that most of the data are within our 3rd quartile.

### Mid Range
This is not considered exactly as the central tendency but still it is useful to see how far is our average from the mid range. Mid Range gives the average of the highest and lowest value on the series. Which is simply an arithmetic mean.


```python
(df.Age.min() + df.Age.max())/2
```




    40.21



It seems that our mid range age is 40.21.



### When to use what?
* If the data is normally distributed, that is the mean is exactly in the center.
* We should use median rather than mean when there is huge outliers in the data. Because the mean gets skewed towards direction of outliers but not the median.


## Dispersion
Dispersion is a measure of the spread in the data. It is often used along with the central tendency and sometimes it depends on the central tendency too.

### Inter Quartile Range (IQR)
It is the difference between Q3 and Q1.


```python
df.Age.quantile(0.75)-df.Age.quantile(0.25)
```




    17.875



It seems that our two quartiles are not much far. Which represents that, majority of the data is around center and this is good in the sense of outliers.

### Range
It is the simplest measure of the dispersion, where we subtract smallest value from the largest value. This gives the idea of how huge is our data difference is.


```python
df.Age.max()-df.Age.min()
```




    79.58



It seems that the maximum difference between value is 79.58 but our IQR was only 17.875. There are certainly outliers in the data.

### Standard Deviation
It is the measure of the distance between each points of data with the mean data. This is most widely used measurement of the dispersion because this often gives the error in the data points.
$$
s = \sqrt{\dfrac{\sum_{i=1}^{n}(x_i - \overline{x})^{2}}{n}}
$$


```python
df.Age.std()
```




    14.526497332334044



### Z-Score 
This answers the question `How many standard deviation away is the data from the mean?` It is calculated as:

$$
z = \frac{x-\mu}{\sigma}
$$

Where,
* x is the estimation of the mean or raw score
* 𝜇 is the population mean
* 𝜎 is the population standard deviation



```python

```

Z Score is often used to compare the mean of two different samples and examine which is better than another. Lets take an example of comparing z score. A scored 172 in x and 50 on 37 on y. While mean and std for x and y are (151, 10) and (25.1, 6.4) resp. Z score for both is, 2.1, 1.86. We can conclude that, A scored relatively better in x than y. Lets do this in our Age by taking 500 random samples from our data and comparing it with our population z score.


```python
pop = df.Age
sample1 = df.Age.sample(500)
sample2 = df.Age.sample(500)
sample1_z = (pop.mean()-sample1.mean())/pop.std()
sample2_z = (pop.mean()-sample2.mean())/pop.std()
sample1_z, sample2_z
```




    (0.030374493169490115, 0.020496044621050025)



Looking over above two sample's z scores, we can say that the mean age of sample two is slightly nearer towards the population mean.

## When to use what?
* If we are about to compare means from two different population then we should be using z score.
* If we are about to look how huge data points are, then simple range should be used.
* For looking into how far are the mid points of two halves are, IQR should be used.
* For measuring how far is each data point is from the mean value, Standard Deviation is used.

## Shifting and Scaling of Mean and Standard Deviation
* Mean shifts by same amount as the data shift(add) amount.
* Mean increases by amount as the data was increased by amount.
* Median has property as same as of mean for shift and scale.
* Standard deviation does not shift but scales by amount.
* IQR has same property as of standard deviation.

Lets verify above properties. First calculate the describe of age and then another describe by multiplying age by 2.


```python
df.Age.describe()
```




    count    714.000000
    mean      29.699118
    std       14.526497
    min        0.420000
    25%       20.125000
    50%       28.000000
    75%       38.000000
    max       80.000000
    Name: Age, dtype: float64




```python
(2*df.Age).describe()
```




    count    714.000000
    mean      59.398235
    std       29.052995
    min        0.840000
    25%       40.250000
    50%       56.000000
    75%       76.000000
    max      160.000000
    Name: Age, dtype: float64



It seems that, mean and standard deviation has been increased by 2 times. But lets see what happens when we shift the data by 5 points.


```python
(2+df.Age).describe()
```




    count    714.000000
    mean      31.699118
    std       14.526497
    min        2.420000
    25%       22.125000
    50%       30.000000
    75%       40.000000
    max       82.000000
    Name: Age, dtype: float64



It seems that our means has been shifted but the standard deviation has not changed. Thus we can say that, the shift and the scaling property of dispersion and central tendency is different.

## References 
* [Z-Score Introduction](https://www.youtube.com/watch?v=5S-Zfa-vOXs)
