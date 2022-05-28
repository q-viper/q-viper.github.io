---
title:  "A General Way to Perform an EDA"
date:   2022-05-22 09:29:17 +0545
categories:
    - EDA
    - Data Science
    
tags:
    - Pandas
    - Python

header:
  teaser: assets/general_eda/output_101_1.png
---
## Introduction
Hello everyone, welcome back to another new blog where we will explore different ideas and concept one could perform while performing an EDA. In simple words, this blog is a simple walk-through of an average EDA process which might include (in top down order):
* **Data Loading**: From various sources (remote, local) and various formats (excel, csv, sql etc.)
* **Data Check**: This is very important task where we check the data types (numerical, categorical, binary etc) of a data. We often focus on number of missing values.
* **Data Transformation**: This includes filling up null values, or removing them from the table. We also do some data type conversions if required.
* **Descriptive Analysis**: This is the heart of any EDA because here, we do lots of statistical tasks like finding mean, median, quartiles, mode, distribution, relationships of fields. We also plot different plots to support the analysis. This is sometimes enough to give insights about the data and if the data is rich and we need to find more insights and make assumptions, we have to do Inferential Analysis.
* **Inferential Analysis**: This task sometimes is taken into the EDA part but most of the time we do inferential analysis along with model development. However, we do perform different tests (e.g Chi- Square Test) to calculate feature importance. Here we often do tests based on hypothesis and samples drawn from the population.

While walking through these major steps, one will try to answer different questions of analysis like how many times some categorical data has appeared, what is the distribution over a date, what is the performance over certain cases and so on.

## Data Loading

### Installing Libraries

```python
!pip install autoviz
!pip install seaborn
!pip install plotly
!pip install cufflinks
!pip install pandas
```

* Autoviz is for auto visualization but it is heavy and power hungry. 
* Seaborn is built on top of the matplotlib and is best for making rich static plots.
* Plotly is for interactive visualization.
* Cufflinks is for connecting pandas and plotly.
* Pandas is for data analysis.

### Importing Libraries
If you do not have these libraries installed, please install them like below:




```python
import autoviz
from autoviz.AutoViz_Class import AutoViz_Class
from pandas_profiling import ProfileReport
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from plotly.offline import init_notebook_mode, iplot
import plotly.figure_factory as ff
import cufflinks
import plotly.io as pio 
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')
pio.renderers.default = "notebook" # should change by looking into pio.renderers

pd.options.display.max_columns = None
%matplotlib inline
```


In above step, we have told cufflinks to make plotly plots available offline. And if we are working locally on Jupyter Notebook we should make sure have `pio.renderers.default="notebook"`. 

### Reading File
To make things easier, I am reading file from local storage which is downloaded from [Kaggle](https://www.kaggle.com/garystafford/environmental-sensor-data-132k).

According to the author, the data is collected by 3 IoT devices under different environmental conditions. These environmental conditions plays major role on the analysis later on.

| device            | environmental conditions                 |
|-------------------|------------------------------------------|
| 00:0f:00:70:91:0a | stable conditions, cooler and more humid |
| 1c:bf:ce:15:ec:4d | highly variable temperature and humidity |
| b8:27:eb:bf:9d:51 | stable conditions, warmer and dryer      |


```python
df=pd.read_csv("iot_telemetry_data.csv")
```


```python

```

### Viewing Shape of Data
How many rows and columns are there?


```python
df.shape
```




    (405184, 9)



There are only 9 columns but lots of rows.

### Viewing Top Data


```python
df.head()
```

  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ts</th>
      <th>device</th>
      <th>co</th>
      <th>humidity</th>
      <th>light</th>
      <th>lpg</th>
      <th>motion</th>
      <th>smoke</th>
      <th>temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.594512e+09</td>
      <td>b8:27:eb:bf:9d:51</td>
      <td>0.004956</td>
      <td>51.000000</td>
      <td>False</td>
      <td>0.007651</td>
      <td>False</td>
      <td>0.020411</td>
      <td>22.700000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.594512e+09</td>
      <td>00:0f:00:70:91:0a</td>
      <td>0.002840</td>
      <td>76.000000</td>
      <td>False</td>
      <td>0.005114</td>
      <td>False</td>
      <td>0.013275</td>
      <td>19.700001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.594512e+09</td>
      <td>b8:27:eb:bf:9d:51</td>
      <td>0.004976</td>
      <td>50.900000</td>
      <td>False</td>
      <td>0.007673</td>
      <td>False</td>
      <td>0.020475</td>
      <td>22.600000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.594512e+09</td>
      <td>1c:bf:ce:15:ec:4d</td>
      <td>0.004403</td>
      <td>76.800003</td>
      <td>True</td>
      <td>0.007023</td>
      <td>False</td>
      <td>0.018628</td>
      <td>27.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.594512e+09</td>
      <td>b8:27:eb:bf:9d:51</td>
      <td>0.004967</td>
      <td>50.900000</td>
      <td>False</td>
      <td>0.007664</td>
      <td>False</td>
      <td>0.020448</td>
      <td>22.600000</td>
    </tr>
  </tbody>
</table>



## Data Check

### Viewing Data Types


```python
df.dtypes
```




    ts          float64
    device       object
    co          float64
    humidity    float64
    light          bool
    lpg         float64
    motion         bool
    smoke       float64
    temp        float64
    dtype: object



It seems that we have float data in most of the columns. According to the Author the definition of the columns is

| column   | description          | units      |
|----------|----------------------|------------|
| ts       | timestamp of event   | epoch      |
| device   | unique device name   | string     |
| co       | carbon monoxide      | ppm (%)    |
| humidity | humidity             | percentage |
| light    | light detected?      | boolean    |
| lpg      | liquid petroleum gas | ppm (%)    |
| motion   | motion detected?     | boolean    |
| smoke    | smoke                | ppm (%)    |
| temp     | temperature          | Fahrenheit |

### Checking Missing Values
This is very crucial as missing values could lead to false assumption and sometimes we have to remove or replace them. Lets check how many of columns have missing values.


```python
total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending = False)
mdf = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
mdf = mdf.reset_index()
mdf
```

    NumExpr defaulting to 8 threads.
    




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ts</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>device</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>co</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>humidity</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>light</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>lpg</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>motion</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>smoke</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>temp</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>



It seems that there is no missing data in our dataset. Which is great. But what about outliers? Because outliers also plays huge role in making data modeling tough task. This task falls under the Descriptive Analysis part.

## Data Transformation
It seems that we do not have missing data so we do not have to do much to do besides converting time stamp to datetime.But we might need to transform our data based on the outliers later.

### Datetime
Lets convert timestamp to date time because we will visualize some sort of time series analysis later on.


```python
from datetime import datetime


df["date"]= df.ts.apply(datetime.fromtimestamp)
```

### Device Name
Lets make our device little bit readable. Create a new column `device_name` and add the mapped value of environment and device id.


```python
d={"00:0f:00:70:91:0a":"cooler,more,humid", 
   "1c:bf:ce:15:ec:4d":"variable temp/humidity",
   "b8:27:eb:bf:9d:51":"stable, warmer, dry"}
df["device_name"] = df.device.apply(lambda x: d[x])
df
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ts</th>
      <th>device</th>
      <th>co</th>
      <th>humidity</th>
      <th>light</th>
      <th>lpg</th>
      <th>motion</th>
      <th>smoke</th>
      <th>temp</th>
      <th>date</th>
      <th>device_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.594512e+09</td>
      <td>b8:27:eb:bf:9d:51</td>
      <td>0.004956</td>
      <td>51.000000</td>
      <td>False</td>
      <td>0.007651</td>
      <td>False</td>
      <td>0.020411</td>
      <td>22.700000</td>
      <td>2020-07-12 05:46:34.385975</td>
      <td>stable, warmer, dry</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.594512e+09</td>
      <td>00:0f:00:70:91:0a</td>
      <td>0.002840</td>
      <td>76.000000</td>
      <td>False</td>
      <td>0.005114</td>
      <td>False</td>
      <td>0.013275</td>
      <td>19.700001</td>
      <td>2020-07-12 05:46:34.735568</td>
      <td>cooler,more,humid</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.594512e+09</td>
      <td>b8:27:eb:bf:9d:51</td>
      <td>0.004976</td>
      <td>50.900000</td>
      <td>False</td>
      <td>0.007673</td>
      <td>False</td>
      <td>0.020475</td>
      <td>22.600000</td>
      <td>2020-07-12 05:46:38.073573</td>
      <td>stable, warmer, dry</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.594512e+09</td>
      <td>1c:bf:ce:15:ec:4d</td>
      <td>0.004403</td>
      <td>76.800003</td>
      <td>True</td>
      <td>0.007023</td>
      <td>False</td>
      <td>0.018628</td>
      <td>27.000000</td>
      <td>2020-07-12 05:46:39.589146</td>
      <td>variable temp/humidity</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.594512e+09</td>
      <td>b8:27:eb:bf:9d:51</td>
      <td>0.004967</td>
      <td>50.900000</td>
      <td>False</td>
      <td>0.007664</td>
      <td>False</td>
      <td>0.020448</td>
      <td>22.600000</td>
      <td>2020-07-12 05:46:41.761235</td>
      <td>stable, warmer, dry</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>405179</th>
      <td>1.595203e+09</td>
      <td>00:0f:00:70:91:0a</td>
      <td>0.003745</td>
      <td>75.300003</td>
      <td>False</td>
      <td>0.006247</td>
      <td>False</td>
      <td>0.016437</td>
      <td>19.200001</td>
      <td>2020-07-20 05:48:33.162015</td>
      <td>cooler,more,humid</td>
    </tr>
    <tr>
      <th>405180</th>
      <td>1.595203e+09</td>
      <td>b8:27:eb:bf:9d:51</td>
      <td>0.005882</td>
      <td>48.500000</td>
      <td>False</td>
      <td>0.008660</td>
      <td>False</td>
      <td>0.023301</td>
      <td>22.200000</td>
      <td>2020-07-20 05:48:33.576561</td>
      <td>stable, warmer, dry</td>
    </tr>
    <tr>
      <th>405181</th>
      <td>1.595203e+09</td>
      <td>1c:bf:ce:15:ec:4d</td>
      <td>0.004540</td>
      <td>75.699997</td>
      <td>True</td>
      <td>0.007181</td>
      <td>False</td>
      <td>0.019076</td>
      <td>26.600000</td>
      <td>2020-07-20 05:48:36.167959</td>
      <td>variable temp/humidity</td>
    </tr>
    <tr>
      <th>405182</th>
      <td>1.595203e+09</td>
      <td>00:0f:00:70:91:0a</td>
      <td>0.003745</td>
      <td>75.300003</td>
      <td>False</td>
      <td>0.006247</td>
      <td>False</td>
      <td>0.016437</td>
      <td>19.200001</td>
      <td>2020-07-20 05:48:36.979522</td>
      <td>cooler,more,humid</td>
    </tr>
    <tr>
      <th>405183</th>
      <td>1.595203e+09</td>
      <td>b8:27:eb:bf:9d:51</td>
      <td>0.005914</td>
      <td>48.400000</td>
      <td>False</td>
      <td>0.008695</td>
      <td>False</td>
      <td>0.023400</td>
      <td>22.200000</td>
      <td>2020-07-20 05:48:37.264313</td>
      <td>stable, warmer, dry</td>
    </tr>
  </tbody>
</table>



## Descriptive Analysis
Descriptive Statistics is all about describing the data in the terms of some numbers, charts, graphs or plots. In descriptive statistics, our focus will be on the summary of the data like mean, spread, quartiles, percentiles and so on.

Lets get little bit deep into the descriptive analysis here, we will measure:
* Central tendency which focuses on the average.
* Variability (measure of dispersion) which focuses on how far the data has spreaded.
* Distribution (Frequency distribution) which focuses of number of times something occured.

### Frequency Distribution


#### What is number of observations for each device?


```python
df.groupby("device_name").ts.count().rename("Counts").reset_index().iplot(kind="pie", labels="device_name", values="Counts")
```
![]({{site.url}}/assets/general_eda/pie1.png)     


##### Insights
There seems to be high number of records from the device which was kept on stable, warmer and dry place.

#### What is the distribution of a field over a time?
This question can be done on the time series analysis but we are not focusing on that in this blog.

We have a date time column prepared already from a timestamp. Lets use that column here.


```python
cols = [i for i in df.columns if i not in ["date", "ts", "device", "device_name"]]
for c in cols:
    plt.figure(figsize=(15, 10))
    sns.scatterplot(data=df, x="date", y=c, hue="device_name")
    plt.title(label=f"Distribution of {c} over a time for each Device")
    plt.show()
```


    
![png]({{site.url}}/asstes/general_eda/output_31_0.png)
    



    
![png]({{site.url}}/asstes/general_eda/output_31_1.png)
    



    
![png]({{site.url}}/asstes/general_eda/output_31_2.png)
    



    
![png]({{site.url}}/asstes/general_eda/output_31_3.png)
    



    
![png]({{site.url}}/asstes/general_eda/output_31_4.png)
    



    
![png]({{site.url}}/asstes/general_eda/output_31_5.png)
    



    
![png]({{site.url}}/asstes/general_eda/output_31_6.png)
    


##### Insights
* There seems to be high spikes in CO recorded by cooler,more humid place's device.
* Humidity seems to be normal for all 3 devices but there is not normal flow for device of cooler, more humid place.
* And LPG seems to decreasing for cooler, more humid and increasing for stable, warmer dry place's device.
* And so on.

#### What is the distribution of each Columns?



```python
df.co.iplot(kind="hist", xTitle="ppm in %", yTitle="Frequency", title="Frequency Distribution of CO")
```

![]({{site.url}}/assets/general_eda/hist1.png)     


```python
# df.co.plot(kind="hist", title="Frequency Distribution of CO")
plt.figure(figsize=(8,5))
sns.distplot(df.co, kde=False, color='red', bins=100)
plt.title('Frequency Distribution of CO As a Whole', fontsize=18)
plt.xlabel('Units in ppm (%)', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
```




    Text(0, 0.5, 'Frequency')




    
![png]({{site.url}}/asstes/general_eda/output_35_1.png)
    



```python
plt.figure(figsize=(18,10))
sns.histplot(data=df, x="co", hue="device_name")
plt.title('Frequency Distribution of CO with Device', fontsize=18)
plt.xlabel('Units in ppm (%)', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
```




    Text(0, 0.5, 'Frequency')




    
![png]({{site.url}}/asstes/general_eda/output_36_1.png)
    


##### CO Insights\
* It seems that there is huge number of CO readings for ppm 0.004 to 0.006.
* There is some readings of 0.012 too which might be a outliers in our case and we will later visualize it based on the device.
* The device starting with b8 seems to have read much CO. This device was placed on stable conditions, dry places.

##### All
Lets try to visualize histogram of each fields based on device name.


```python
for c in [i for i in df.columns if i not in ["date", "ts", "device", "device_name"]]:
    plt.figure(figsize=(18,10))
    sns.histplot(data=df, x=c, hue="device_name")
    plt.title(f'Frequency Distribution of {c}', fontsize=18)
    plt.xlabel(f'Values of {c}', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.show()
```


    
![png]({{site.url}}/asstes/general_eda/output_39_0.png)
    



    
![png]({{site.url}}/asstes/general_eda/output_39_1.png)
    



    
![png]({{site.url}}/asstes/general_eda/output_39_2.png)
    



    
![png]({{site.url}}/asstes/general_eda/output_39_3.png)
    



    
![png]({{site.url}}/asstes/general_eda/output_39_4.png)
    



    
![png]({{site.url}}/asstes/general_eda/output_39_5.png)
    



    
![png]({{site.url}}/asstes/general_eda/output_39_6.png)
    


##### All Insights
* Temp seems to be distributed largely for device which was in variable temp.
* Smoke seems to be distributed largely for device which was in cooler temp.
* LPG seems to be distributed largely for device which was in cooler temp.


```python

```

### Central Tendency
Lets view the summary of each numerical data first.

#### Overall Insights


```python
df[[i for i in df.columns if i not in ["date", "ts", "device", "device_name"]]].describe()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>co</th>
      <th>humidity</th>
      <th>lpg</th>
      <th>smoke</th>
      <th>temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>405184.000000</td>
      <td>405184.000000</td>
      <td>405184.000000</td>
      <td>405184.000000</td>
      <td>405184.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.004639</td>
      <td>60.511694</td>
      <td>0.007237</td>
      <td>0.019264</td>
      <td>22.453987</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.001250</td>
      <td>11.366489</td>
      <td>0.001444</td>
      <td>0.004086</td>
      <td>2.698347</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.001171</td>
      <td>1.100000</td>
      <td>0.002693</td>
      <td>0.006692</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.003919</td>
      <td>51.000000</td>
      <td>0.006456</td>
      <td>0.017024</td>
      <td>19.900000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.004812</td>
      <td>54.900000</td>
      <td>0.007489</td>
      <td>0.019950</td>
      <td>22.200000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.005409</td>
      <td>74.300003</td>
      <td>0.008150</td>
      <td>0.021838</td>
      <td>23.600000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.014420</td>
      <td>99.900002</td>
      <td>0.016567</td>
      <td>0.046590</td>
      <td>30.600000</td>
    </tr>
  </tbody>
</table>




```python
cols = [i for i in df.columns if i not in ["date", "ts", "device", "device_name", "motion", "light"]]

df[cols].iplot(kind="box", subplots=True)
```

![]({{site.url}}/assets/general_eda/box1.png)     


* By hovering over each subplots, we could get the min, max, mean, median values. 
* Looking over how the horizontal lines are placed, we could make assumptions like how much is the data skewed.
* It seems there there is high deviation in temperature and humidity which means there could be outliers.

#### Insights Based on Device
Since our overall data might be biased, we have to look into insights based on device. But why biased? The reasons are:
* Each device was on distinct environment
* Each device have different numbers of recordings 


```python
df[[i for i in df.columns if i not in ["date", "ts", "device"]]].groupby("device_name").describe()
```




<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">co</th>
      <th colspan="8" halign="left">humidity</th>
      <th colspan="8" halign="left">lpg</th>
      <th colspan="8" halign="left">smoke</th>
      <th colspan="8" halign="left">temp</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>device_name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cooler,more,humid</th>
      <td>111815.0</td>
      <td>0.003527</td>
      <td>0.001479</td>
      <td>0.001171</td>
      <td>0.002613</td>
      <td>0.003230</td>
      <td>0.004116</td>
      <td>0.014420</td>
      <td>111815.0</td>
      <td>75.444361</td>
      <td>1.975801</td>
      <td>1.1</td>
      <td>74.400002</td>
      <td>75.400002</td>
      <td>76.500000</td>
      <td>99.900002</td>
      <td>111815.0</td>
      <td>0.005893</td>
      <td>0.001700</td>
      <td>0.002693</td>
      <td>0.004815</td>
      <td>0.005613</td>
      <td>0.006689</td>
      <td>0.016567</td>
      <td>111815.0</td>
      <td>0.015489</td>
      <td>0.004809</td>
      <td>0.006692</td>
      <td>0.012445</td>
      <td>0.014662</td>
      <td>0.017682</td>
      <td>0.046590</td>
      <td>111815.0</td>
      <td>19.362552</td>
      <td>0.643786</td>
      <td>0.0</td>
      <td>19.100000</td>
      <td>19.4</td>
      <td>19.700001</td>
      <td>20.200001</td>
    </tr>
    <tr>
      <th>stable, warmer, dry</th>
      <td>187451.0</td>
      <td>0.005560</td>
      <td>0.000559</td>
      <td>0.004646</td>
      <td>0.005079</td>
      <td>0.005439</td>
      <td>0.005993</td>
      <td>0.007955</td>
      <td>187451.0</td>
      <td>50.814077</td>
      <td>1.888926</td>
      <td>45.1</td>
      <td>49.600000</td>
      <td>50.900000</td>
      <td>52.100000</td>
      <td>63.300000</td>
      <td>187451.0</td>
      <td>0.008306</td>
      <td>0.000599</td>
      <td>0.007301</td>
      <td>0.007788</td>
      <td>0.008183</td>
      <td>0.008778</td>
      <td>0.010774</td>
      <td>187451.0</td>
      <td>0.022288</td>
      <td>0.001720</td>
      <td>0.019416</td>
      <td>0.020803</td>
      <td>0.021931</td>
      <td>0.023640</td>
      <td>0.029422</td>
      <td>187451.0</td>
      <td>22.279969</td>
      <td>0.481902</td>
      <td>21.0</td>
      <td>21.900000</td>
      <td>22.3</td>
      <td>22.600000</td>
      <td>24.100000</td>
    </tr>
    <tr>
      <th>variable temp/humidity</th>
      <td>105918.0</td>
      <td>0.004183</td>
      <td>0.000320</td>
      <td>0.003391</td>
      <td>0.003931</td>
      <td>0.004089</td>
      <td>0.004391</td>
      <td>0.006224</td>
      <td>105918.0</td>
      <td>61.910247</td>
      <td>8.944792</td>
      <td>1.6</td>
      <td>55.599998</td>
      <td>59.599998</td>
      <td>65.300003</td>
      <td>92.000000</td>
      <td>105918.0</td>
      <td>0.006764</td>
      <td>0.000373</td>
      <td>0.005814</td>
      <td>0.006470</td>
      <td>0.006657</td>
      <td>0.007009</td>
      <td>0.009022</td>
      <td>105918.0</td>
      <td>0.017895</td>
      <td>0.001055</td>
      <td>0.015224</td>
      <td>0.017064</td>
      <td>0.017592</td>
      <td>0.018589</td>
      <td>0.024341</td>
      <td>105918.0</td>
      <td>26.025511</td>
      <td>2.026427</td>
      <td>0.0</td>
      <td>24.299999</td>
      <td>25.9</td>
      <td>27.299999</td>
      <td>30.600000</td>
    </tr>
  </tbody>
</table>



It is hard to get any insights from above table. Lets view it by looping.


```python
for d in df.device_name.unique():
    df.query(f"device_name=='{d}'")[cols].iplot(kind="box", subplots=True, title=f"Box Plot of device placed at {d}")
```

![]({{site.url}}/assets/general_eda/box2.png) 

![]({{site.url}}/assets/general_eda/box3.png) 

![]({{site.url}}/assets/general_eda/box4.png) 

If we observe plots clearly, there can be seen significant difference in each for the column `temp`.
* CO recorded seems to be higher for a device placed at variable temp. But much spread is of stable, warmer and dry.
* Humidity recorded seems to be lower for stable, warmer dry place's device.
* LPG recorded seems to be well spread on stable, warmer place's device.
* Smoke recorded seems to be spread for stable warmer dry place's device.
* Temperature is self explained that it is lower for cooler place. And so on.

#### Finding Outliers


```python
for c in cols:
    plt.figure(figsize=(15,8))
    sns.boxplot(x="device_name", y=c, data=df)
    plt.title(label=f"Box Plot of {c}")
    plt.show()
```


    
![png]({{site.url}}/asstes/general_eda/output_53_0.png)
    



    
![png]({{site.url}}/asstes/general_eda/output_53_1.png)
    



    
![png]({{site.url}}/asstes/general_eda/output_53_2.png)
    



    
![png]({{site.url}}/asstes/general_eda/output_53_3.png)
    



    
![png]({{site.url}}/asstes/general_eda/output_53_4.png)
    


Outliers are those for which points are away from the horizontal bars.
* There seems to be high outliers in **co** for device which was in **cooler, more humid**.
* There seems to be high outliers in **humidity** for device which was in **cooler, more humid** and **variable temp/humidity**.
* There seems to be high outliers in **LPG** for device which was in **cooler, more humid**.
* There seems to be high outliers in **smoke** for device which was in **cooler, more humid**.
* There seems to be high outliers in **temp** for device which was in c**variable temp/humidity**.

### Correlations
Lets find Pearson's correlation, whose range lies from -1 to 1. Value of -1 means negatively correlated where as +1 means highly correlated.

#### Overall


```python
df.corr().iplot(kind="heatmap")
```

![]({{site.url}}/assets/general_eda/cor1.png) 


```python
df.corr()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ts</th>
      <th>co</th>
      <th>humidity</th>
      <th>light</th>
      <th>lpg</th>
      <th>motion</th>
      <th>smoke</th>
      <th>temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ts</th>
      <td>1.000000</td>
      <td>0.025757</td>
      <td>0.017752</td>
      <td>-0.020868</td>
      <td>0.014178</td>
      <td>-0.006911</td>
      <td>0.016349</td>
      <td>0.074443</td>
    </tr>
    <tr>
      <th>co</th>
      <td>0.025757</td>
      <td>1.000000</td>
      <td>-0.656750</td>
      <td>-0.230197</td>
      <td>0.997331</td>
      <td>-0.000706</td>
      <td>0.998192</td>
      <td>0.110905</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>0.017752</td>
      <td>-0.656750</td>
      <td>1.000000</td>
      <td>0.079703</td>
      <td>-0.672113</td>
      <td>-0.009826</td>
      <td>-0.669863</td>
      <td>-0.410427</td>
    </tr>
    <tr>
      <th>light</th>
      <td>-0.020868</td>
      <td>-0.230197</td>
      <td>0.079703</td>
      <td>1.000000</td>
      <td>-0.208926</td>
      <td>0.033594</td>
      <td>-0.212969</td>
      <td>0.747485</td>
    </tr>
    <tr>
      <th>lpg</th>
      <td>0.014178</td>
      <td>0.997331</td>
      <td>-0.672113</td>
      <td>-0.208926</td>
      <td>1.000000</td>
      <td>0.000232</td>
      <td>0.999916</td>
      <td>0.136396</td>
    </tr>
    <tr>
      <th>motion</th>
      <td>-0.006911</td>
      <td>-0.000706</td>
      <td>-0.009826</td>
      <td>0.033594</td>
      <td>0.000232</td>
      <td>1.000000</td>
      <td>0.000062</td>
      <td>0.037649</td>
    </tr>
    <tr>
      <th>smoke</th>
      <td>0.016349</td>
      <td>0.998192</td>
      <td>-0.669863</td>
      <td>-0.212969</td>
      <td>0.999916</td>
      <td>0.000062</td>
      <td>1.000000</td>
      <td>0.131891</td>
    </tr>
    <tr>
      <th>temp</th>
      <td>0.074443</td>
      <td>0.110905</td>
      <td>-0.410427</td>
      <td>0.747485</td>
      <td>0.136396</td>
      <td>0.037649</td>
      <td>0.131891</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>




* CO have high positive correlation with LPG, Smoke. Negative with Light and Humidity.
* Humidity have negative correlation with smoke, temp, LPG, CO which means that as Humidity increases these fields decreases.
* Light have high correlation with temp. 
* Smoke have high correlation with LPG, CO but negative with humidity.
* And so on.

#### For Each Device
Again, our data is biased and we have to further analyze it for distinct device.


```python
for d in df.device_name.unique():
    corr=df.query(f"device_name=='{d}'").corr()
    print(corr)
    corr.iplot(kind="heatmap", title=f"Correlation of fields for device at {d}")
```

                    ts        co  humidity     light       lpg    motion  \
    ts        1.000000  0.696208  0.042347 -0.112667  0.703218 -0.009851   
    co        0.696208  1.000000 -0.077022 -0.095929  0.999845 -0.003513   
    humidity  0.042347 -0.077022  1.000000 -0.042066 -0.079296 -0.007169   
    light    -0.112667 -0.095929 -0.042066  1.000000 -0.096124  0.007202   
    lpg       0.703218  0.999845 -0.079296 -0.096124  1.000000 -0.003606   
    motion   -0.009851 -0.003513 -0.007169  0.007202 -0.003606  1.000000   
    smoke     0.701994  0.999895 -0.078891 -0.096093  0.999995 -0.003590   
    temp      0.149731 -0.035695 -0.372977  0.008124 -0.033369 -0.000086   
    
                 smoke      temp  
    ts        0.701994  0.149731  
    co        0.999895 -0.035695  
    humidity -0.078891 -0.372977  
    light    -0.096093  0.008124  
    lpg       0.999995 -0.033369  
    motion   -0.003590 -0.000086  
    smoke     1.000000 -0.033786  
    temp     -0.033786  1.000000  
    
      
![]({{site.url}}/assets/general_eda/cor2.png) 

                    ts        co  humidity     light       lpg    motion  \
    ts        1.000000 -0.322829  0.298280 -0.034300 -0.331622  0.004054   
    co       -0.322829  1.000000 -0.221073 -0.048450  0.994789 -0.005022   
    humidity  0.298280 -0.221073  1.000000 -0.169963 -0.227099  0.022255   
    light    -0.034300 -0.048450 -0.169963  1.000000 -0.047746  0.018596   
    lpg      -0.331622  0.994789 -0.227099 -0.047746  1.000000 -0.005482   
    motion    0.004054 -0.005022  0.022255  0.018596 -0.005482  1.000000   
    smoke    -0.330315  0.996474 -0.226195 -0.047971  0.999835 -0.005404   
    temp      0.043851 -0.296603  0.293223 -0.053637 -0.301287  0.001910   
    
                 smoke      temp  
    ts       -0.330315  0.043851  
    co        0.996474 -0.296603  
    humidity -0.226195  0.293223  
    light    -0.047971 -0.053637  
    lpg       0.999835 -0.301287  
    motion   -0.005404  0.001910  
    smoke     1.000000 -0.300719  
    temp     -0.300719  1.000000  
    

![]({{site.url}}/assets/general_eda/cor3.png) 

                    ts        co  humidity  light       lpg    motion     smoke  \
    ts        1.000000 -0.165952 -0.012370    NaN -0.167243 -0.007758 -0.167018   
    co       -0.165952  1.000000 -0.313322    NaN  0.999907  0.013455  0.999937   
    humidity -0.012370 -0.313322  1.000000    NaN -0.314211 -0.011879 -0.314058   
    light          NaN       NaN       NaN    NaN       NaN       NaN       NaN   
    lpg      -0.167243  0.999907 -0.314211    NaN  1.000000  0.013532  0.999997   
    motion   -0.007758  0.013455 -0.011879    NaN  0.013532  1.000000  0.013518   
    smoke    -0.167018  0.999937 -0.314058    NaN  0.999997  0.013518  1.000000   
    temp      0.320340  0.044866 -0.397001    NaN  0.044504  0.021263  0.044566   
    
                  temp  
    ts        0.320340  
    co        0.044866  
    humidity -0.397001  
    light          NaN  
    lpg       0.044504  
    motion    0.021263  
    smoke     0.044566  
    temp      1.000000  
    
![]({{site.url}}/assets/general_eda/cor4.png) 


* One valuable insight can be found in first plot where there is high correlation between time and smoke, then co for device at stable and warmer place. 
* But there is negative correlation between time and smoke for other two devices.

### Conclusion from Descriptive Analysis

As we could see on the above plots and correlation plots, values, we could say that we can not make any judgment based on the overall data because the relationship between fields is different for different place. This could be found in real world that we often have to sub divide the data and perform distinct tests, operations for each. Now we will move on to the next part of our analysis which is Inferential Analysis.

## Inferential Data Analysis

In Inferential Statistics, we take a step forward from the descriptive information we had and try to make some inferences or predictions. In general case, we try to prove, estimate and hypothesize something by taking a sample from the population. Mainly in inferential statistics, our focus will be on making conclusion about something. 

From our descriptive analysis, we knew that there is difference in correlation values of fields for each device and lets focus our test, hypothesis based on that. There are lots of thing we could inference and test here and I think sky is the limit. Also, looking over the time series analysis, there was distinct grouping of each field for distinct devices.

In all of the inferential analysis there there are mainly two things we do:
* Making inferences or predictions about the population. Example,the average age of the passengers is 29 years.
* Making and testing hypothesis about the populations. Example, whether the survival rate of one gender differs from another’s.

### Sampling
Sampling is a concept of taking a small part of a population data with (or without) a hope of having a central tendency of population. Sampling is done when size of the population is high. 

Sampling is very popular in risk analyzing. For example, if a bulb company manufactures bulbs then in order to find the durability, they often take small sample and test on it. Similarly, in data collection types like questionnaire, we often make assumptions based on small number of data and try to claim something about a population. If we want to find out what is the ratio of smokers in male/female gender then we will collect small data and perform some tests to claim some conclusion and apply that in the population.

While working with a sample two terms are used to represent sample and population metrics:
* **Statistics**: It is a measure or metric of sample. e.g. sample average CO.
* **Parameter**: It is a measure or metric of a population. e.g. population average CO.

#### Problems with Sampling
* Sample simply means to draw out the subset of the data from the population and whose size should always be smaller than that of the population. One major problem could be found in sampling is that the mean and variance of sample might not ressemble the population. It is often called as sample error.

In Pandas, we could take sample easily. So lets take a sample of size 10k from the population of size 405184.


```python
sample1=df.sample(n=10000)
sd = sample1[cols].describe()
sd
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>co</th>
      <th>humidity</th>
      <th>lpg</th>
      <th>smoke</th>
      <th>temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.004658</td>
      <td>60.409360</td>
      <td>0.007259</td>
      <td>0.019327</td>
      <td>22.423650</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.001250</td>
      <td>11.344686</td>
      <td>0.001443</td>
      <td>0.004084</td>
      <td>2.650209</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.001171</td>
      <td>4.200000</td>
      <td>0.002693</td>
      <td>0.006692</td>
      <td>5.900000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.003924</td>
      <td>51.000000</td>
      <td>0.006462</td>
      <td>0.017042</td>
      <td>19.900000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.004828</td>
      <td>54.600000</td>
      <td>0.007508</td>
      <td>0.020004</td>
      <td>22.200000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.005460</td>
      <td>74.300003</td>
      <td>0.008206</td>
      <td>0.021998</td>
      <td>23.400000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.013180</td>
      <td>91.599998</td>
      <td>0.015524</td>
      <td>0.043461</td>
      <td>30.600000</td>
    </tr>
  </tbody>
</table>




```python
pod = df[cols].describe()
pod
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>co</th>
      <th>humidity</th>
      <th>lpg</th>
      <th>smoke</th>
      <th>temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>405184.000000</td>
      <td>405184.000000</td>
      <td>405184.000000</td>
      <td>405184.000000</td>
      <td>405184.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.004639</td>
      <td>60.511694</td>
      <td>0.007237</td>
      <td>0.019264</td>
      <td>22.453987</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.001250</td>
      <td>11.366489</td>
      <td>0.001444</td>
      <td>0.004086</td>
      <td>2.698347</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.001171</td>
      <td>1.100000</td>
      <td>0.002693</td>
      <td>0.006692</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.003919</td>
      <td>51.000000</td>
      <td>0.006456</td>
      <td>0.017024</td>
      <td>19.900000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.004812</td>
      <td>54.900000</td>
      <td>0.007489</td>
      <td>0.019950</td>
      <td>22.200000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.005409</td>
      <td>74.300003</td>
      <td>0.008150</td>
      <td>0.021838</td>
      <td>23.600000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.014420</td>
      <td>99.900002</td>
      <td>0.016567</td>
      <td>0.046590</td>
      <td>30.600000</td>
    </tr>
  </tbody>
</table>



The result will come different each time for the sample because it will have random samples each time. But lets find the difference of sample statistics from population parameters.


```python
pod-sd
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>co</th>
      <th>humidity</th>
      <th>lpg</th>
      <th>smoke</th>
      <th>temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.951840e+05</td>
      <td>395184.000000</td>
      <td>3.951840e+05</td>
      <td>395184.000000</td>
      <td>395184.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-1.939797e-05</td>
      <td>0.102334</td>
      <td>-2.223391e-05</td>
      <td>-0.000063</td>
      <td>0.030337</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.081408e-08</td>
      <td>0.021803</td>
      <td>7.320333e-07</td>
      <td>0.000002</td>
      <td>0.048138</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>-3.100000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>-5.900000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-5.239762e-06</td>
      <td>0.000000</td>
      <td>-6.243145e-06</td>
      <td>-0.000018</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-1.696835e-05</td>
      <td>0.300000</td>
      <td>-1.909604e-05</td>
      <td>-0.000054</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-5.135261e-05</td>
      <td>0.000000</td>
      <td>-5.590530e-05</td>
      <td>-0.000160</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.240580e-03</td>
      <td>8.300003</td>
      <td>1.043818e-03</td>
      <td>0.003129</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>



In above table we can see that some value is higher for population while some is for sample.

### Estimation
While working with prediction/hypothesis in inferential analysis, we often have to deal with two types of estimates:
* **Point Estimation**: It is simply a single value estimation for example the sample mean CO is equal to the population mean CO.
* **Interval Estimation**: This estimation is based on finding a value in some range. For example the confidence interval is used in tests like Chi Square, t-test etc. In above example we have seen that there is difference in the trend of field value for each device. But is it significantly different that we should consider each as distinct?


In above example, we could do point estimation like the Temp mean of sample will be equal to population. Example of interval can be, the population mean of Temp will be around 5% left/right of sample.

### Test
Once we are done taking samples and made some estimations, our next step is to test whether we will be able to claim such. So we will test our assumption. This step is known as test.

There are lots of test based upon the nature of estimation, calculation and prediction but all of those can be divided into 3 categories:
* Comparison Test
* Correlation Test
* Regression Test

Based on parameters, we can also categorize tests into two groups:
* **Parametric Test**: Parametric tests are those in which we work with parameters like mean and variance. One example of this test is t-test.
* **Non Parametric Test**: These tests are non parametric because does not use parameters in the hypothesis. One example is Mann Whitney U test.

Based on the measurement (Nominal, Ordinal, Interval and Ratio) of the data we can choose best test for our data. 


### Terms Widely Used in Testing
* **Confidence Interval**: Confidence interval is all about giving some room for the error. Which is often used with tests. For example, if we are trying to make a test where we have set our hypothesis that the average CO recorded by device first is not more than 2% of device second. Here we are giving some room for possible error.
* **Confidence Level**: It sounds similar to confidence interval but no it is not. But these two terms are related to each other. Confidence level tells us how much probability is there that the sample statistics or estimated parameter lies within the confidence interval. For example, if we set the confidence level to 5%, then we will be claiming that if there are 100 tests done, at max 5 will be predicting wrong prediction. Or in other words, out of 100 tests, 95 tests will have the estimated value lie within the confidence interval.
* **Hypothesis**: As the term suggests, hypothesis is something that we are assuming to happen. In Hypothesis testing, we will have different hypothesis against the default or null hypothesis. Those hypothesis against the default are known as alternative hypothesis.

### Comparison Test
This kind of test is mostly done where we will compare the parameters, metrics between different samples or population vs sample. Generally we perform parametric tests here.

Test|Parametric|Comparison With|No. Samples|
----|----|----|-----|
t-test|Yes|Mean, Variance|2|
ANOVA|Yes|Variance, Mean|3+|
Mann-Whitney U (Wilcoxon Rank Sum)|No|Sum of rankings|2|
Wilcoxon Signed Rank|	No|	Distributions|2|
Kruskal-Wallis H|	No|	Mean Rankings|	3+|
Mood’s Median|	No|	Medians|	2+|

#### Is the mean value of each fields same for each device's recorded data?
ANOVA means Analysis of Variance. This test is used when we have to compare statistics between two or more samples. If we have two sample, we will use t-test.

Lets test it by assuming 5% of alpha value which is significance level. We assume that if there will be 5 wrong prediction out of 100, then we will ignore it.

* Null Hypothesis: There is no difference in mean values of each devices.
* Alternate Hypothesis: There is significant difference in mean value for each devices.




```python
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# formula = 'len ~ C(supp) + C(dose) + C(supp):C(dose)'
for c in cols:
    formula = f'{c} ~ device_name'
    model = ols(formula, data=df).fit()
    aov_table = anova_lm(model,typ=2)
    print(aov_table)
```

                   sum_sq        df              F  PR(>F)
    device_name  0.319266       2.0  206081.057177     0.0
    Residual     0.313859  405181.0            NaN     NaN
                       sum_sq        df              F  PR(>F)
    device_name  4.276879e+07       2.0  904472.329682     0.0
    Residual     9.579674e+06  405181.0            NaN     NaN
                   sum_sq        df              F  PR(>F)
    device_name  0.439852       2.0  219945.965812     0.0
    Residual     0.405145  405181.0            NaN     NaN
                   sum_sq        df              F  PR(>F)
    device_name  3.506427       2.0  217991.815523     0.0
    Residual     3.258695  405181.0            NaN     NaN
                       sum_sq        df              F  PR(>F)
    device_name  2.425356e+06       2.0  936247.353097     0.0
    Residual     5.248123e+05  405181.0            NaN     NaN
    

It seems that that the p value is smaller than 5%, thus we reject the null hypothesis and claim that there is significant difference in mean values of fields of each device. But lets use ANOVA from SciPy's stats and result must be same.


```python
import scipy.stats as stats


for c in cols:
    devs = df.device_name.unique()
    groups = df.groupby("device_name").groups

    co0 = df[c][groups[devs[0]]]
    co1 = df[c][groups[devs[1]]]
    co2 = df[c][groups[devs[2]]]

    print(stats.f_oneway(co0, co1, co2))
```

    F_onewayResult(statistic=206081.05717747274, pvalue=0.0)
    F_onewayResult(statistic=904472.329681998, pvalue=0.0)
    F_onewayResult(statistic=219945.96581178883, pvalue=0.0)
    F_onewayResult(statistic=217991.81552333018, pvalue=0.0)
    F_onewayResult(statistic=936247.3530974094, pvalue=0.0)
    


```python

```

### Correlation Test
Correlation tests are done to calculate the strength of the association between data. 

Test|	Parametric|	Data Type|
---|---|---|
Pearson’s r|	Yes|	Interval/Ratio
Spearman’s r|	No|	Ordinal/Interval/Ratio
Chi Square Test of Independence|No|	Nominal/Ordinal

Pearson's r test is statistically powerful than Spearman's but Spearman's test is appropriate for interval and ratio type of data.

Only Chi Square Test of Independence is the only test that can be used with nominal variables.

#### Pearson's and Spearman's Test
##### Pearson's Test For Linear Relationship Between Variables

The coefficient returns a value between -1 and 1 that represents the limits of correlation from a full negative correlation to a full positive correlation. A value of 0 means no correlation. The value must be interpreted, where often a value below -0.5 or above 0.5 indicates a notable correlation, and values below those values suggests a less notable correlation.

A formula is:

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/0a96c914bb811b84698b4d4118794cf4c8167ca7)
*From Wikipedia*

We have already done this test on the Descriptive Analysis Part.




##### Spearman’s Correlation: Non-Linear Relationship between two variables.
Two variables may be related by a nonlinear relationship, such that the relationship is stronger or weaker across the distribution of the variables. In this case Spearman's correlation is used.

Pearson correlation assumes the data is normally distributed. However, Spearman does not make any assumption on the distribution of the data. That is the main difference between these two.

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/ee94267b983c2f16be1d3c61556e264762d5cba9)
*From Wikipedia*



```python
df.corr("spearman")
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ts</th>
      <th>co</th>
      <th>humidity</th>
      <th>light</th>
      <th>lpg</th>
      <th>motion</th>
      <th>smoke</th>
      <th>temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ts</th>
      <td>1.000000</td>
      <td>0.077576</td>
      <td>0.051555</td>
      <td>-0.020867</td>
      <td>0.077576</td>
      <td>-0.006917</td>
      <td>0.077576</td>
      <td>0.055377</td>
    </tr>
    <tr>
      <th>co</th>
      <td>0.077576</td>
      <td>1.000000</td>
      <td>-0.764622</td>
      <td>-0.337479</td>
      <td>1.000000</td>
      <td>-0.003210</td>
      <td>1.000000</td>
      <td>0.121469</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>0.051555</td>
      <td>-0.764622</td>
      <td>1.000000</td>
      <td>0.210620</td>
      <td>-0.764622</td>
      <td>-0.006705</td>
      <td>-0.764622</td>
      <td>-0.334038</td>
    </tr>
    <tr>
      <th>light</th>
      <td>-0.020867</td>
      <td>-0.337479</td>
      <td>0.210620</td>
      <td>1.000000</td>
      <td>-0.337479</td>
      <td>0.033594</td>
      <td>-0.337479</td>
      <td>0.713951</td>
    </tr>
    <tr>
      <th>lpg</th>
      <td>0.077576</td>
      <td>1.000000</td>
      <td>-0.764622</td>
      <td>-0.337479</td>
      <td>1.000000</td>
      <td>-0.003210</td>
      <td>1.000000</td>
      <td>0.121469</td>
    </tr>
    <tr>
      <th>motion</th>
      <td>-0.006917</td>
      <td>-0.003210</td>
      <td>-0.006705</td>
      <td>0.033594</td>
      <td>-0.003210</td>
      <td>1.000000</td>
      <td>-0.003210</td>
      <td>0.033095</td>
    </tr>
    <tr>
      <th>smoke</th>
      <td>0.077576</td>
      <td>1.000000</td>
      <td>-0.764622</td>
      <td>-0.337479</td>
      <td>1.000000</td>
      <td>-0.003210</td>
      <td>1.000000</td>
      <td>0.121469</td>
    </tr>
    <tr>
      <th>temp</th>
      <td>0.055377</td>
      <td>0.121469</td>
      <td>-0.334038</td>
      <td>0.713951</td>
      <td>0.121469</td>
      <td>0.033095</td>
      <td>0.121469</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>



#### Insights
Some of notable insights:
* High +ve correlation of co with lpg, smoke.
* High -ve correlation of humidity with co, lpg.
* High +ve correlation of light with temp.
* And so on.

#### Chi Square Test
**[When to use Chi Square?](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3900058/)**

The Chi-square test is a non-parametric statistic, also called a distribution free test. Non-parametric tests should be used when any one of the following conditions pertains to the data:

* The level of measurement of all the variables is nominal or ordinal.
* The sample sizes of the study groups are unequal; for the χ2 the groups may be of equal size or unequal size whereas some parametric tests require groups of equal or approximately equal size.
* The original data were measured at an interval or ratio level, but violate one of the following assumptions of a parametric test:
    * The distribution of the data was seriously skewed or kurtotic (parametric tests assume approximately normal distribution of the dependent variable), and thus the researcher must use a distribution free statistic rather than a parametric statistic.
    * The data violate the assumptions of equal variance or homoscedasticity.
    * For any of a number of reasons (1), the continuous data were collapsed into a small number of categories, and thus the data are no longer interval or ratio.
    
**Note:**

* **Null Hypothesis(H0):** Two variables are not dependent. (no association between the two variables)
* **Alternate Hypothesis(H1):** There is relationship between variables. 


* If Statistic >= Critical Value: significant result, reject null hypothesis (H0), dependent.
* If Statistic < Critical Value: not significant result, fail to reject null hypothesis (H0), independent.

In terms of a p-value and a chosen significance level (alpha), the test can be interpreted as follows:

* If p-value <= alpha: significant result, reject null hypothesis (H0), dependent.
* If p-value > alpha: not significant result, fail to reject null hypothesis (H0), independent.

We do not have nominal data here thus we will not perform any test here yet.

#### Collinearity vs Multicollinearity
Correlation and collinearity are similar things with few differences:
* Correlation measures the relationship strength and direction of the relationship between two fields in our data.
* Collinearity is a situation where two fields are linearly associated (high correlation) and they are used as predictors for the target.
* Multicollinearity is a case if collinearity where a there exists linear relationship with two or more features.

While training ML models, it is important that we remove those features that exhibit multicollinearity and we could do so by calculating VIF (Variance Inflation Factor). VIF allows us to determine the strength of correlation between other variables.
VIF calculates how much the variance of a coefficient is inflated because of its linear dependencies with other predictors. Hence its name.

Referenced from [here](https://towardsdatascience.com/statistics-in-python-collinearity-and-multicollinearity-4cc4dcd82b3f).

![](https://miro.medium.com/max/223/1*kh_lcXAhwdfRarDKkpsSBg.png)

* `(1-R**2)` is known as tolerance factor.

R-squared (R2) is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model

Referenced from [here](https://www.investopedia.com/terms/r/r-squared.asp).

Interpreting VIF:
* 1 — features are not correlated
* 1<VIF<5 — features are moderately correlated
* VIF>5 — features are highly correlated
* VIF>10 — high correlation between features and is cause for concern


```python
from sklearn.linear_model import LinearRegression
def calculate_vif(df, features):    
    vif, tolerance = {}, {}
    # all the features that you want to examine
    for feature in features:
        # extract all the other features you will regress against
        X = [f for f in features if f != feature]        
        X, y = df[X], df[feature]
        # extract r-squared from the fit
        r2 = LinearRegression().fit(X, y).score(X, y)                
        
        # calculate tolerance
        tolerance[feature] = 1 - r2
        # calculate VIF
        vif[feature] = 1/(tolerance[feature])
    # return VIF DataFrame
    return pd.DataFrame({'VIF': vif, 'Tolerance': tolerance})
```

Now calculating VIF of our columns with each other.


```python
calculate_vif(df=df, features=[c for c in df.columns if c not in ["device", "device_name", "date"]])
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF</th>
      <th>Tolerance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ts</th>
      <td>1.124875e+00</td>
      <td>8.889881e-01</td>
    </tr>
    <tr>
      <th>co</th>
      <td>8.709637e+04</td>
      <td>1.148153e-05</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>3.618642e+00</td>
      <td>2.763468e-01</td>
    </tr>
    <tr>
      <th>light</th>
      <td>4.123083e+00</td>
      <td>2.425369e-01</td>
    </tr>
    <tr>
      <th>lpg</th>
      <td>1.872582e+06</td>
      <td>5.340219e-07</td>
    </tr>
    <tr>
      <th>motion</th>
      <td>1.001580e+00</td>
      <td>9.984225e-01</td>
    </tr>
    <tr>
      <th>smoke</th>
      <td>2.765493e+06</td>
      <td>3.615991e-07</td>
    </tr>
    <tr>
      <th>temp</th>
      <td>4.835901e+00</td>
      <td>2.067867e-01</td>
    </tr>
  </tbody>
</table>



In above table, we can see that, 
* LPG, Smoke, CO have high correlation between other features and thus it can be our concerned features.
* Also Temp, Humidity seems to be having good correlation but Time stamp, motion, does not seem to be having good relationships.

Lets remove `co` as highly correlated feature and calculating VIF again to see what effect can be seen.


```python
calculate_vif(df=df, features=[c for c in df.columns if c not in ["device", "device_name", "date", "co"]])
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF</th>
      <th>Tolerance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ts</th>
      <td>1.085452</td>
      <td>0.921275</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>3.079545</td>
      <td>0.324723</td>
    </tr>
    <tr>
      <th>light</th>
      <td>4.023971</td>
      <td>0.248511</td>
    </tr>
    <tr>
      <th>lpg</th>
      <td>7206.123295</td>
      <td>0.000139</td>
    </tr>
    <tr>
      <th>motion</th>
      <td>1.001578</td>
      <td>0.998424</td>
    </tr>
    <tr>
      <th>smoke</th>
      <td>7185.519140</td>
      <td>0.000139</td>
    </tr>
    <tr>
      <th>temp</th>
      <td>4.833069</td>
      <td>0.206908</td>
    </tr>
  </tbody>
</table>



The change can be seen in the terms that the VIF of LPG, Smoke has also decreased. It is sure that these 3 fields have high collinearity. Now again removing feature `smoke` and calculating VIF.


```python
calculate_vif(df=df, features=[c for c in df.columns if c not in ["device", "device_name", "date", "smoke"]])
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF</th>
      <th>Tolerance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ts</th>
      <td>1.081658</td>
      <td>0.924507</td>
    </tr>
    <tr>
      <th>co</th>
      <td>226.300545</td>
      <td>0.004419</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>3.096043</td>
      <td>0.322993</td>
    </tr>
    <tr>
      <th>light</th>
      <td>4.017793</td>
      <td>0.248893</td>
    </tr>
    <tr>
      <th>lpg</th>
      <td>231.395623</td>
      <td>0.004322</td>
    </tr>
    <tr>
      <th>motion</th>
      <td>1.001578</td>
      <td>0.998425</td>
    </tr>
    <tr>
      <th>temp</th>
      <td>4.832018</td>
      <td>0.206953</td>
    </tr>
  </tbody>
</table>



The changes seems to be more reflected. And it is clear that smoke have more collinearity than that of co with others. But again checking by removing LPG and calculating VIF.


```python
calculate_vif(df=df, features=[c for c in df.columns if c not in ["device", "device_name", "date", "lpg"]])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF</th>
      <th>Tolerance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ts</th>
      <td>1.080860</td>
      <td>0.925189</td>
    </tr>
    <tr>
      <th>co</th>
      <td>335.166702</td>
      <td>0.002984</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>3.101372</td>
      <td>0.322438</td>
    </tr>
    <tr>
      <th>light</th>
      <td>4.016575</td>
      <td>0.248968</td>
    </tr>
    <tr>
      <th>motion</th>
      <td>1.001578</td>
      <td>0.998425</td>
    </tr>
    <tr>
      <th>smoke</th>
      <td>341.732959</td>
      <td>0.002926</td>
    </tr>
    <tr>
      <th>temp</th>
      <td>4.831788</td>
      <td>0.206963</td>
    </tr>
  </tbody>
</table>


The effects are similar to the case where we removed smoke.

### Regression Tests
Regression tests are done where we try to estimate some parameter. If we have one dependent and one independent variable then we will be using simple linear regression like $y=mx+c$. If we have multiple variables then it will be mulilinear regression. But besides linear, there is logistic regression which tries to classify between two class. 

The regression test examines whether the change is dependent variable have any effect in the independent variable or not.

Test|	Predictor|	Outcome |
----|-----|-----|
Simple Linear|	1 interval/ratio|	1 interval/ratio|
Multi Linear|	2+ interval/ratio	|1 interval/ratio|
Logistic regression	|1+ |	1 binary|
Nominal regression	|1+ |	1 nominal|
Ordinal regression	|1+ |	1 ordinal|

The linear relationship between features has been already discovered like the rise in CO has something to do with LPG and Smoke thus we can skip this test for now.

## Using Autoviz for Fast EDA
Autoviz is a kind of auto EDA tool which performs lots of EDA and plots graphs and provides some valuable insights. However, manual EDA always gives much insights if we have time to perform one. And using Pandas profiler, we can get insights like correlation in terms of sentence.


```python
av = AutoViz_Class()
dfa = av.AutoViz("iot_telemetry_data.csv")

```

        max_rows_analyzed is smaller than dataset shape 405184...
            randomly sampled 150000 rows from read CSV file
    Shape of your Data Set loaded: (150000, 9)
    ############## C L A S S I F Y I N G  V A R I A B L E S  ####################
    Classifying variables in data set...
        9 Predictors classified...
            No variables removed since no ID or low-information variables found in data set
    Since Number of Rows in data 150000 exceeds maximum, randomly sampling 150000 rows for EDA...
    Number of All Scatter Plots = 21
    


    
![png]({{site.url}}/asstes/general_eda/output_101_1.png)
    



    
![png]({{site.url}}/asstes/general_eda/output_101_2.png)
    



    
![png]({{site.url}}/asstes/general_eda/output_101_3.png)
    



    
![png]({{site.url}}/asstes/general_eda/output_101_4.png)
    



    
![png]({{site.url}}/asstes/general_eda/output_101_5.png)
    



    
![png]({{site.url}}/asstes/general_eda/output_101_6.png)
    


    Time to run AutoViz = 35 seconds 
    
     ###################### AUTO VISUALIZATION Completed ########################
    

As we can see in the above outputs, there are lots of plots to find outliers, relationships and so on. Most of them are done by us manually on earlier steps but if we are on hurry and want to grasp insight as soon as possible, Autoviz is highly recommended. Pandas Profiling is even richer and it gives us interactive way to tune between different aspects of EDA like correlation, null counts, plots and so on. But this blog doesn't contain the result because this is a static blog. :)


```python
ProfileReport(dfa)
```