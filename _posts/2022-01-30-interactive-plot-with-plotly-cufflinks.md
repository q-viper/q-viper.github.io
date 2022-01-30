---
title:  "Plotting interactive plots with Plotly and Cufflinks"
date:   2022-01-30 03:29:17 +0545
# last_modified_at: 2022-01-23 12:29:17 +0545
categories:
    - data visualization
tags:
    - data analysis
    - plotly
header:
  teaser: assets/plotly/map2.png
---
# Plotting High Quality Plots in Python with Plotly and Clufflinks

## Interactive Plot
This blog contains static images and is not rendering interactive plots thus I request you to visit [mine this interactive blog.]({{site.url}}/html_posts/interactive_plot_with_plotly).

## Introduction

Hello everyone, in this blog we are going to explore some of most used and simplest plots in the data analysis. If you have made your hand dirty playing with data then you might have come across at least anyone of these plots. And in Python, we have been doing these plots using Matplotlib. But above that, we have some tools like Seaborn (built on the top of Matplotlib) which gave use nice graphs. But those were not interactive plots. Plotly is all about interactivity!

This blog will be updated frequently.
* January 28 2022, started blog writing.

## Installation

This blog was prepared and run on the google colab and if you are trying to run codes in local computer, please install plotly first by `pip install plotly`. You can visit [official link](https://plotly.com/python/getting-started/) if you want. Then cufflinks by `pip install cufflinks`.

```python
import pandas as pd
import numpy as np
import warnings
from plotly.offline import init_notebook_mode, iplot
import plotly.figure_factory as ff
import cufflinks
import plotly.io as pio 
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')
pio.renderers.default = "colab" # should change by looking into pio.renderers

pd.options.display.max_columns = None
# pd.options.display.max_rows = None
```


```python
pio.renderers
```




    Renderers configuration
    -----------------------
        Default renderer: 'colab'
        Available renderers:
            ['plotly_mimetype', 'jupyterlab', 'nteract', 'vscode',
             'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab',
             'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg',
             'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe',
             'iframe_connected', 'sphinx_gallery', 'sphinx_gallery_png']



If you are running Plotly on colab then use `pio.renderers.default = "colab"` else choose according to your need.

## Get Dataset

For the purpose of visualization, we are going to look into COVID 19 Dataset publicly available on [GitHub](https://github.com/owid/covid-19-data/tree/master/public/data). 


> Since the main goal of this blog is to explore visualization not the analysis part, we will be skipping most of analysis and focus only on the plots.


```python
df = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv")
df["date"] = pd.to_datetime(df.date)
df
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>iso_code</th>
      <th>continent</th>
      <th>location</th>
      <th>date</th>
      <th>total_cases</th>
      <th>new_cases</th>
      <th>new_cases_smoothed</th>
      <th>total_deaths</th>
      <th>new_deaths</th>
      <th>new_deaths_smoothed</th>
      <th>total_cases_per_million</th>
      <th>new_cases_per_million</th>
      <th>new_cases_smoothed_per_million</th>
      <th>total_deaths_per_million</th>
      <th>new_deaths_per_million</th>
      <th>new_deaths_smoothed_per_million</th>
      <th>reproduction_rate</th>
      <th>icu_patients</th>
      <th>icu_patients_per_million</th>
      <th>hosp_patients</th>
      <th>hosp_patients_per_million</th>
      <th>weekly_icu_admissions</th>
      <th>weekly_icu_admissions_per_million</th>
      <th>weekly_hosp_admissions</th>
      <th>weekly_hosp_admissions_per_million</th>
      <th>new_tests</th>
      <th>total_tests</th>
      <th>total_tests_per_thousand</th>
      <th>new_tests_per_thousand</th>
      <th>new_tests_smoothed</th>
      <th>new_tests_smoothed_per_thousand</th>
      <th>positive_rate</th>
      <th>tests_per_case</th>
      <th>tests_units</th>
      <th>total_vaccinations</th>
      <th>people_vaccinated</th>
      <th>people_fully_vaccinated</th>
      <th>total_boosters</th>
      <th>new_vaccinations</th>
      <th>new_vaccinations_smoothed</th>
      <th>total_vaccinations_per_hundred</th>
      <th>people_vaccinated_per_hundred</th>
      <th>people_fully_vaccinated_per_hundred</th>
      <th>total_boosters_per_hundred</th>
      <th>new_vaccinations_smoothed_per_million</th>
      <th>new_people_vaccinated_smoothed</th>
      <th>new_people_vaccinated_smoothed_per_hundred</th>
      <th>stringency_index</th>
      <th>population</th>
      <th>population_density</th>
      <th>median_age</th>
      <th>aged_65_older</th>
      <th>aged_70_older</th>
      <th>gdp_per_capita</th>
      <th>extreme_poverty</th>
      <th>cardiovasc_death_rate</th>
      <th>diabetes_prevalence</th>
      <th>female_smokers</th>
      <th>male_smokers</th>
      <th>handwashing_facilities</th>
      <th>hospital_beds_per_thousand</th>
      <th>life_expectancy</th>
      <th>human_development_index</th>
      <th>excess_mortality_cumulative_absolute</th>
      <th>excess_mortality_cumulative</th>
      <th>excess_mortality</th>
      <th>excess_mortality_cumulative_per_million</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AFG</td>
      <td>Asia</td>
      <td>Afghanistan</td>
      <td>2020-02-24</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.126</td>
      <td>0.126</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.33</td>
      <td>39835428.0</td>
      <td>54.422</td>
      <td>18.6</td>
      <td>2.581</td>
      <td>1.337</td>
      <td>1803.987</td>
      <td>NaN</td>
      <td>597.029</td>
      <td>9.59</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>37.746</td>
      <td>0.5</td>
      <td>64.83</td>
      <td>0.511</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AFG</td>
      <td>Asia</td>
      <td>Afghanistan</td>
      <td>2020-02-25</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.126</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.33</td>
      <td>39835428.0</td>
      <td>54.422</td>
      <td>18.6</td>
      <td>2.581</td>
      <td>1.337</td>
      <td>1803.987</td>
      <td>NaN</td>
      <td>597.029</td>
      <td>9.59</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>37.746</td>
      <td>0.5</td>
      <td>64.83</td>
      <td>0.511</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AFG</td>
      <td>Asia</td>
      <td>Afghanistan</td>
      <td>2020-02-26</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.126</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.33</td>
      <td>39835428.0</td>
      <td>54.422</td>
      <td>18.6</td>
      <td>2.581</td>
      <td>1.337</td>
      <td>1803.987</td>
      <td>NaN</td>
      <td>597.029</td>
      <td>9.59</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>37.746</td>
      <td>0.5</td>
      <td>64.83</td>
      <td>0.511</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AFG</td>
      <td>Asia</td>
      <td>Afghanistan</td>
      <td>2020-02-27</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.126</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.33</td>
      <td>39835428.0</td>
      <td>54.422</td>
      <td>18.6</td>
      <td>2.581</td>
      <td>1.337</td>
      <td>1803.987</td>
      <td>NaN</td>
      <td>597.029</td>
      <td>9.59</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>37.746</td>
      <td>0.5</td>
      <td>64.83</td>
      <td>0.511</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AFG</td>
      <td>Asia</td>
      <td>Afghanistan</td>
      <td>2020-02-28</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.126</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.33</td>
      <td>39835428.0</td>
      <td>54.422</td>
      <td>18.6</td>
      <td>2.581</td>
      <td>1.337</td>
      <td>1803.987</td>
      <td>NaN</td>
      <td>597.029</td>
      <td>9.59</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>37.746</td>
      <td>0.5</td>
      <td>64.83</td>
      <td>0.511</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>157471</th>
      <td>ZWE</td>
      <td>Africa</td>
      <td>Zimbabwe</td>
      <td>2022-01-22</td>
      <td>228179.0</td>
      <td>218.0</td>
      <td>363.143</td>
      <td>5292.0</td>
      <td>4.0</td>
      <td>7.714</td>
      <td>15119.031</td>
      <td>14.445</td>
      <td>24.062</td>
      <td>350.645</td>
      <td>0.265</td>
      <td>0.511</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2626.0</td>
      <td>1822879.0</td>
      <td>120.783</td>
      <td>0.174</td>
      <td>4145.0</td>
      <td>0.275</td>
      <td>0.0876</td>
      <td>11.4</td>
      <td>tests performed</td>
      <td>7506786.0</td>
      <td>4239537.0</td>
      <td>3267249.0</td>
      <td>NaN</td>
      <td>9904.0</td>
      <td>10567.0</td>
      <td>49.74</td>
      <td>28.09</td>
      <td>21.65</td>
      <td>NaN</td>
      <td>700.0</td>
      <td>5058.0</td>
      <td>0.034</td>
      <td>NaN</td>
      <td>15092171.0</td>
      <td>42.729</td>
      <td>19.6</td>
      <td>2.822</td>
      <td>1.882</td>
      <td>1899.775</td>
      <td>21.4</td>
      <td>307.846</td>
      <td>1.82</td>
      <td>1.6</td>
      <td>30.7</td>
      <td>36.791</td>
      <td>1.7</td>
      <td>61.49</td>
      <td>0.571</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>157472</th>
      <td>ZWE</td>
      <td>Africa</td>
      <td>Zimbabwe</td>
      <td>2022-01-23</td>
      <td>228254.0</td>
      <td>75.0</td>
      <td>310.857</td>
      <td>5294.0</td>
      <td>2.0</td>
      <td>6.714</td>
      <td>15124.000</td>
      <td>4.969</td>
      <td>20.597</td>
      <td>350.778</td>
      <td>0.133</td>
      <td>0.445</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1541.0</td>
      <td>1824420.0</td>
      <td>120.885</td>
      <td>0.102</td>
      <td>3912.0</td>
      <td>0.259</td>
      <td>0.0795</td>
      <td>12.6</td>
      <td>tests performed</td>
      <td>7512903.0</td>
      <td>4242647.0</td>
      <td>3270256.0</td>
      <td>NaN</td>
      <td>6117.0</td>
      <td>10631.0</td>
      <td>49.78</td>
      <td>28.11</td>
      <td>21.67</td>
      <td>NaN</td>
      <td>704.0</td>
      <td>5182.0</td>
      <td>0.034</td>
      <td>NaN</td>
      <td>15092171.0</td>
      <td>42.729</td>
      <td>19.6</td>
      <td>2.822</td>
      <td>1.882</td>
      <td>1899.775</td>
      <td>21.4</td>
      <td>307.846</td>
      <td>1.82</td>
      <td>1.6</td>
      <td>30.7</td>
      <td>36.791</td>
      <td>1.7</td>
      <td>61.49</td>
      <td>0.571</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>157473</th>
      <td>ZWE</td>
      <td>Africa</td>
      <td>Zimbabwe</td>
      <td>2022-01-24</td>
      <td>228541.0</td>
      <td>287.0</td>
      <td>297.286</td>
      <td>5305.0</td>
      <td>11.0</td>
      <td>6.714</td>
      <td>15143.017</td>
      <td>19.016</td>
      <td>19.698</td>
      <td>351.507</td>
      <td>0.729</td>
      <td>0.445</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4913.0</td>
      <td>1829333.0</td>
      <td>121.211</td>
      <td>0.326</td>
      <td>4043.0</td>
      <td>0.268</td>
      <td>0.0735</td>
      <td>13.6</td>
      <td>tests performed</td>
      <td>7517985.0</td>
      <td>4245063.0</td>
      <td>3272922.0</td>
      <td>NaN</td>
      <td>5082.0</td>
      <td>10273.0</td>
      <td>49.81</td>
      <td>28.13</td>
      <td>21.69</td>
      <td>NaN</td>
      <td>681.0</td>
      <td>5009.0</td>
      <td>0.033</td>
      <td>NaN</td>
      <td>15092171.0</td>
      <td>42.729</td>
      <td>19.6</td>
      <td>2.822</td>
      <td>1.882</td>
      <td>1899.775</td>
      <td>21.4</td>
      <td>307.846</td>
      <td>1.82</td>
      <td>1.6</td>
      <td>30.7</td>
      <td>36.791</td>
      <td>1.7</td>
      <td>61.49</td>
      <td>0.571</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>157474</th>
      <td>ZWE</td>
      <td>Africa</td>
      <td>Zimbabwe</td>
      <td>2022-01-25</td>
      <td>228776.0</td>
      <td>235.0</td>
      <td>330.857</td>
      <td>5316.0</td>
      <td>11.0</td>
      <td>8.286</td>
      <td>15158.588</td>
      <td>15.571</td>
      <td>21.922</td>
      <td>352.236</td>
      <td>0.729</td>
      <td>0.549</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7525574.0</td>
      <td>4248576.0</td>
      <td>3276998.0</td>
      <td>NaN</td>
      <td>7589.0</td>
      <td>9579.0</td>
      <td>49.86</td>
      <td>28.15</td>
      <td>21.71</td>
      <td>NaN</td>
      <td>635.0</td>
      <td>4638.0</td>
      <td>0.031</td>
      <td>NaN</td>
      <td>15092171.0</td>
      <td>42.729</td>
      <td>19.6</td>
      <td>2.822</td>
      <td>1.882</td>
      <td>1899.775</td>
      <td>21.4</td>
      <td>307.846</td>
      <td>1.82</td>
      <td>1.6</td>
      <td>30.7</td>
      <td>36.791</td>
      <td>1.7</td>
      <td>61.49</td>
      <td>0.571</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>157475</th>
      <td>ZWE</td>
      <td>Africa</td>
      <td>Zimbabwe</td>
      <td>2022-01-26</td>
      <td>228943.0</td>
      <td>167.0</td>
      <td>293.714</td>
      <td>5321.0</td>
      <td>5.0</td>
      <td>7.857</td>
      <td>15169.653</td>
      <td>11.065</td>
      <td>19.461</td>
      <td>352.567</td>
      <td>0.331</td>
      <td>0.521</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15092171.0</td>
      <td>42.729</td>
      <td>19.6</td>
      <td>2.822</td>
      <td>1.882</td>
      <td>1899.775</td>
      <td>21.4</td>
      <td>307.846</td>
      <td>1.82</td>
      <td>1.6</td>
      <td>30.7</td>
      <td>36.791</td>
      <td>1.7</td>
      <td>61.49</td>
      <td>0.571</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>157476 rows × 67 columns</p>

      
      
## Check Missing Columns

First step of any data analysis is checking for missing columns.


```python
total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending = False)
mdf = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
mdf = mdf.reset_index()
mdf
```

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
      <td>weekly_icu_admissions</td>
      <td>153085</td>
      <td>0.972116</td>
    </tr>
    <tr>
      <th>1</th>
      <td>weekly_icu_admissions_per_million</td>
      <td>153085</td>
      <td>0.972116</td>
    </tr>
    <tr>
      <th>2</th>
      <td>excess_mortality_cumulative_per_million</td>
      <td>152056</td>
      <td>0.965582</td>
    </tr>
    <tr>
      <th>3</th>
      <td>excess_mortality</td>
      <td>152056</td>
      <td>0.965582</td>
    </tr>
    <tr>
      <th>4</th>
      <td>excess_mortality_cumulative_absolute</td>
      <td>152056</td>
      <td>0.965582</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>62</th>
      <td>total_cases</td>
      <td>2850</td>
      <td>0.018098</td>
    </tr>
    <tr>
      <th>63</th>
      <td>population</td>
      <td>1037</td>
      <td>0.006585</td>
    </tr>
    <tr>
      <th>64</th>
      <td>date</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>65</th>
      <td>location</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>66</th>
      <td>iso_code</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>67 rows × 3 columns</p>


It seems that we have lots of missing data (97%+).

## Pie Chart

#### Missing Values Columns
How about plotting the counts of missing columns in pie chart?

To make it more fast, we will be using only columns that are missing more than 100000 values.


```python
mdf.query("Total>100000").iplot(kind='pie',labels = "index", 
                                values="Total", textinfo="percent+label",
                                title='Top Columns with Missing Values', hole = 0.5)
```

![pie1]({{site.url}}/assets/plotly/pie1.png)

Above plot seems little bit dirty and we could smoothen it by not providing textinfo.


```python
mdf.query("Total>100000").iplot(kind='pie',labels = "index", 
                                values="Total",
                                title='Top Columns with Missing Values', hole = 0.5)
```

![pie2]({{site.url}}/assets/plotly/pie2.png)


## Line Chart

### New Cases Per day

The location field of our data seems to be having country name, continent name and world so we will skip those locations first. Then we will calculate the aggregated value of each day by grouping on date level

Lets first plot simple line chart with only total cases. But we could always plot more lines within it.


```python
todf = df[~df.location.isin(["Lower middle income", "North America", "World", "Asia", "Europe", 
                           "European Union", "Upper middle income", 
                           "High income", "South America"])]
tdf = todf.groupby("date").aggregate(new_cases=("new_cases", "sum"),
                                   new_deaths = ("new_deaths", "sum"),
                                   new_vaccinations = ("new_vaccinations", "sum"),
                                   new_tests = ("new_tests", "sum")
                                   ).reset_index()

tdf.iplot(kind="line",
          y="new_cases",
          x="date",
          xTitle="Date",
          width=2,
          yTitle="new_cases", 
          title="New Cases from Jan 2020 to Jan 2022")
```

![line1]({{site.url}}/assets/plotly/line1.png)


Above plot seems to be cool but now lets plot multiple lines at the same time on same figure.


```python
tdf.iplot(kind="line",
          y=["new_deaths", "new_vaccinations", "new_tests"],
          x="date",
          xTitle="Date",
          width=2,
          yTitle="Cases", 
          title="Cases from Jan 2020 to Jan 2022")
```

![line2]({{site.url}}/assets/plotly/line2.png)

It does not look that good because the new_deaths is not clearly visible lets draw them in sub plots so that we could see each lines distinctly.


```python
tdf.iplot(kind="line",
          y=["new_deaths", "new_vaccinations", "new_tests"],
          x="date",
          xTitle="Date",
          width=2,
          yTitle="Cases", 
          subplots=True,
          title="Cases from Jan 2020 to Jan 2022")
```
![line3]({{site.url}}/assets/plotly/line3.png)

Now its better.

We could even plot secondary y variable. Now lets plot new tests and new vaccinations side by side.


```python
tdf.iplot(kind="line",
          y=["new_vaccinations"],
          secondary_y = "new_tests",
          x="date",
          xTitle="Date",
          width=2,
          yTitle="new_vaccinations",
          secondary_y_title="new_tests", 
          title="Cases from Jan 2020 to Jan 2022")
```
![line4]({{site.url}}/assets/plotly/line4.png)



In above plot, we were able to insert two y axes.

## Scatter Plot

### New deaths vs New Cases
How about viewing the deaths vs cases in scatter plot?



```python
tdf.iplot(kind="scatter",
              y="new_deaths", x='new_cases',
              mode='markers',
              yTitle="New Deaths", xTitle="New Cases",
              title="New Deaths vs New Cases")
```
![scatter1]({{site.url}}/assets/plotly/scatter1.png)

It seems that most of the deaths happened while cases were little.

We could even plot secondary y. Lets visualize new tests along with them.


```python
tdf.iplot(kind="scatter",
              x="new_deaths", y='new_cases',
              secondary_y="new_tests",
              secondary_y_title="New Tests",
              mode='markers',
              xTitle="New Deaths", yTitle="New Cases",
              title="New Deaths vs New Cases")
```

![scatter2]({{site.url}}/assets/plotly/scatter2.png)


We could even use subplots on it.


```python
tdf.iplot(kind="scatter",
              x="new_deaths", y='new_cases',
              secondary_y="new_tests",
              secondary_y_title="New Tests",
              mode='markers',
              subplots=True,
              xTitle="New Deaths", yTitle="New Cases",
              title="New Deaths vs New Cases")
```

![scatter3]({{site.url}}/assets/plotly/scatter3.png)

## Bar Plot

How about plotting top 20 countries where most death have occured?

But first, take the aggregate data by taking maximum of total deaths column. Thanks to the author of this dataset we do not have to make our hands dirty much. Then take top 20 by using `nlargest`.


```python
tdf = df[~df.location.isin(["Lower middle income", "North America", "World", "Asia", "Europe", 
                           "European Union", "Upper middle income", 
                           "High income", "South America"])].groupby("location").aggregate(total_deaths=("total_deaths", "max"),
                                                                                           total_cases = ("total_cases", "max"),
                                                                                           total_tests = ("total_tests", "max")).reset_index()
topdf = tdf.nlargest(20, "total_deaths")

```


```python
topdf.iplot(kind="bar", x="location",
                                      y="total_deaths",
                                      theme="polar",
                                      xTitle="Countries", yTitle="Total Deaths", 
                                       title="Top 20 Countries according to total deaths")
```

![bar1]({{site.url}}/assets/plotly/bar.png)
It seems awesome. We could play with theme also.

We could even make it horizontal.


```python
topdf.iplot(kind="bar", x="location",
            y="total_deaths",
            theme="polar", orientation='h',
            xTitle="Countries", yTitle="Total Deaths", 
            title="Top 20 Countries according to total deaths")
```

![bar2]({{site.url}}/assets/plotly/bar1.png)

We could even plot multiple bars at the same time. In seaborn, we could do this by using Hue but here, we only have to pass it in y. Lets plot bars of total deaths, total cases and total tests.


```python
topdf.iplot(kind="bar", x="location",
            y=["total_deaths", "total_cases", "total_tests"],
            theme="polar",
            xTitle="Countries", yTitle="Total Deaths", 
            title="Top 20 Countries according to total deaths")
```

![bar3]({{site.url}}/assets/plotly/bar2.png)

But total deaths is not visible clearly, lets try to use different mode of bar. We could choose one from the ` 'stack', 'group', 'overlay', 'relative'`.


```python
topdf.iplot(kind="bar", x="location",
                        y=["total_deaths", "total_cases", "total_tests"],
                        theme="polar",
                        barmode="overlay",
                        xTitle="Countries", yTitle="Total Deaths", 
                        title="Top 20 Countries according to total deaths")
```

![bar4]({{site.url}}/assets/plotly/bar3.png)

But it is still not clear. One solution is to plot in subplots.


```python
topdf.iplot(kind="bar", x="location",
                        y=["total_deaths", "total_cases", "total_tests"],
                        theme="polar",
                        barmode="overlay",
                        xTitle="Countries", yTitle="Total Deaths", 
                        subplots=True,
                        title="Top 20 Countries according to total deaths")
```

![bar5]({{site.url}}/assets/plotly/bar4.png)

Much better.

## Histogram Chart
How about viewing the distribution of totel tests done?


```python
tdf.iplot(kind="hist",
              bins=50, 
              colors=["red"],
              keys=["total_tests"],
              title="Total tests Histogram")
```

![hist1]({{site.url}}/assets/plotly/hist1.png)

To see histogram of other columns in same figure we will use keys.


```python
tdf.iplot(kind="hist",
              bins=100, 
              colors=["red"],
              keys=["total_tests", "total_cases", "total_deaths"],
              title="Multiple Histogram")
```
![hist2]({{site.url}}/assets/plotly/hist2.png)


It does not look good as the data is not distributed properly. Lets visualize it in different plots.

```python
tdf.iplot(kind="hist",
              subplots=True,
              keys=["total_tests", "total_cases", "total_deaths"],
              title="Multiple Histogram")
```
![hist3]({{site.url}}/assets/plotly/hist3.png)

## Box Plot

How about viewing outliers in data?

```python
tdf.iplot(kind="box",
              keys=["total_tests", "total_cases", "total_deaths"], 
              boxpoints="outliers",
              x="location",
              xTitle="Columns", title="Box Plot Tests, Cases and Deaths")
```

![box1]({{site.url}}/assets/plotly/box1.png)

It is not clearly visible as the data have lot of outliers and not all columns have similar distributions.

```python
tdf.iplot(kind="box",
              keys=["total_tests", "total_cases", "total_deaths"], 
              boxpoints="outliers",
              x="location",
              subplots=True,
              xTitle="Columns", title="Box Plot Tests, Cases and Deaths")
```
![box2]({{site.url}}/assets/plotly/box2.png)

## HeatMaps

How about viewing the correlation between columns? We will not check with all the 67 columns but lets test with 3.


```python
df[["new_cases", "new_deaths", "new_tests"]].corr().iplot(kind="heatmap")
```

![corr1]({{site.url}}/assets/plotly/corr1.png)

Simple yet much informative and interactive right?

## Choropleth on Map
Plotting on map was once mine dream but now it can be done within few clicks.

### Lets plot a choropleth on world map for the total deaths as of the latest day


```python
import plotly.graph_objects as go

ldf = df[~df.location.isin(["Lower middle income", "North America", "World", "Asia", "Europe", 
                           "European Union", "Upper middle income", 
                           "High income", "South America"])].drop_duplicates("location", keep="last") 

fig = go.Figure(data=go.Choropleth(
    locations = ldf['iso_code'],
    z = ldf['total_deaths'],
    text = ldf['location'],
    colorscale = 'Blues',
    autocolorscale=False,
    reversescale=True,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_title = 'total_deaths',
))

fig.update_layout(
    title_text='total_deaths vs Country',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    )
)

fig.show()
```

![choro1]({{site.url}}/assets/plotly/choro1.png)

Above plot is of current date only but what if w want to view data of each available date?

## Choropleth with Slider

We could add a slider to slide between different dates but it will be too much power hungry plot so beware of your system. We will plot total number of cases at the end of the month for each country.


```python
tldf = df[~df.location.isin(["Lower middle income", "North America", "World", "Asia", "Europe", 
                           "European Union", "Upper middle income", 
                           "High income", "South America"])]
tldf = tldf.groupby(["location", "iso_code", pd.Grouper(key="date", freq="1M")]).aggregate(total_cases=("total_cases", "max")).reset_index()
tldf["date"] = tldf["date"].dt.date
tldf

```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location</th>
      <th>iso_code</th>
      <th>date</th>
      <th>total_cases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>2020-02-29</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>2020-03-31</td>
      <td>166.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>2020-04-30</td>
      <td>1827.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>2020-05-31</td>
      <td>15180.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>2020-06-30</td>
      <td>31445.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5101</th>
      <td>Zimbabwe</td>
      <td>ZWE</td>
      <td>2021-09-30</td>
      <td>130820.0</td>
    </tr>
    <tr>
      <th>5102</th>
      <td>Zimbabwe</td>
      <td>ZWE</td>
      <td>2021-10-31</td>
      <td>132977.0</td>
    </tr>
    <tr>
      <th>5103</th>
      <td>Zimbabwe</td>
      <td>ZWE</td>
      <td>2021-11-30</td>
      <td>134625.0</td>
    </tr>
    <tr>
      <th>5104</th>
      <td>Zimbabwe</td>
      <td>ZWE</td>
      <td>2021-12-31</td>
      <td>213258.0</td>
    </tr>
    <tr>
      <th>5105</th>
      <td>Zimbabwe</td>
      <td>ZWE</td>
      <td>2022-01-31</td>
      <td>228943.0</td>
    </tr>
  </tbody>
</table>
<p>5106 rows × 4 columns</p>


```python

first_day = tldf.date.min()

scl = [[0.0, '#ffffff'],[0.2, '#b4a8ce'],[0.4, '#8573a9'],
       [0.6, '#7159a3'],[0.8, '#5732a1'],[1.0, '#2c0579']] # purples

data_slider = []
for date in tldf['date'].unique():
    df_segmented =  tldf[(tldf['date']== date)]

    for col in df_segmented.columns:
        df_segmented[col] = df_segmented[col].astype(str)

    data_each_yr = dict(
                        type='choropleth',
                        locations = df_segmented['iso_code'],
                        z=df_segmented["total_cases"].astype(float),
                        colorbar= {'title':'Total Cases'}
                        )

    data_slider.append(data_each_yr)

steps = []
for i,date in enumerate(tldf.date.unique()):
    step = dict(method='restyle',
                args=['visible', [False] * len(data_slider)],
                label='Date {}'.format(date))
    step['args'][1][i] = True
    steps.append(step)

sliders = [dict(active=0, pad={"t": 1}, steps=steps)]

layout = dict(title ='Total Cases at the End of Month Across the World',
              sliders=sliders)

fig = dict(data=data_slider, layout=layout)
iplot(fig)
```
![choro2]({{site.url}}/assets/plotly/choro2.png)

If I have to explain the above code, we have created a data for each of slider point and in our case a slider's single point is end of the month.
* Loop through unique date.
  * Mask the data to get data of current date.
  * Make a dictionary by giving common and essential values required to make a `chloropeth`.
  * Give locations as `iso_code`.
  * Give z axis as total cases.  
  * And use total cases on color bar title.
  * Add this data to slider.
* For each date step, prepare a label.
* Update sliders and layout then make figure and plot it using iplot.

## Density Mapbox
Another useful plot is density map box where we will plot density plot on the map. But we need longitude and latitude for that. And I have prepared it in GitHub already. Please find it on below link:
* [State Location Coordinates](https://github.com/q-viper/State-Location-Coordinates/)


```python
country_df = pd.read_csv("https://github.com/q-viper/State-Location-Coordinates/raw/main/world_country.csv")
country_df = country_df[["country", "lon", "lat", "iso_con"]]
tldf["country"] = tldf.location
tldf = tldf.merge(country_df[["country", "lat", "lon"]], on="country")
```


```python
tldf.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location</th>
      <th>iso_code</th>
      <th>date</th>
      <th>total_cases</th>
      <th>country</th>
      <th>lat_x</th>
      <th>lon_x</th>
      <th>lat_y</th>
      <th>lon_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>2020-02-29</td>
      <td>5.0</td>
      <td>Afghanistan</td>
      <td>33.768006</td>
      <td>66.238514</td>
      <td>33.768006</td>
      <td>66.238514</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>2020-03-31</td>
      <td>166.0</td>
      <td>Afghanistan</td>
      <td>33.768006</td>
      <td>66.238514</td>
      <td>33.768006</td>
      <td>66.238514</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>2020-04-30</td>
      <td>1827.0</td>
      <td>Afghanistan</td>
      <td>33.768006</td>
      <td>66.238514</td>
      <td>33.768006</td>
      <td>66.238514</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>2020-05-31</td>
      <td>15180.0</td>
      <td>Afghanistan</td>
      <td>33.768006</td>
      <td>66.238514</td>
      <td>33.768006</td>
      <td>66.238514</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>2020-06-30</td>
      <td>31445.0</td>
      <td>Afghanistan</td>
      <td>33.768006</td>
      <td>66.238514</td>
      <td>33.768006</td>
      <td>66.238514</td>
    </tr>
  </tbody>
</table>


```python
import plotly.express as px


fig = px.density_mapbox(tldf.drop_duplicates(keep="last"), 
                          lat = tldf["lat"],
                          lon = tldf["lon"],
                          hover_name="location", 
                          hover_data=["total_cases"], 
                          color_continuous_scale="Portland",
                          radius=7, 
                          zoom=0,
                          height=700,
                          z="total_cases"
                          )
fig.update_layout(title=f'Country vs total_cases',
                  font=dict(family="Courier New, monospace",
                            size=18,
                            color="#7f7f7f")
                )
fig.update_layout(mapbox_style="open-street-map", mapbox_center_lon=0)


fig.show()
```
![map1]({{site.url}}/assets/plotly/map1.png)

Density map plot is useful and clear when we are ploting onto state or city because it will make our plot little bit visible. Here it is not clearly visible.

## Density Mapbox with Slider

```python

first_day = tldf.date.min()

scl = [[0.0, '#ffffff'],[0.2, '#b4a8ce'],[0.4, '#8573a9'],
       [0.6, '#7159a3'],[0.8, '#5732a1'],[1.0, '#2c0579']] # purples

data_slider = []
for date in tldf['date'].unique():
    df_segmented =  tldf[(tldf['date']== date)]

    for col in df_segmented.columns:
        df_segmented[col] = df_segmented[col].astype(str)

    data_each_yr = dict(
                        type='densitymapbox',
                        lat = df_segmented["lat"],
                        lon = df_segmented["lon"],
                        hoverinfo="text",
                        # name = "country",
                        text = df_segmented["country"],                        
                        z=df_segmented["total_cases"].astype(float),
                        colorbar= {'title':'Total Cases'}
                        )

    data_slider.append(data_each_yr)

steps = []
for i,date in enumerate(tldf.date.unique()):
    step = dict(method='restyle',
                args=['visible', [False] * len(data_slider)],
                label='Date {}'.format(date))
    step['args'][1][i] = True
    steps.append(step)

sliders = [dict(active=0, pad={"t": 1}, steps=steps)]

layout = dict(mapbox_style="open-street-map",
              title ='Total Cases at the End of Month Across the World',
              sliders=sliders)

fig = dict(data=data_slider, layout=layout)

iplot(fig)
```
![map2]({{site.url}}/assets/plotly/map2.png)


## References
* [Sliders](https://plotly.com/python/sliders/)
* [Cufflinks How To Create Plotly Charts From Pandas Dataframe With one Line of Code](https://coderzcolumn.com/tutorials/data-science/cufflinks-how-to-create-plotly-charts-from-pandas-dataframe-with-one-line-of-code)
