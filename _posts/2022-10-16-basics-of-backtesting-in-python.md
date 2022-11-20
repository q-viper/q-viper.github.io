---
title:  Basics of Backtesting in Python
date:   2022-10-16 01:29:17 +0545
categories:
    - Python
    - Stock
    - Backtesting
tags:
    - python
    - stock
    - backtesting
---
Stock Backtesting in Python is way of testing our strategy in a historical data to see if our strategy makes any money or not. Let's start with a simple story.

John and Joe are two best friends. They both earned some money from their hard working corporate job and wanted to invest it in a stock market. Unlike Joe, John is clever and does not fall for any influence of stock's price increasing and decreasing. They studied some Statistics and Probability along with Economics in college and and they love their money. Joe followed trend and bought some stock of X and felt glad that his stock's price increased by some % in few days. John was calm person and thought that John's stock position is increased but he is not earning any money and only way to earn is by selling it. John wanted to get back in time and questioned himself what will happen if I try to buy some stock of X and sell it if price increased by 10% or decrease by 5%. Then I will buy as much stock as possible from the amount I have. How much would have I earned today? Well he did not know but what he tried to do is a simple stock backtesting example.

Here in this blog, we will start with our very simple strategy and then try to use of the most popular stock backtesting Python package [`Backtesting.py`](https://kernc.github.io/backtesting.py/doc/backtesting/#manuals). But first, let's install it. 


```python
!pip install backtesting
```

    Requirement already satisfied: backtesting in c:\programdata\anaconda3\lib\site-packages (0.3.3)
    Requirement already satisfied: numpy>=1.17.0 in c:\programdata\anaconda3\lib\site-packages (from backtesting) (1.19.2)
    Requirement already satisfied: pandas!=0.25.0,>=0.25.0 in c:\users\viper\appdata\roaming\python\python38\site-packages (from backtesting) (1.3.5)
    Requirement already satisfied: bokeh>=1.4.0 in c:\users\viper\appdata\roaming\python\python38\site-packages (from backtesting) (2.4.3)
    Requirement already satisfied: python-dateutil>=2.7.3 in c:\programdata\anaconda3\lib\site-packages (from pandas!=0.25.0,>=0.25.0->backtesting) (2.8.1)
    Requirement already satisfied: pytz>=2017.3 in c:\programdata\anaconda3\lib\site-packages (from pandas!=0.25.0,>=0.25.0->backtesting) (2020.1)
    Requirement already satisfied: Jinja2>=2.9 in c:\programdata\anaconda3\lib\site-packages (from bokeh>=1.4.0->backtesting) (2.11.2)
    Requirement already satisfied: tornado>=5.1 in c:\programdata\anaconda3\lib\site-packages (from bokeh>=1.4.0->backtesting) (6.0.4)
    Requirement already satisfied: packaging>=16.8 in c:\programdata\anaconda3\lib\site-packages (from bokeh>=1.4.0->backtesting) (20.4)
    Requirement already satisfied: typing-extensions>=3.10.0 in c:\users\viper\appdata\roaming\python\python38\site-packages (from bokeh>=1.4.0->backtesting) (4.3.0)
    Requirement already satisfied: pillow>=7.1.0 in c:\programdata\anaconda3\lib\site-packages (from bokeh>=1.4.0->backtesting) (8.0.1)
    Requirement already satisfied: PyYAML>=3.10 in c:\users\viper\appdata\roaming\python\python38\site-packages (from bokeh>=1.4.0->backtesting) (6.0)
    Requirement already satisfied: six>=1.5 in c:\programdata\anaconda3\lib\site-packages (from python-dateutil>=2.7.3->pandas!=0.25.0,>=0.25.0->backtesting) (1.15.0)
    Requirement already satisfied: MarkupSafe>=0.23 in c:\programdata\anaconda3\lib\site-packages (from Jinja2>=2.9->bokeh>=1.4.0->backtesting) (1.1.1)
    Requirement already satisfied: pyparsing>=2.0.2 in c:\users\viper\appdata\roaming\python\python38\site-packages (from packaging>=16.8->bokeh>=1.4.0->backtesting) (3.0.9)
    

Before going into stock backtesting, lets choose the data of any stock. We will choose data of AAPL from `yfinance`. If it is not installed, we can do so by `pip install yfinance`.


```python
!pip install yfinance --user
```

    Requirement already satisfied: yfinance in c:\users\viper\appdata\roaming\python\python38\site-packages (0.1.87)
    Requirement already satisfied: requests>=2.26 in c:\users\viper\appdata\roaming\python\python38\site-packages (from yfinance) (2.28.1)
    Requirement already satisfied: pandas>=0.24.0 in c:\users\viper\appdata\roaming\python\python38\site-packages (from yfinance) (1.3.5)
    Requirement already satisfied: multitasking>=0.0.7 in c:\users\viper\appdata\roaming\python\python38\site-packages (from yfinance) (0.0.11)
    Requirement already satisfied: appdirs>=1.4.4 in c:\programdata\anaconda3\lib\site-packages (from yfinance) (1.4.4)
    Requirement already satisfied: lxml>=4.5.1 in c:\programdata\anaconda3\lib\site-packages (from yfinance) (4.6.1)
    Requirement already satisfied: numpy>=1.15 in c:\programdata\anaconda3\lib\site-packages (from yfinance) (1.19.2)
    Requirement already satisfied: charset-normalizer<3,>=2 in c:\users\viper\appdata\roaming\python\python38\site-packages (from requests>=2.26->yfinance) (2.1.0)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\users\viper\appdata\roaming\python\python38\site-packages (from requests>=2.26->yfinance) (1.25.11)
    Requirement already satisfied: certifi>=2017.4.17 in c:\programdata\anaconda3\lib\site-packages (from requests>=2.26->yfinance) (2020.6.20)
    Requirement already satisfied: idna<4,>=2.5 in c:\programdata\anaconda3\lib\site-packages (from requests>=2.26->yfinance) (2.10)
    Requirement already satisfied: pytz>=2017.3 in c:\programdata\anaconda3\lib\site-packages (from pandas>=0.24.0->yfinance) (2020.1)
    Requirement already satisfied: python-dateutil>=2.7.3 in c:\programdata\anaconda3\lib\site-packages (from pandas>=0.24.0->yfinance) (2.8.1)
    Requirement already satisfied: six>=1.5 in c:\programdata\anaconda3\lib\site-packages (from python-dateutil>=2.7.3->pandas>=0.24.0->yfinance) (1.15.0)
    


```python
import pandas as pd
import yfinance as yf
```

Next is to download data. We can download data as following.


```python
data = yf.download("AAPL", start="2015-01-01", end="2022-04-30")
del data['Adj Close']
del data['Volume']
data
```

    [*********************100%***********************]  1 of 1 completed
    





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-02</th>
      <td>27.847500</td>
      <td>27.860001</td>
      <td>26.837500</td>
      <td>27.332500</td>
    </tr>
    <tr>
      <th>2015-01-05</th>
      <td>27.072500</td>
      <td>27.162500</td>
      <td>26.352501</td>
      <td>26.562500</td>
    </tr>
    <tr>
      <th>2015-01-06</th>
      <td>26.635000</td>
      <td>26.857500</td>
      <td>26.157499</td>
      <td>26.565001</td>
    </tr>
    <tr>
      <th>2015-01-07</th>
      <td>26.799999</td>
      <td>27.049999</td>
      <td>26.674999</td>
      <td>26.937500</td>
    </tr>
    <tr>
      <th>2015-01-08</th>
      <td>27.307501</td>
      <td>28.037500</td>
      <td>27.174999</td>
      <td>27.972500</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2022-04-25</th>
      <td>161.119995</td>
      <td>163.169998</td>
      <td>158.460007</td>
      <td>162.880005</td>
    </tr>
    <tr>
      <th>2022-04-26</th>
      <td>162.250000</td>
      <td>162.339996</td>
      <td>156.720001</td>
      <td>156.800003</td>
    </tr>
    <tr>
      <th>2022-04-27</th>
      <td>155.910004</td>
      <td>159.789993</td>
      <td>155.380005</td>
      <td>156.570007</td>
    </tr>
    <tr>
      <th>2022-04-28</th>
      <td>159.250000</td>
      <td>164.520004</td>
      <td>158.929993</td>
      <td>163.639999</td>
    </tr>
    <tr>
      <th>2022-04-29</th>
      <td>161.839996</td>
      <td>166.199997</td>
      <td>157.250000</td>
      <td>157.649994</td>
    </tr>
  </tbody>
</table>
<p>1845 rows × 4 columns</p>



Our data will be daily floorsheet data and we will make strategy on it. Alternatively we could get data for testing from backtesting.py too but it only allows GOOG.


```python
import backtesting.test as btest
```


```python
btest.GOOG
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2004-08-19</th>
      <td>100.00</td>
      <td>104.06</td>
      <td>95.96</td>
      <td>100.34</td>
      <td>22351900</td>
    </tr>
    <tr>
      <th>2004-08-20</th>
      <td>101.01</td>
      <td>109.08</td>
      <td>100.50</td>
      <td>108.31</td>
      <td>11428600</td>
    </tr>
    <tr>
      <th>2004-08-23</th>
      <td>110.75</td>
      <td>113.48</td>
      <td>109.05</td>
      <td>109.40</td>
      <td>9137200</td>
    </tr>
    <tr>
      <th>2004-08-24</th>
      <td>111.24</td>
      <td>111.60</td>
      <td>103.57</td>
      <td>104.87</td>
      <td>7631300</td>
    </tr>
    <tr>
      <th>2004-08-25</th>
      <td>104.96</td>
      <td>108.00</td>
      <td>103.88</td>
      <td>106.00</td>
      <td>4598900</td>
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
      <th>2013-02-25</th>
      <td>802.30</td>
      <td>808.41</td>
      <td>790.49</td>
      <td>790.77</td>
      <td>2303900</td>
    </tr>
    <tr>
      <th>2013-02-26</th>
      <td>795.00</td>
      <td>795.95</td>
      <td>784.40</td>
      <td>790.13</td>
      <td>2202500</td>
    </tr>
    <tr>
      <th>2013-02-27</th>
      <td>794.80</td>
      <td>804.75</td>
      <td>791.11</td>
      <td>799.78</td>
      <td>2026100</td>
    </tr>
    <tr>
      <th>2013-02-28</th>
      <td>801.10</td>
      <td>806.99</td>
      <td>801.03</td>
      <td>801.20</td>
      <td>2265800</td>
    </tr>
    <tr>
      <th>2013-03-01</th>
      <td>797.80</td>
      <td>807.14</td>
      <td>796.15</td>
      <td>806.19</td>
      <td>2175400</td>
    </tr>
  </tbody>
</table>
<p>2148 rows × 5 columns</p>




## Preparing SMA


We will work on our data from yfinance next. There is a good availability of classes and modules in backtesting and lets use them instead of writing our own indicators. But I have written many indicators from scratch and [you can find them here](https://q-viper.github.io/2022/03/20/python-for-stock-market-analysis-technical-indicators/). Here, SMA stands for Simple Moving Average.


We start by making a class that inherits `Strategy` class inside backtesting and we do not need anything at all at this time but lets use `crossover` and `SMA` too. But this will be covered later. First lets take a look into our data and try to plot SMA of two periods, one longer and one shorter. One SMA of 20 days and another of 40 days. Our simple strategy will be to buy when small SMA crosses over bigger SMA. 


```python
n1,n2=20,40

ndata = data.copy()

ndata[f'SMA_{n1}'] = ndata.Close.rolling(n1).mean()
ndata[f'SMA_{n2}'] = ndata.Close.rolling(n2).mean()

ndata[[f'SMA_{n1}', f'SMA_{n2}']].plot(figsize=(15,10))
```




    <AxesSubplot:xlabel='Date'>




    
![png]({{site.url}}/assets/backtesting/output_11_1.png)
    


We can see that SMA_20 and SMA_40 are crossing over each other in multiple times. But the plot looks little huge so lets take data of last 200 days only.


```python
last = 200
n1,n2=20,40

tdata = data.copy().tail(last)

tdata[f'SMA_{n1}'] = tdata.Close.rolling(n1).mean()
tdata[f'SMA_{n2}'] = tdata.Close.rolling(n2).mean()

tdata[[f'SMA_{n1}', f'SMA_{n2}']].plot(figsize=(15,10))
```




    <AxesSubplot:xlabel='Date'>




    
![png]({{site.url}}/assets/backtesting/output_13_1.png)
    


## Our Simple Strategy

Now let's make our stock backtesting strategy. If the short SMA crosses over large SMA, we buy and hold positions because we saw that it has increased the value of price recently and could increase in future too. But if short SMA crosses below large SMA, we sell our holding positions because there has been recent price drops. In above example we will do trades whenever crossover happens. A simple way to find a crossover is by comparing difference between current price and previous. If the difference was positive in past and negative now then we do trade and vice versa. Note that we buy on the Open price of next day.


```python
tdata['sma1_gt_sma2'] = tdata[f'SMA_{n1}']>tdata[f'SMA_{n2}']
tdata['crossed'] = (tdata.sma1_gt_sma2!=tdata.sma1_gt_sma2.shift(1))
print(f"Num Corssed: {tdata.crossed.sum()-1}")
tdata

```

    Num Corssed: 7
    




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>SMA_20</th>
      <th>SMA_40</th>
      <th>sma1_gt_sma2</th>
      <th>crossed</th>
    </tr>
    <tr>
      <th>Date</th>
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
      <th>2021-07-16</th>
      <td>148.460007</td>
      <td>149.759995</td>
      <td>145.880005</td>
      <td>146.389999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2021-07-19</th>
      <td>143.750000</td>
      <td>144.070007</td>
      <td>141.669998</td>
      <td>142.449997</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2021-07-20</th>
      <td>143.460007</td>
      <td>147.100006</td>
      <td>142.960007</td>
      <td>146.149994</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2021-07-21</th>
      <td>145.529999</td>
      <td>146.130005</td>
      <td>144.630005</td>
      <td>145.399994</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2021-07-22</th>
      <td>145.940002</td>
      <td>148.199997</td>
      <td>145.809998</td>
      <td>146.800003</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
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
    </tr>
    <tr>
      <th>2022-04-25</th>
      <td>161.119995</td>
      <td>163.169998</td>
      <td>158.460007</td>
      <td>162.880005</td>
      <td>170.435000</td>
      <td>166.72550</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-04-26</th>
      <td>162.250000</td>
      <td>162.339996</td>
      <td>156.720001</td>
      <td>156.800003</td>
      <td>169.495000</td>
      <td>166.51750</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-04-27</th>
      <td>155.910004</td>
      <td>159.789993</td>
      <td>155.380005</td>
      <td>156.570007</td>
      <td>168.375500</td>
      <td>166.35175</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-04-28</th>
      <td>159.250000</td>
      <td>164.520004</td>
      <td>158.929993</td>
      <td>163.639999</td>
      <td>167.668999</td>
      <td>166.27875</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-04-29</th>
      <td>161.839996</td>
      <td>166.199997</td>
      <td>157.250000</td>
      <td>157.649994</td>
      <td>166.820999</td>
      <td>166.06425</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 8 columns</p>




```python

```

In above code, we made new column where we checked if SMA1 is higher than SMA2 or not and in next crossed column we checked if the status of SMA1>SMA2 still holds same from the previous time. And when it is false, we do trade. We should ignore the first one because it will give us NaN value on shift. Let's assume that we have USD 10000 in cash and want to do trade. Since we have SMA1>SMA2 column, we buy only when there is `crossed` True and `sma1_gt_sma2` True as well. And we sell only when there is `crossed` True and `sma1_gt_sma2` is False.

### Trading Result
To find stock backtesting trades data, we loop through the data and if yesterday's SMA1>SMA2 then we buy on today's Open price and selling happens on same way.

* If crossed and SMA1>SMA2: buy positions based on remaining amount and add positions.
* If crossed and SMA1<SMA2: sell available positions and add remaining amount.
* On last day sell all positions and add remaining amount.


```python
ntdata = tdata.reset_index().copy()
ntdata['crossed']=ntdata.crossed.shift(1)
ntdata['sma1_gt_sma2']=ntdata.sma1_gt_sma2.shift(1)

positions = 0
rem_amt=10000
lr = len(ntdata)-1
trades = []
tinfo=[]

for i, row in ntdata.iterrows():
    if i!=0:
        if row.crossed and row.sma1_gt_sma2:
            positions=int(rem_amt/row.Open)
            rem_amt= rem_amt-row.Open*positions
            tinfo.append(positions)
            tinfo.append(row.Open)
            tinfo.append(row.Date)
            
        if row.crossed==True and row.sma1_gt_sma2==False and positions>0:
            rem_amt = rem_amt+row.Open*positions
            tinfo.append(row.Date)
            tinfo.append(row.Open)
            trades.append(tinfo)
            tinfo=[]
            positions = 0
    
    if i==lr and positions>0:
        rem_amt=rem_amt + positions*row.Open
        
        tinfo.append(row.Date)
        tinfo.append(row.Open)
        trades.append(tinfo)
        
        positions = 0
        ntdata.loc[i, 'positions'] = positions
        ntdata.loc[i, 'rem_amount'] = rem_amt

    
trades = pd.DataFrame(trades, columns=['Positions', 'Buy', 'Entry', 'Exit', 'Sell'])
trades['return']=((trades['Sell']-trades['Buy'])*trades.Positions).cumsum()

trades
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Positions</th>
      <th>Buy</th>
      <th>Entry</th>
      <th>Exit</th>
      <th>Sell</th>
      <th>return</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>66</td>
      <td>150.630005</td>
      <td>2021-09-13</td>
      <td>2021-10-01</td>
      <td>141.899994</td>
      <td>-576.180725</td>
    </tr>
    <tr>
      <th>1</th>
      <td>62</td>
      <td>150.389999</td>
      <td>2021-11-03</td>
      <td>2022-01-27</td>
      <td>162.449997</td>
      <td>171.539124</td>
    </tr>
    <tr>
      <th>2</th>
      <td>61</td>
      <td>164.699997</td>
      <td>2022-03-01</td>
      <td>2022-03-02</td>
      <td>164.389999</td>
      <td>152.629272</td>
    </tr>
    <tr>
      <th>3</th>
      <td>58</td>
      <td>172.360001</td>
      <td>2022-04-06</td>
      <td>2022-04-29</td>
      <td>161.839996</td>
      <td>-457.530975</td>
    </tr>
  </tbody>
</table>



Looking over the above table, in return column, we are in 457 loss overall. What if we did this testing with larger period of data?

## Our Strategy in a Larger Period

Lets start from the last 1000 day and forth.


```python
n1,n2=20,40
last = 1000

tdata = data.copy().tail(last)

tdata[f'SMA_{n1}'] = tdata.Close.rolling(n1).mean()
tdata[f'SMA_{n2}'] = tdata.Close.rolling(n2).mean()

tdata['sma1_gt_sma2'] = tdata[f'SMA_{n1}']>tdata[f'SMA_{n2}']
tdata['crossed'] = (tdata.sma1_gt_sma2!=tdata.sma1_gt_sma2.shift(1))
print(f"Num Corssed: {tdata.crossed.sum()-1}")

ntdata = tdata.reset_index().copy()
ntdata['crossed']=ntdata.crossed.shift(1)
ntdata['sma1_gt_sma2']=ntdata.sma1_gt_sma2.shift(1)

positions = 0
rem_amt=10000
lr = len(ntdata)-1
trades = []
tinfo=[]

for i, row in ntdata.iterrows():
    if i!=0:
        if row.crossed and row.sma1_gt_sma2:
            positions=int(rem_amt/row.Open)
            rem_amt= rem_amt-row.Open*positions
            tinfo.append(positions)
            tinfo.append(row.Open)
            tinfo.append(row.Date)
            
        if row.crossed==True and row.sma1_gt_sma2==False and positions>0:
            rem_amt = rem_amt+row.Open*positions
            tinfo.append(row.Date)
            tinfo.append(row.Open)
            trades.append(tinfo)
            tinfo=[]
            positions = 0
    
    if i==lr and positions>0:
        rem_amt=rem_amt + positions*row.Open
        
        tinfo.append(row.Date)
        tinfo.append(row.Open)
        trades.append(tinfo)
        
        positions = 0
        ntdata.loc[i, 'positions'] = positions
        ntdata.loc[i, 'rem_amount'] = rem_amt

    
trades = pd.DataFrame(trades, columns=['Positions', 'Buy', 'Entry', 'Exit', 'Sell'])
trades['return']=((trades['Sell']-trades['Buy'])*trades.Positions).cumsum()

trades
```

    Num Corssed: 23
    



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Positions</th>
      <th>Buy</th>
      <th>Entry</th>
      <th>Exit</th>
      <th>Sell</th>
      <th>return</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>205</td>
      <td>48.652500</td>
      <td>2018-07-26</td>
      <td>2018-10-26</td>
      <td>53.974998</td>
      <td>1091.112156</td>
    </tr>
    <tr>
      <th>1</th>
      <td>257</td>
      <td>43.099998</td>
      <td>2019-02-07</td>
      <td>2019-05-23</td>
      <td>44.950001</td>
      <td>1566.562744</td>
    </tr>
    <tr>
      <th>2</th>
      <td>232</td>
      <td>49.669998</td>
      <td>2019-06-28</td>
      <td>2019-08-28</td>
      <td>51.025002</td>
      <td>1880.923523</td>
    </tr>
    <tr>
      <th>3</th>
      <td>228</td>
      <td>52.097500</td>
      <td>2019-09-04</td>
      <td>2020-03-02</td>
      <td>70.570000</td>
      <td>6092.653488</td>
    </tr>
    <tr>
      <th>4</th>
      <td>232</td>
      <td>69.300003</td>
      <td>2020-04-24</td>
      <td>2020-09-29</td>
      <td>114.550003</td>
      <td>16590.653488</td>
    </tr>
    <tr>
      <th>5</th>
      <td>233</td>
      <td>114.010002</td>
      <td>2020-10-26</td>
      <td>2020-11-18</td>
      <td>118.610001</td>
      <td>17662.453133</td>
    </tr>
    <tr>
      <th>6</th>
      <td>228</td>
      <td>121.010002</td>
      <td>2020-12-01</td>
      <td>2021-02-26</td>
      <td>122.589996</td>
      <td>18022.691811</td>
    </tr>
    <tr>
      <th>7</th>
      <td>207</td>
      <td>134.940002</td>
      <td>2021-04-14</td>
      <td>2021-05-24</td>
      <td>126.010002</td>
      <td>16174.181747</td>
    </tr>
    <tr>
      <th>8</th>
      <td>194</td>
      <td>134.449997</td>
      <td>2021-06-24</td>
      <td>2021-10-01</td>
      <td>141.899994</td>
      <td>17619.481155</td>
    </tr>
    <tr>
      <th>9</th>
      <td>183</td>
      <td>150.389999</td>
      <td>2021-11-03</td>
      <td>2022-01-27</td>
      <td>162.449997</td>
      <td>19826.460709</td>
    </tr>
    <tr>
      <th>10</th>
      <td>181</td>
      <td>164.699997</td>
      <td>2022-03-01</td>
      <td>2022-03-02</td>
      <td>164.389999</td>
      <td>19770.351151</td>
    </tr>
    <tr>
      <th>11</th>
      <td>172</td>
      <td>172.360001</td>
      <td>2022-04-06</td>
      <td>2022-04-29</td>
      <td>161.839996</td>
      <td>17960.910416</td>
    </tr>
  </tbody>
</table>




It looks like we actually made some money while testing on larger period.

## Strategy with Backtesting
Until now we designed a very simple strategy and did trading and to do so, we had to write too many codes but why do we need to struggle that hard while there is already one open source package available which handles our struggles? Following is a modified version of our strategy and it is modified from the [Quick Start page](https://kernc.github.io/backtesting.py/doc/examples/Quick%20Start%20User%20Guide.html).


```python
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA

class SmaCross(Strategy):
    # Define the two MA lags as *class variables*
    # for later optimization
    n1 = 20
    n2 = 40
    
    def init(self):
        # Precompute the two moving averages
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
    
    def next(self):
        # If sma1 crosses above sma2, close any existing
        # short trades, and buy the asset
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy()

        # Else, if sma1 crosses below sma2, close any existing
        # long trades, and sell the asset
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell()
            

```




```python
from backtesting import Backtest

bt = Backtest(data.tail(last), SmaCross, cash=10000, commission=0)
stats = bt.run()
stats
```




    Start                     2018-05-11 00:00:00
    End                       2022-04-29 00:00:00
    Duration                   1449 days 00:00:00
    Exposure Time [%]                        94.8
    Equity Final [$]                 21836.199413
    Equity Peak [$]                  34464.403912
    Return [%]                         118.361994
    Buy & Hold Return [%]              234.376153
    Return (Ann.) [%]                   21.751022
    Volatility (Ann.) [%]               38.394298
    Sharpe Ratio                         0.566517
    Sortino Ratio                        1.031964
    Calmar Ratio                         0.562183
    Max. Drawdown [%]                  -38.690305
    Avg. Drawdown [%]                   -5.507696
    Max. Drawdown Duration      458 days 00:00:00
    Avg. Drawdown Duration       34 days 00:00:00
    # Trades                                   23
    Win Rate [%]                        52.173913
    Best Trade [%]                      65.295812
    Worst Trade [%]                     -10.50055
    Avg. Trade [%]                       3.457111
    Max. Trade Duration         180 days 00:00:00
    Avg. Trade Duration          60 days 00:00:00
    Profit Factor                        2.831255
    Expectancy [%]                       4.500417
    SQN                                  0.873935
    _strategy                            SmaCross
    _equity_curve                             ...
    _trades                       Size  EntryB...
    dtype: object



We start by importing necessary classes and methods. We create a new class for our own strategy which inherits Strategy. We initialize variables and then SMA.  When doing `run()`,  the `next()` method loops through the data rows and perform checks inside it. We can pass commission percent to calculate how much commission do we have to pay to our broker.

### Trades
The trades table using backtesting is different than ours.


```python
stats['_trades']  # Contains individual trade data
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Size</th>
      <th>EntryBar</th>
      <th>ExitBar</th>
      <th>EntryPrice</th>
      <th>ExitPrice</th>
      <th>PnL</th>
      <th>ReturnPct</th>
      <th>EntryTime</th>
      <th>ExitTime</th>
      <th>Duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>205</td>
      <td>52</td>
      <td>117</td>
      <td>48.652500</td>
      <td>53.974998</td>
      <td>1091.112156</td>
      <td>0.109398</td>
      <td>2018-07-26</td>
      <td>2018-10-26</td>
      <td>92 days</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-205</td>
      <td>117</td>
      <td>186</td>
      <td>53.974998</td>
      <td>43.099998</td>
      <td>2229.375000</td>
      <td>0.201482</td>
      <td>2018-10-26</td>
      <td>2019-02-07</td>
      <td>104 days</td>
    </tr>
    <tr>
      <th>2</th>
      <td>309</td>
      <td>186</td>
      <td>259</td>
      <td>43.099998</td>
      <td>44.950001</td>
      <td>571.650707</td>
      <td>0.042923</td>
      <td>2019-02-07</td>
      <td>2019-05-23</td>
      <td>105 days</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-309</td>
      <td>259</td>
      <td>284</td>
      <td>44.950001</td>
      <td>49.669998</td>
      <td>-1458.479198</td>
      <td>-0.105006</td>
      <td>2019-05-23</td>
      <td>2019-06-28</td>
      <td>36 days</td>
    </tr>
    <tr>
      <th>4</th>
      <td>250</td>
      <td>284</td>
      <td>326</td>
      <td>49.669998</td>
      <td>51.025002</td>
      <td>338.750839</td>
      <td>0.027280</td>
      <td>2019-06-28</td>
      <td>2019-08-28</td>
      <td>61 days</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-250</td>
      <td>326</td>
      <td>330</td>
      <td>51.025002</td>
      <td>52.097500</td>
      <td>-268.124580</td>
      <td>-0.021019</td>
      <td>2019-08-28</td>
      <td>2019-09-04</td>
      <td>7 days</td>
    </tr>
    <tr>
      <th>6</th>
      <td>240</td>
      <td>330</td>
      <td>453</td>
      <td>52.097500</td>
      <td>70.570000</td>
      <td>4433.399963</td>
      <td>0.354576</td>
      <td>2019-09-04</td>
      <td>2020-03-02</td>
      <td>180 days</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-240</td>
      <td>453</td>
      <td>491</td>
      <td>70.570000</td>
      <td>69.300003</td>
      <td>304.799194</td>
      <td>0.017996</td>
      <td>2020-03-02</td>
      <td>2020-04-24</td>
      <td>53 days</td>
    </tr>
    <tr>
      <th>8</th>
      <td>248</td>
      <td>491</td>
      <td>600</td>
      <td>69.300003</td>
      <td>114.550003</td>
      <td>11222.000000</td>
      <td>0.652958</td>
      <td>2020-04-24</td>
      <td>2020-09-29</td>
      <td>158 days</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-248</td>
      <td>600</td>
      <td>619</td>
      <td>114.550003</td>
      <td>114.010002</td>
      <td>133.920227</td>
      <td>0.004714</td>
      <td>2020-09-29</td>
      <td>2020-10-26</td>
      <td>27 days</td>
    </tr>
    <tr>
      <th>10</th>
      <td>250</td>
      <td>619</td>
      <td>636</td>
      <td>114.010002</td>
      <td>118.610001</td>
      <td>1149.999619</td>
      <td>0.040347</td>
      <td>2020-10-26</td>
      <td>2020-11-18</td>
      <td>23 days</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-250</td>
      <td>636</td>
      <td>644</td>
      <td>118.610001</td>
      <td>121.010002</td>
      <td>-600.000381</td>
      <td>-0.020234</td>
      <td>2020-11-18</td>
      <td>2020-12-01</td>
      <td>13 days</td>
    </tr>
    <tr>
      <th>12</th>
      <td>240</td>
      <td>644</td>
      <td>703</td>
      <td>121.010002</td>
      <td>122.589996</td>
      <td>379.198608</td>
      <td>0.013057</td>
      <td>2020-12-01</td>
      <td>2021-02-26</td>
      <td>87 days</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-240</td>
      <td>703</td>
      <td>735</td>
      <td>122.589996</td>
      <td>134.940002</td>
      <td>-2964.001465</td>
      <td>-0.100742</td>
      <td>2021-02-26</td>
      <td>2021-04-14</td>
      <td>47 days</td>
    </tr>
    <tr>
      <th>14</th>
      <td>196</td>
      <td>735</td>
      <td>763</td>
      <td>134.940002</td>
      <td>126.010002</td>
      <td>-1750.280060</td>
      <td>-0.066178</td>
      <td>2021-04-14</td>
      <td>2021-05-24</td>
      <td>40 days</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-196</td>
      <td>763</td>
      <td>785</td>
      <td>126.010002</td>
      <td>134.449997</td>
      <td>-1654.238983</td>
      <td>-0.066979</td>
      <td>2021-05-24</td>
      <td>2021-06-24</td>
      <td>31 days</td>
    </tr>
    <tr>
      <th>16</th>
      <td>172</td>
      <td>785</td>
      <td>854</td>
      <td>134.449997</td>
      <td>141.899994</td>
      <td>1281.399475</td>
      <td>0.055411</td>
      <td>2021-06-24</td>
      <td>2021-10-01</td>
      <td>99 days</td>
    </tr>
    <tr>
      <th>17</th>
      <td>-172</td>
      <td>854</td>
      <td>877</td>
      <td>141.899994</td>
      <td>150.389999</td>
      <td>-1460.280945</td>
      <td>-0.059831</td>
      <td>2021-10-01</td>
      <td>2021-11-03</td>
      <td>33 days</td>
    </tr>
    <tr>
      <th>18</th>
      <td>152</td>
      <td>877</td>
      <td>935</td>
      <td>150.389999</td>
      <td>162.449997</td>
      <td>1833.119629</td>
      <td>0.080191</td>
      <td>2021-11-03</td>
      <td>2022-01-27</td>
      <td>85 days</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-152</td>
      <td>935</td>
      <td>957</td>
      <td>162.449997</td>
      <td>164.699997</td>
      <td>-342.000000</td>
      <td>-0.013850</td>
      <td>2022-01-27</td>
      <td>2022-03-01</td>
      <td>33 days</td>
    </tr>
    <tr>
      <th>20</th>
      <td>148</td>
      <td>957</td>
      <td>958</td>
      <td>164.699997</td>
      <td>164.389999</td>
      <td>-45.879639</td>
      <td>-0.001882</td>
      <td>2022-03-01</td>
      <td>2022-03-02</td>
      <td>1 days</td>
    </tr>
    <tr>
      <th>21</th>
      <td>-148</td>
      <td>958</td>
      <td>983</td>
      <td>164.389999</td>
      <td>172.360001</td>
      <td>-1179.560181</td>
      <td>-0.048482</td>
      <td>2022-03-02</td>
      <td>2022-04-06</td>
      <td>35 days</td>
    </tr>
    <tr>
      <th>22</th>
      <td>134</td>
      <td>983</td>
      <td>999</td>
      <td>172.360001</td>
      <td>161.839996</td>
      <td>-1409.680573</td>
      <td>-0.061035</td>
      <td>2022-04-06</td>
      <td>2022-04-29</td>
      <td>23 days</td>
    </tr>
  </tbody>
</table>



### Plotting

We can even plot our trading with bokeh plot. It is interactive just like plotly


```python
bt.plot()
```

![png]({{site.url}}/assets/backtesting/bt1.png)
![png]({{site.url}}/assets/backtesting/bt2.png)
![png]({{site.url}}/assets/backtesting/bt3.png)



### Stop Profit and Stop Loss
Profit and stop loss are often used to stay in the safe side. We exit from the trade when there is increase in price and take a profit but reversely, we exit from the trade when there is decrease in price and realize loss.


```python
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA

class SmaCross(Strategy):
    # Define the two MA lags as *class variables*
    # for later optimization
    n1 = 20
    n2 = 40
    
    def init(self):
        # Precompute the two moving averages
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
    
    def next(self):
        # If sma1 crosses above sma2, close any existing
        # short trades, and buy the asset
        if crossover(self.sma1, self.sma2):
#             self.position.close()
            self.buy(tp=self.data.Close[-1]*1.2, sl=self.data.Close[-1]*0.95)

        # Else, if sma1 crosses below sma2, close any existing
        # long trades, and sell the asset
        elif crossover(self.sma2, self.sma1):
            self.position.close()
#             self.sell()
            

from backtesting import Backtest

bt = Backtest(data.tail(last), SmaCross, cash=10000, commission=0)
stats = bt.run()
stats         
```




    Start                     2018-05-11 00:00:00
    End                       2022-04-29 00:00:00
    Duration                   1449 days 00:00:00
    Exposure Time [%]                        39.7
    Equity Final [$]                 25715.433767
    Equity Peak [$]                  26697.815822
    Return [%]                         157.154338
    Buy & Hold Return [%]              234.376153
    Return (Ann.) [%]                   26.872895
    Volatility (Ann.) [%]               19.883748
    Sharpe Ratio                         1.351501
    Sortino Ratio                        2.733988
    Calmar Ratio                         2.457561
    Max. Drawdown [%]                  -10.934783
    Avg. Drawdown [%]                   -2.628023
    Max. Drawdown Duration      163 days 00:00:00
    Avg. Drawdown Duration       20 days 00:00:00
    # Trades                                   12
    Win Rate [%]                        66.666667
    Best Trade [%]                      21.740146
    Worst Trade [%]                     -5.359055
    Avg. Trade [%]                        8.20145
    Max. Trade Duration          99 days 00:00:00
    Avg. Trade Duration          47 days 00:00:00
    Profit Factor                        8.893546
    Expectancy [%]                       8.683464
    SQN                                  2.395319
    _strategy                            SmaCross
    _equity_curve                             ...
    _trades                       Size  EntryB...
    dtype: object



In above example, we exit the trade once price increases by 20% or decreases by 5%. Doing so we made some profit as well.

## Our Own Strategy in Backtesting

Lets make our own strategy here and implement it on backtesting.
I want to do something like below:
* If EMA 9 > EMA 20 or EMA 50 > EMA 100 then buy.
* If EMA 9 < EMA 20 or EMA 50 < EMA 100 then close positions.

For calculation of EMA, we can use `pandas_ta`. We can install it like `pip install pandas-ta`.



```python
import pandas_ta as ta

from backtesting import Backtest
from backtesting import Strategy
from backtesting.lib import crossover

class EmaCross(Strategy):
    def init(self):
        self.ema9 = self.I(ta.ema, pd.Series(self.data.Close), 9)
        self.ema20 = self.I(ta.ema, pd.Series(self.data.Close), 20)
        self.ema50 = self.I(ta.ema, pd.Series(self.data.Close), 50)
        self.ema100 = self.I(ta.ema, pd.Series(self.data.Close), 100)
    
    def next(self):
        if crossover(self.ema9, self.ema20) or crossover(self.ema50, self.ema100):
            self.buy()

        elif crossover(self.ema20, self.ema9) or crossover(self.ema100, self.ema50):
            self.position.close()
            # self.sell()
    
            
bt = Backtest(data, EmaCross, cash=10000, commission=0.02)
stats = bt.run()
bt.plot()
stats
```

![png]({{site.url}}/assets/backtesting/bt4.png)
![png]({{site.url}}/assets/backtesting/bt5.png)
![png]({{site.url}}/assets/backtesting/bt6.png)






    Start                     2015-01-02 00:00:00
    End                       2022-04-29 00:00:00
    Duration                   2674 days 00:00:00
    Exposure Time [%]                   63.631436
    Equity Final [$]                 24777.296183
    Equity Peak [$]                  30443.052752
    Return [%]                         147.772962
    Buy & Hold Return [%]              476.785846
    Return (Ann.) [%]                   13.193633
    Volatility (Ann.) [%]                21.81465
    Sharpe Ratio                         0.604806
    Sortino Ratio                         1.02413
    Calmar Ratio                         0.463525
    Max. Drawdown [%]                  -28.463721
    Avg. Drawdown [%]                   -4.544362
    Max. Drawdown Duration      776 days 00:00:00
    Avg. Drawdown Duration       53 days 00:00:00
    # Trades                                   30
    Win Rate [%]                        46.666667
    Best Trade [%]                      60.672271
    Worst Trade [%]                      -7.85946
    Avg. Trade [%]                       3.077886
    Max. Trade Duration         190 days 00:00:00
    Avg. Trade Duration          56 days 00:00:00
    Profit Factor                        2.584764
    Expectancy [%]                       3.953845
    SQN                                  1.328035
    _strategy                            EmaCross
    _equity_curve                             ...
    _trades                       Size  EntryB...
    dtype: object



Looks like we made some money. But this is just another bad strategy we tested.

### Testing Percentage Price Oscillator
Following is taken from my another [blog](https://q-viper.github.io/2022/03/20/python-for-stock-market-analysis-technical-indicators/). 
* This is a momentum indicator (determines the strength or weakness of a value). But we can view the volatility too.
* Two EMAs, 26 period and 12 periods are used to calculate PPO.
* It contains 2 lines, PPO line and signal line. Signal line is an EMA of the 9 Period PPO, so it moves slower than PPO.
* When PPO line crosses the signal line, it is the time for rise/fall of the price or stock.
* When PPO line crosses over the signal line from below, then it is a buy signal. Reversely, it is a sell signal when PPO line crosses belo the signal line from above.
* When PPO line is below the 0, the short term average is below the longer-term average average, which helps indicate a fall of price.
* Conversely, when PPO line is above 0, the short term average is above the long term average, which helps indicate rise of price.

`pandas_ta` has PPO too so we do not have to write our own code for it.



```python
ta.ppo(data.Close)
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PPO_12_26_9</th>
      <th>PPOh_12_26_9</th>
      <th>PPOs_12_26_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-02</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-01-05</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-01-06</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-01-07</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-01-08</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2022-04-25</th>
      <td>-1.987252</td>
      <td>-2.241286</td>
      <td>0.254034</td>
    </tr>
    <tr>
      <th>2022-04-26</th>
      <td>-2.580172</td>
      <td>-2.267365</td>
      <td>-0.312807</td>
    </tr>
    <tr>
      <th>2022-04-27</th>
      <td>-3.049812</td>
      <td>-2.189604</td>
      <td>-0.860208</td>
    </tr>
    <tr>
      <th>2022-04-28</th>
      <td>-3.039588</td>
      <td>-1.743504</td>
      <td>-1.296084</td>
    </tr>
    <tr>
      <th>2022-04-29</th>
      <td>-3.256113</td>
      <td>-1.568023</td>
      <td>-1.688090</td>
    </tr>
  </tbody>
</table>
<p>1845 rows × 3 columns</p>





```python
class PPO(Strategy):
    def init(self):
        self.ppo = self.I(ta.ppo, pd.Series(self.data.Close))
        
    def next(self):
        if crossover(self.ppo[0], self.ppo[2]):
        # if crossover(self.ppo[1], 0):
            # self.position.close()
            self.buy()

        elif crossover(self.ppo[2], self.ppo[0]):
        #if crossover(0,self.ppo[1]):
            self.position.close()
            # self.sell()
            
bt = Backtest(data, PPO, cash=10000, commission=0.02)
stats = bt.run()
bt.plot()
print(stats)


```

![png]({{site.url}}/assets/backtesting/bt7.png)
![png]({{site.url}}/assets/backtesting/bt8.png)
![png]({{site.url}}/assets/backtesting/bt9.png)
![png]({{site.url}}/assets/backtesting/bt10.png)




    Start                     2015-01-02 00:00:00
    End                       2022-04-29 00:00:00
    Duration                   2674 days 00:00:00
    Exposure Time [%]                    51.00271
    Equity Final [$]                 12163.349445
    Equity Peak [$]                  14453.298656
    Return [%]                          21.633494
    Buy & Hold Return [%]              476.785846
    Return (Ann.) [%]                    2.711015
    Volatility (Ann.) [%]                18.61385
    Sharpe Ratio                         0.145645
    Sortino Ratio                         0.22052
    Calmar Ratio                         0.067051
    Max. Drawdown [%]                  -40.432272
    Avg. Drawdown [%]                   -7.414292
    Max. Drawdown Duration     1942 days 00:00:00
    Avg. Drawdown Duration      169 days 00:00:00
    # Trades                                   56
    Win Rate [%]                        44.642857
    Best Trade [%]                      17.890366
    Worst Trade [%]                    -15.387924
    Avg. Trade [%]                       0.353242
    Max. Trade Duration          70 days 00:00:00
    Avg. Trade Duration          23 days 00:00:00
    Profit Factor                        1.242225
    Expectancy [%]                       0.592473
    SQN                                  0.433939
    _strategy                                 PPO
    _equity_curve                             ...
    _trades                       Size  EntryB...
    dtype: object
    

Looks like we again made some money. There is not a golden rule that will make a money, its kind of hit and trial.

### PPO on BABA


```python

bdata = yf.download("BABA", start="2015-01-01", end="2022-11-30")
del bdata['Adj Close']
del bdata['Volume']

bt = Backtest(bdata, PPO, cash=10000, commission=0.02)
stats = bt.run()
bt.plot()
print(stats)

```

    [*********************100%***********************]  1 of 1 completed
    

![png]({{site.url}}/assets/backtesting/bt11.png)
![png]({{site.url}}/assets/backtesting/bt12.png)
![png]({{site.url}}/assets/backtesting/bt13.png)
![png]({{site.url}}/assets/backtesting/bt14.png)








    Start                     2015-01-02 00:00:00
    End                       2022-11-18 00:00:00
    Duration                   2877 days 00:00:00
    Exposure Time [%]                   52.265861
    Equity Final [$]                  1485.461341
    Equity Peak [$]                  11330.885306
    Return [%]                         -85.145387
    Buy & Hold Return [%]              -22.316598
    Return (Ann.) [%]                  -21.491087
    Volatility (Ann.) [%]               21.490062
    Sharpe Ratio                              0.0
    Sortino Ratio                             0.0
    Calmar Ratio                              0.0
    Max. Drawdown [%]                  -88.917088
    Avg. Drawdown [%]                  -28.010671
    Max. Drawdown Duration     2571 days 00:00:00
    Avg. Drawdown Duration      705 days 00:00:00
    # Trades                                   71
    Win Rate [%]                        26.760563
    Best Trade [%]                      26.501533
    Worst Trade [%]                    -22.092189
    Avg. Trade [%]                       -2.73311
    Max. Trade Duration          56 days 00:00:00
    Avg. Trade Duration          20 days 00:00:00
    Profit Factor                        0.496783
    Expectancy [%]                      -2.346556
    SQN                                 -2.054515
    _strategy                                 PPO
    _equity_curve                             ...
    _trades                       Size  EntryB...
    dtype: object
    

In first PPO strategy, we tested with AAPL and in second we tested with BABA. In BABA, we lost money but in AAPL we made some.

There are many features and strategy to try on stock backtesting using Backtesting.py and those will be covered in next part. Thank you :)
