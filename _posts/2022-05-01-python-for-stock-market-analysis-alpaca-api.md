---
title:  "Python for Stock Market Analysis: Alpaca API"
date:   2022-05-01 09:29:17 +0545
categories:
    - Data Analysis
    - Alpaca API
    - Stock Market Analysis
    
tags:
    - data analysis
    - alpaca api
    - stock market
    - nasdaq
    - nyse
---

## Introduction
[Alpaca Trading API](https://github.com/alpacahq/alpaca-trade-api-python) is an API using which we can retrieve stock data in realtime. It provides various APIs and even streaming services. Please read about it in their [docs](https://alpaca.markets/docs/trading/). What is most exciting about this API and the librariy is that it returns the data as a Pandas Dataframe or even simple Dict object. And here we will focus on how to do real time stock market data retrieval. 



## Getting API Keys
* To access API, we need to have an account. So lets signup in [https://app.alpaca.markets/signup](https://app.alpaca.markets/signup). Then fill up the form and go to the home page.

![png]({{site.url}}/assets/alpaca/app.png)

* In order to make first ever API request, we need to have API keys and secret. So go to [papertrading and signup for keys](https://app.alpaca.markets/paper/dashboard/overview). **It requires real identity card and live camera at some point**.
* Save your API Keys and Secret somewhere safe.

![png]({{site.url}}/assets/alpaca/api_keys.png)



### Making First Request
* Install the library first using `pip3 install alpaca-trade-api`. Or your own package manager in Python.



```python
!pip3 install alpaca-trade-api
```


```python
import pandas as pd

key="PK5OLX1GAB5TYELHCV1J"
secret="LFdvHRFf93NydXF2kDAskTKDjHtBPyBHnszciU4G"

```

Please insert your generated keys in above variable.


```python

```


```python
import alpaca_trade_api as tradeapi

# to which url we will send request
base_url="https://paper-api.alpaca.markets"

# instantiate REST API
api = tradeapi.REST(key, secret, base_url, api_version='v2')

# obtain account information
account = api.get_account()
print(account)
```

    Account({   'account_blocked': False,
        'account_number': 'PA3BQI9QBTE3',
        'accrued_fees': '0',
        'buying_power': '200000',
        'cash': '100000',
        'created_at': '2022-04-30T14:29:12.722985Z',
        'crypto_status': 'ACTIVE',
        'currency': 'USD',
        'daytrade_count': 0,
        'daytrading_buying_power': '0',
        'equity': '100000',
        'id': 'ddeb8bc4-87d3-4ade-8b4a-8a40c7207fed',
        'initial_margin': '0',
        'last_equity': '100000',
        'last_maintenance_margin': '0',
        'long_market_value': '0',
        'maintenance_margin': '0',
        'multiplier': '2',
        'non_marginable_buying_power': '100000',
        'pattern_day_trader': False,
        'pending_transfer_in': '0',
        'portfolio_value': '100000',
        'regt_buying_power': '200000',
        'short_market_value': '0',
        'shorting_enabled': True,
        'sma': '0',
        'status': 'ACTIVE',
        'trade_suspended_by_user': False,
        'trading_blocked': False,
        'transfers_blocked': False})
    

It shows that the fresh new account that I just created. 

## Getting List of Symbols
We can get the list of assets like below.


```python

active_assets=api.list_assets(status='active')
for asset in active_assets:
  print(asset)
  break
```

    Asset({   'class': 'us_equity',
        'easy_to_borrow': False,
        'exchange': 'OTC',
        'fractionable': False,
        'id': 'f377f9ef-4b3b-425d-890d-dc7698edc623',
        'marginable': False,
        'name': 'GWG HLDGS INC Common Stock',
        'shortable': False,
        'status': 'active',
        'symbol': 'GWGHQ',
        'tradable': False})
    

## Rest APIs
Alpaca provides various REST APIs that handles the most common requests like getting trade, quote, floorsheet.

### Get Floorsheet Data

Now is the time to make first request to get data. So lets define a list of stock symbols that we will be focusing on. We will focus on Tesla and Apple stocks. And we want to see floorsheet value of each minute.

In terms of Alpaca, its bars.

#### All At one time
We can get all data at one time or iterate over it. Getting all data is little bit slow and might require extra memory.


```python
symbols = ["TSLA", "AAPL"]
resp = api.get_bars(symbols,timeframe='1Min', start="2022-04-30",limit=10000)
resp.df.reset_index()
```





  
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
Today is Friday and hence there will not be any data at the moment. Thus, we will need to change start date to yesterday.


```python
symbols = ["TSLA", "AAPL"]
resp = api.get_bars(symbols,timeframe='1Min', start="2022-04-29",limit=None)
resp.df.reset_index()
```





  
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
      <th>trade_count</th>
      <th>vwap</th>
      <th>symbol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-04-29 08:00:00+00:00</td>
      <td>161.00</td>
      <td>161.50</td>
      <td>160.36</td>
      <td>160.36</td>
      <td>6107</td>
      <td>116</td>
      <td>160.736835</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-04-29 08:01:00+00:00</td>
      <td>160.74</td>
      <td>160.84</td>
      <td>160.70</td>
      <td>160.77</td>
      <td>3929</td>
      <td>87</td>
      <td>160.773235</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-04-29 08:02:00+00:00</td>
      <td>160.84</td>
      <td>160.84</td>
      <td>160.77</td>
      <td>160.77</td>
      <td>2929</td>
      <td>69</td>
      <td>160.803947</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-04-29 08:03:00+00:00</td>
      <td>160.75</td>
      <td>160.77</td>
      <td>160.70</td>
      <td>160.77</td>
      <td>3335</td>
      <td>71</td>
      <td>160.734747</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-04-29 08:04:00+00:00</td>
      <td>160.78</td>
      <td>160.98</td>
      <td>160.77</td>
      <td>160.98</td>
      <td>1496</td>
      <td>42</td>
      <td>160.826170</td>
      <td>AAPL</td>
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
    </tr>
    <tr>
      <th>1726</th>
      <td>2022-04-29 23:55:00+00:00</td>
      <td>876.68</td>
      <td>876.68</td>
      <td>876.68</td>
      <td>876.68</td>
      <td>432</td>
      <td>36</td>
      <td>876.634097</td>
      <td>TSLA</td>
    </tr>
    <tr>
      <th>1727</th>
      <td>2022-04-29 23:56:00+00:00</td>
      <td>876.64</td>
      <td>877.00</td>
      <td>876.64</td>
      <td>877.00</td>
      <td>3178</td>
      <td>65</td>
      <td>876.884078</td>
      <td>TSLA</td>
    </tr>
    <tr>
      <th>1728</th>
      <td>2022-04-29 23:57:00+00:00</td>
      <td>877.00</td>
      <td>877.00</td>
      <td>876.80</td>
      <td>877.00</td>
      <td>1599</td>
      <td>87</td>
      <td>877.110913</td>
      <td>TSLA</td>
    </tr>
    <tr>
      <th>1729</th>
      <td>2022-04-29 23:58:00+00:00</td>
      <td>878.00</td>
      <td>878.02</td>
      <td>877.58</td>
      <td>877.58</td>
      <td>1431</td>
      <td>100</td>
      <td>877.933099</td>
      <td>TSLA</td>
    </tr>
    <tr>
      <th>1730</th>
      <td>2022-04-29 23:59:00+00:00</td>
      <td>877.88</td>
      <td>877.88</td>
      <td>877.50</td>
      <td>877.50</td>
      <td>1299</td>
      <td>90</td>
      <td>877.808984</td>
      <td>TSLA</td>
    </tr>
  </tbody>
</table>
<p>1731 rows × 9 columns</p>


Now looking over the dataframe above, we can see that there are 1731 rows with floorsheet value of each minute for each stock. 

##### Custom Timeframe

We can even get floorsheet data of every 45, 10, 30 minute or even more customized using `TimeFrame` and `TimeFrameUnit` given by its package.



```python
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit

df = api.get_bars(symbols, TimeFrame(13, TimeFrameUnit.Minute), "2022-04-29").df.reset_index()
df
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
      <th>trade_count</th>
      <th>vwap</th>
      <th>symbol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-04-29 07:57:00+00:00</td>
      <td>161.00</td>
      <td>161.50</td>
      <td>160.36</td>
      <td>160.62</td>
      <td>46766</td>
      <td>847</td>
      <td>160.805158</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-04-29 08:10:00+00:00</td>
      <td>160.63</td>
      <td>160.78</td>
      <td>160.52</td>
      <td>160.56</td>
      <td>53331</td>
      <td>755</td>
      <td>160.655755</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-04-29 08:23:00+00:00</td>
      <td>160.56</td>
      <td>160.63</td>
      <td>159.63</td>
      <td>160.00</td>
      <td>72334</td>
      <td>1059</td>
      <td>160.035541</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-04-29 08:36:00+00:00</td>
      <td>159.92</td>
      <td>160.00</td>
      <td>159.85</td>
      <td>160.00</td>
      <td>15276</td>
      <td>349</td>
      <td>159.939272</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-04-29 08:49:00+00:00</td>
      <td>159.96</td>
      <td>160.00</td>
      <td>159.74</td>
      <td>159.80</td>
      <td>14169</td>
      <td>347</td>
      <td>159.896077</td>
      <td>AAPL</td>
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
    </tr>
    <tr>
      <th>145</th>
      <td>2022-04-29 23:07:00+00:00</td>
      <td>877.26</td>
      <td>877.71</td>
      <td>876.09</td>
      <td>876.24</td>
      <td>6727</td>
      <td>335</td>
      <td>877.222977</td>
      <td>TSLA</td>
    </tr>
    <tr>
      <th>146</th>
      <td>2022-04-29 23:20:00+00:00</td>
      <td>875.20</td>
      <td>875.60</td>
      <td>875.01</td>
      <td>875.60</td>
      <td>2143</td>
      <td>187</td>
      <td>875.371510</td>
      <td>TSLA</td>
    </tr>
    <tr>
      <th>147</th>
      <td>2022-04-29 23:33:00+00:00</td>
      <td>875.57</td>
      <td>876.50</td>
      <td>875.57</td>
      <td>876.50</td>
      <td>9723</td>
      <td>443</td>
      <td>876.288069</td>
      <td>TSLA</td>
    </tr>
    <tr>
      <th>148</th>
      <td>2022-04-29 23:46:00+00:00</td>
      <td>876.50</td>
      <td>878.02</td>
      <td>876.49</td>
      <td>877.58</td>
      <td>11632</td>
      <td>507</td>
      <td>876.892889</td>
      <td>TSLA</td>
    </tr>
    <tr>
      <th>149</th>
      <td>2022-04-29 23:59:00+00:00</td>
      <td>877.88</td>
      <td>877.88</td>
      <td>877.50</td>
      <td>877.50</td>
      <td>1299</td>
      <td>90</td>
      <td>877.808984</td>
      <td>TSLA</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 9 columns</p>

In above example, we are looking for floorsheet of every 13 minutes. We can even use Hour there.

#### Iterating Over
Instead of getting entire data, it might be a good idea to loop through if we have many rows. Thus we can get a iterable and iterate over it like below.


```python
fls = api.get_bars_iter(symbols, TimeFrame(10, TimeFrameUnit.Hour), "2022-04-29")
for fs in fls:
  print(fs)
```

    Bar({   'S': 'AAPL',
        'c': 160.01,
        'h': 166.2,
        'l': 158.97,
        'n': 699101,
        'o': 161,
        't': '2022-04-29T08:00:00Z',
        'v': 75360694,
        'vw': 162.450706})
    Bar({   'S': 'AAPL',
        'c': 157.93,
        'h': 160.39,
        'l': 157.25,
        'n': 323819,
        'o': 160,
        't': '2022-04-29T18:00:00Z',
        'v': 56041611,
        'vw': 158.373787})
    Bar({   'S': 'TSLA',
        'c': 891.86,
        'h': 934.3999,
        'l': 884.46,
        'n': 651589,
        'o': 899,
        't': '2022-04-29T08:00:00Z',
        'v': 20826462,
        'vw': 910.364386})
    Bar({   'S': 'TSLA',
        'c': 877.5,
        'h': 892.81,
        'l': 868,
        'n': 222398,
        'o': 891.88,
        't': '2022-04-29T18:00:00Z',
        'v': 8468767,
        'vw': 878.750304})
    

### Getting Quotes
Quotes are the biddings that has been done.



```python
df = api.get_quotes(symbols, start="2022-04-29",limit=1000).df.reset_index()
df
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>ask_exchange</th>
      <th>ask_price</th>
      <th>ask_size</th>
      <th>bid_exchange</th>
      <th>bid_price</th>
      <th>bid_size</th>
      <th>conditions</th>
      <th>tape</th>
      <th>symbol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-04-29 08:00:00.001169445+00:00</td>
      <td></td>
      <td>0.00</td>
      <td>0</td>
      <td>Q</td>
      <td>160.66</td>
      <td>4</td>
      <td>[Y]</td>
      <td>C</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-04-29 08:00:00.001241377+00:00</td>
      <td>Q</td>
      <td>160.93</td>
      <td>4</td>
      <td>Q</td>
      <td>160.66</td>
      <td>4</td>
      <td>[R]</td>
      <td>C</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-04-29 08:00:00.005655+00:00</td>
      <td>Q</td>
      <td>160.93</td>
      <td>4</td>
      <td>K</td>
      <td>161.50</td>
      <td>1</td>
      <td>[R]</td>
      <td>C</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-04-29 08:00:00.005655+00:00</td>
      <td>Q</td>
      <td>160.93</td>
      <td>4</td>
      <td>Q</td>
      <td>160.66</td>
      <td>4</td>
      <td>[R]</td>
      <td>C</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-04-29 08:00:00.005655+00:00</td>
      <td>Q</td>
      <td>160.93</td>
      <td>4</td>
      <td>K</td>
      <td>161.00</td>
      <td>1</td>
      <td>[R]</td>
      <td>C</td>
      <td>AAPL</td>
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
    </tr>
    <tr>
      <th>995</th>
      <td>2022-04-29 08:22:05.119679488+00:00</td>
      <td>P</td>
      <td>160.55</td>
      <td>2</td>
      <td>Q</td>
      <td>160.52</td>
      <td>1</td>
      <td>[R]</td>
      <td>C</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>996</th>
      <td>2022-04-29 08:22:05.119927355+00:00</td>
      <td>Q</td>
      <td>160.59</td>
      <td>6</td>
      <td>Q</td>
      <td>160.55</td>
      <td>1</td>
      <td>[R]</td>
      <td>C</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>997</th>
      <td>2022-04-29 08:22:05.119937180+00:00</td>
      <td>Q</td>
      <td>160.59</td>
      <td>6</td>
      <td>P</td>
      <td>160.52</td>
      <td>1</td>
      <td>[R]</td>
      <td>C</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>998</th>
      <td>2022-04-29 08:22:05.120335360+00:00</td>
      <td>P</td>
      <td>160.55</td>
      <td>1</td>
      <td>Q</td>
      <td>160.52</td>
      <td>1</td>
      <td>[R]</td>
      <td>C</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>999</th>
      <td>2022-04-29 08:22:05.120487168+00:00</td>
      <td>P</td>
      <td>160.55</td>
      <td>3</td>
      <td>Q</td>
      <td>160.52</td>
      <td>1</td>
      <td>[R]</td>
      <td>C</td>
      <td>AAPL</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 10 columns</p>


There are various fileds and there is a well described documentation in [docs](https://alpaca.markets/deprecated/docs/api-documentation/api-v2/market-data/alpaca-data-api-v2/historical/).

### Getting Trade
To view the trades that has actually happened.


```python
df = api.get_trades(symbols, start="2022-04-29",limit=1000).df.reset_index()
df
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>exchange</th>
      <th>price</th>
      <th>size</th>
      <th>conditions</th>
      <th>id</th>
      <th>tape</th>
      <th>symbol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-04-29 04:00:13.553000+00:00</td>
      <td>D</td>
      <td>160.09</td>
      <td>1</td>
      <td>[@, T, I]</td>
      <td>927</td>
      <td>C</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-04-29 04:00:43.779000+00:00</td>
      <td>D</td>
      <td>160.09</td>
      <td>220</td>
      <td>[@, T]</td>
      <td>871</td>
      <td>C</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-04-29 04:02:52.730000+00:00</td>
      <td>D</td>
      <td>160.10</td>
      <td>1</td>
      <td>[@, T, I]</td>
      <td>912</td>
      <td>C</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-04-29 04:02:52.730000+00:00</td>
      <td>D</td>
      <td>160.10</td>
      <td>4</td>
      <td>[@, T, I]</td>
      <td>940</td>
      <td>C</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-04-29 04:06:04.276000+00:00</td>
      <td>D</td>
      <td>160.10</td>
      <td>20</td>
      <td>[@, T, I]</td>
      <td>941</td>
      <td>C</td>
      <td>AAPL</td>
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
      <th>995</th>
      <td>2022-04-29 08:09:30.317838547+00:00</td>
      <td>Q</td>
      <td>160.63</td>
      <td>1</td>
      <td>[@, F, T, I]</td>
      <td>430</td>
      <td>C</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>996</th>
      <td>2022-04-29 08:09:30.617365760+00:00</td>
      <td>P</td>
      <td>160.61</td>
      <td>137</td>
      <td>[@, F, T]</td>
      <td>258</td>
      <td>C</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>997</th>
      <td>2022-04-29 08:09:30.617365760+00:00</td>
      <td>P</td>
      <td>160.60</td>
      <td>74</td>
      <td>[@, F, T, I]</td>
      <td>259</td>
      <td>C</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>998</th>
      <td>2022-04-29 08:09:30.756697344+00:00</td>
      <td>P</td>
      <td>160.60</td>
      <td>2</td>
      <td>[@, T, I]</td>
      <td>260</td>
      <td>C</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>999</th>
      <td>2022-04-29 08:09:32.422441+00:00</td>
      <td>K</td>
      <td>160.60</td>
      <td>10</td>
      <td>[@, F, T, I]</td>
      <td>129</td>
      <td>C</td>
      <td>AAPL</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 8 columns</p>

Alpaca provides other Rest APIs too which can be seen in their module https://github.com/alpacahq/alpaca-trade-api-python/blob/master/alpaca_trade_api/rest.py. Most of the REST APIs have iterable functions too like `get_bars_iter`, `get_quote_iter`.

## Streaming Data
Streaming Data is done to get realtime data and Alpaca provides websockets to do that. To do so, we will use `async` keyword in front of the function that will be awaitable. Below code is taken from the GitHub repo of [Alpaca package](https://github.com/alpacahq/alpaca-trade-api-python).



```python
from alpaca_trade_api.stream import Stream
from alpaca_trade_api.common import URL

async def trade_callback(t):
    print('trade', t)


async def quote_callback(q):
    print('quote', q)


# Initiate Class Instance
stream = Stream(key,
                secret,
                base_url=URL('https://paper-api.alpaca.markets'),
                data_feed='iex') 

# subscribing to event
stream.subscribe_trades(trade_callback, 'AAPL')
stream.subscribe_quotes(quote_callback, 'IBM')

stream.run()
```

There are some other streamming APIs also like `subscribe_bars` which can be seen inside a module [stream.py](https://github.com/alpacahq/alpaca-trade-api-python/blob/master/alpaca_trade_api/stream.py).

### How to stream bars data?
By default there is not a possibility to stream a bars data but one can do it by using `asyncio` in Python.

This section will be updated soon. Sorry.
