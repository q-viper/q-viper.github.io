---
title:  "Poygon.io for Stock Market Data"
date:   2022-07-10 09:29:17 +0545
categories:
    - Stock
    - Polygon
tags:
    - Websocket
    - Rest
header:
  teaser: assets/polygon/dashboard.png
---

## Introduction
Hello everyone, welcome back to our new blog about getting Stock data in realtime using Polygon.io. Few blogs ago, I've shared how can we use Alpaca API to stream Stock data. But in this blog, we will use Polygon.io and choosing Polyon over Alpaca has its own pros and cons.
* Alpaca is little bit better than Polygon in terms of the documentation in GitHub and the APIs. Which can be seen in [Alpaca Trade API](https://github.com/alpacahq/alpaca-trade-api-python) and [Polygon IO Client Python](https://github.com/polygon-io/client-python).
* Alpaca could give data in dataframe as well as JSON format but Polygon gives only in JSON. However we could make dataframe from JSON as well.
* The Updated candles in Alpaca were arriving little slower than Polygon. It was found that at least 3sec is taken from the Polygon to send corrected bars whereas Alpaca was taking more than 30secs.

So, to choose between Alpaca and Polygon, one should focus if the 30 seconds delay in corrected data is acceptable or not. If it is not then Polygon is best choice else Alpaca wins the race for me as it provides some great modules like `get_clock()`.

## Getting User and Key: Polygon.io
Its easy to get API Key. Just sign up for free version to see the dashboard and then the key will be there somewhere.


```python
key="yMU4qf8_bZIgInGyztizQg5vPjkJr6EY"
```

## Installing Polygon API Client
I am using version 0.2.11 because this version was working as my requirements was for the project.


```python
!pip install polygon-api-client==0.2.11
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting polygon-api-client==0.2.11
      Downloading polygon_api_client-0.2.11-py3-none-any.whl (22 kB)
    Collecting websocket-client>=0.56.0
      Downloading websocket_client-1.3.3-py3-none-any.whl (54 kB)
    [K     |████████████████████████████████| 54 kB 2.2 MB/s 
    [?25hRequirement already satisfied: requests>=2.22.0 in /usr/local/lib/python3.7/dist-packages (from polygon-api-client==0.2.11) (2.23.0)
    Collecting websockets>=8.0.2
      Downloading websockets-10.3-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (112 kB)
    [K     |████████████████████████████████| 112 kB 12.1 MB/s 
    [?25hRequirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests>=2.22.0->polygon-api-client==0.2.11) (1.24.3)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests>=2.22.0->polygon-api-client==0.2.11) (2.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests>=2.22.0->polygon-api-client==0.2.11) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests>=2.22.0->polygon-api-client==0.2.11) (2022.6.15)
    Installing collected packages: websockets, websocket-client, polygon-api-client
    Successfully installed polygon-api-client-0.2.11 websocket-client-1.3.3 websockets-10.3
    

## Rest API to Get Aggregate Data
Aggregate Data means the data that is aggregated with time. There could be OHLC in every minute and in any time frame. Open is always a Opening price of first candle in the timeframe, close is always a closing price of last candle in a timeframe and high is high price among all candles in timeframe and low is low price among all candles in timeframe.


```python
from polygon import RESTClient
import pandas as pd

client = RESTClient(key)

```

Polygon uses timestamp to mention datetime FROM and TO. So we need to get timestamp first. And its also worth mentioning that the timestamp should be using milliseconds. We got the timestamp below but its not upto milliseconds so we added 3 0s in the end of bothn while calling api.


```python
int(pd.to_datetime("2022-06-10 01:22").timestamp()),int(pd.to_datetime("2022-06-21 06:22").timestamp())
```




    (1654824120, 1655792520)



The data we are looking for is 1minute candle and it is available using `stocks_equities_aggregates` in this version. More about this function can be found in documentation [here](https://polygon.io/docs/stocks/get_v2_aggs_ticker__stocksticker__range__multiplier___timespan___from___to).


```python
res=client.stocks_equities_aggregates(ticker='AAPL', multiplier=1, 
                                      timespan="minute", from_="1654824120000", 
                                      to="1655792520000",limit=500000)
```

The result will be in JSON but we can use pandas to make it dataframe.


```python
res.results
```


```python

```


```python
df = pd.DataFrame(res.results)

df
```





  <div id="df-2a46dcf9-8f0a-447a-bb0e-e89ad781d304">
    <div class="colab-df-container">
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
      <th>v</th>
      <th>vw</th>
      <th>o</th>
      <th>c</th>
      <th>h</th>
      <th>l</th>
      <th>t</th>
      <th>n</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2292.0</td>
      <td>142.9749</td>
      <td>143.03</td>
      <td>142.99</td>
      <td>143.03</td>
      <td>142.90</td>
      <td>1654848000000</td>
      <td>64</td>
    </tr>
    <tr>
      <th>1</th>
      <td>817.0</td>
      <td>143.0116</td>
      <td>143.02</td>
      <td>143.03</td>
      <td>143.03</td>
      <td>143.02</td>
      <td>1654848060000</td>
      <td>53</td>
    </tr>
    <tr>
      <th>2</th>
      <td>513.0</td>
      <td>143.0704</td>
      <td>143.10</td>
      <td>143.10</td>
      <td>143.10</td>
      <td>143.10</td>
      <td>1654848120000</td>
      <td>34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>940.0</td>
      <td>143.2342</td>
      <td>143.23</td>
      <td>143.25</td>
      <td>143.25</td>
      <td>143.23</td>
      <td>1654848240000</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1802.0</td>
      <td>143.1876</td>
      <td>143.20</td>
      <td>143.15</td>
      <td>143.20</td>
      <td>143.15</td>
      <td>1654848300000</td>
      <td>58</td>
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
      <th>5033</th>
      <td>536.0</td>
      <td>131.5226</td>
      <td>131.52</td>
      <td>131.52</td>
      <td>131.52</td>
      <td>131.52</td>
      <td>1655509860000</td>
      <td>18</td>
    </tr>
    <tr>
      <th>5034</th>
      <td>706.0</td>
      <td>131.5344</td>
      <td>131.52</td>
      <td>131.55</td>
      <td>131.55</td>
      <td>131.52</td>
      <td>1655509920000</td>
      <td>11</td>
    </tr>
    <tr>
      <th>5035</th>
      <td>2014.0</td>
      <td>131.5570</td>
      <td>131.55</td>
      <td>131.57</td>
      <td>131.57</td>
      <td>131.55</td>
      <td>1655510040000</td>
      <td>27</td>
    </tr>
    <tr>
      <th>5036</th>
      <td>647.0</td>
      <td>131.5287</td>
      <td>131.53</td>
      <td>131.52</td>
      <td>131.53</td>
      <td>131.52</td>
      <td>1655510220000</td>
      <td>21</td>
    </tr>
    <tr>
      <th>5037</th>
      <td>902.0</td>
      <td>131.5651</td>
      <td>131.55</td>
      <td>131.56</td>
      <td>131.56</td>
      <td>131.55</td>
      <td>1655510340000</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
<p>5038 rows × 8 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-2a46dcf9-8f0a-447a-bb0e-e89ad781d304')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-2a46dcf9-8f0a-447a-bb0e-e89ad781d304 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-2a46dcf9-8f0a-447a-bb0e-e89ad781d304');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Columns in above table are using initials, o for open, h for high and so on.

## Web Socket to Get Realtime Data

Polygon also provides WebSocket which allows us to get data in near realtime.


```python
from polygon import WebSocketClient, STOCKS_CLUSTER,RESTClient
import json
```

We need to create a handler inorder to handle a response. In our case, we need to create a handler that will write the data in our database or our desired place.


```python
def close_handler(ws):
    ws=json.loads(ws)
    for w in ws:
        if w["ev"]=="AM":
          print(w)
```

In above function, we are receiving websocket's response as ws and we convert it into dictionary using `json.loads`. Polygon sends bunch of responses in a same response if the system is slow or the result is too many to send one by one. So we loop through them, if there is a event (ev) named `AM` then we pring our data. AM means aggregated minute (I guess).


```python
symbols=["AAPL"]
my_client = WebSocketClient(STOCKS_CLUSTER, key, close_handler)
my_client.run_async()
my_client.subscribe(*[f"AM.{s}" for s in symbols])
```

* We prepare symbols in a list. 
* Then prepare a object of `WebSocketClient` by passing STOCKS_CLUSTER, key and our handler. 
* We run a Async and finally subscribe to the symbols. The `AM` there is responsible for getting realtime Aggregated data per minute.


