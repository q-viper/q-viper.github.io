---
layout: single
title: "Polygon.io Stock Market Data in Python: REST Aggregates, WebSockets, pandas, and Realtime Streams"
date: 2022-07-10 09:29:17 +0545
last_modified_at: 2026-06-16
categories:
  - Stock
  - Polygon
  - Python
tags:
  - "Polygon.io"
  - "Massive"
  - "Stock Market Data"
  - "Python"
  - "REST API"
  - "WebSocket"
  - "Realtime Data"
  - "pandas"
  - "OHLC"
  - "Aggregates"
description: "A beginner-friendly guide to using Polygon.io stock market data in Python, including API keys, REST aggregate bars, pandas DataFrames, WebSocket streams, and realtime minute candles."
excerpt: "Learn how to use Polygon.io stock market data in Python with REST APIs, aggregate OHLC bars, pandas DataFrames, and WebSocket realtime streams."
header:
  teaser: assets/polygon/dashboard.png
  og_image: assets/polygon/dashboard.png
  image_description: "Polygon.io dashboard screenshot for stock market data API tutorial"
toc: true
toc_label: "Polygon.io Stock Data"
toc_icon: "chart-line"
toc_sticky: true
read_time: true
share: true
related: true
classes: wide
---

In this tutorial, we will learn how to get **stock market data in Python using Polygon.io**. We will look at both REST API data and WebSocket streaming data.

The original version of this blog was written in 2022. At that time, I was comparing Alpaca and Polygon for realtime stock data. Polygon was useful because the updated minute bars were arriving faster in my experiments.

Today, the ecosystem has changed. Polygon.io has rebranded as **Massive**, but many developers still know the platform as Polygon.io. Existing Polygon-style API concepts are still important, so this tutorial keeps the original name while also mentioning the newer client library direction.

> This post is for educational use only. It is not financial advice and should not be used as the only basis for trading or investing decisions.

## What This Tutorial Covers

We will cover:

- what Polygon.io is used for
- REST API vs WebSocket API
- how to keep API keys safe
- installing the Python client
- getting aggregate OHLC bars
- converting JSON results to a pandas DataFrame
- cleaning timestamp columns
- plotting stock candles
- using WebSocket for realtime minute aggregates
- common errors and fixes
- when to use REST and when to use WebSocket

## Polygon.io vs Alpaca

A few blogs before the original post, I wrote about using Alpaca API to stream stock data. Polygon.io and Alpaca both provide market data, but they are useful in slightly different ways.

In my earlier experiment:

- Alpaca had very convenient Python tools.
- Alpaca could return data in friendly formats.
- Polygon returned JSON, but it was easy to convert to pandas.
- Polygon's corrected aggregate bars arrived faster in my use case.
- Alpaca had useful trading-related helpers such as market clock functions.

So, the choice depends on what you need.

Use Polygon.io or Massive-style APIs when:

- you mostly need market data
- you need REST and WebSocket data streams
- you want aggregates, trades, quotes, snapshots, or reference data
- you are building dashboards, research tools, or data pipelines

Use Alpaca when:

- you also need trading APIs
- you want broker-style account features
- you want paper trading and market-data features together

## Current Note: Polygon.io Rebrand

Polygon.io has rebranded as Massive. If you are updating older code, you may see two styles:

Old style:

```bash
pip install polygon-api-client
```

Newer style:

```bash
pip install -U massive
```

Old imports may look like this:

```python
from polygon import RESTClient
```

Newer imports may look like this:

```python
from massive import RESTClient
```

If you are maintaining an older project, check which package version the project uses before changing imports.

## Getting an API Key

To use Polygon.io or Massive market data APIs, you need an API key.

General steps:

1. create an account
2. open the dashboard
3. find your API key
4. copy it into an environment variable
5. never commit it to GitHub

In the original notebook, the API key was written directly as:

```python
key = ""
```

That is not safe. A better approach is to use environment variables.

## Store API Key Safely

Create a `.env` file:

```text
POLYGON_API_KEY=your_api_key_here
```

Add `.env` to `.gitignore`:

```text
.env
__pycache__/
*.pyc
```

Install `python-dotenv`:

```bash
pip install python-dotenv
```

Load the key:

```python
import os

from dotenv import load_dotenv


load_dotenv()

API_KEY = os.environ["POLYGON_API_KEY"]
```

This keeps the key out of your source code.

## Install Python Client

For newer projects, install the current official client:

```bash
pip install -U massive
```

Then import:

```python
from massive import RESTClient
```

If you are following older code from 2022, you may see:

```bash
pip install polygon-api-client==0.2.11
```

and:

```python
from polygon import RESTClient
```

The old version may still be useful if you are reproducing an old notebook, but for new projects, start from the latest client and official documentation.

## REST API vs WebSocket API

There are two common ways to get market data.

| API Type | Best For |
|---|---|
| REST API | Historical data, one-time queries, backtesting, dashboards |
| WebSocket API | Realtime streams, live dashboards, alert systems |

Use REST when you want data for a date range.

Use WebSocket when you want to listen for live updates.

## What Is Aggregate Data?

Aggregate data means OHLCV data grouped by time.

OHLCV means:

| Column | Meaning |
|---|---|
| Open | first price in the period |
| High | highest price in the period |
| Low | lowest price in the period |
| Close | last price in the period |
| Volume | traded volume in the period |

For example, one-minute aggregate bars contain one row per minute.

## Get Aggregate Stock Data with REST API

First, create the client.

```python
import os

import pandas as pd
from dotenv import load_dotenv
from massive import RESTClient


load_dotenv()

API_KEY = os.environ["POLYGON_API_KEY"]

client = RESTClient(api_key=API_KEY)
```

Now request one-minute aggregate bars for Apple.

```python
ticker = "AAPL"

aggs = []

for bar in client.list_aggs(
    ticker=ticker,
    multiplier=1,
    timespan="minute",
    from_="2022-06-10",
    to="2022-06-21",
    limit=50000
):
    aggs.append(bar)
```

This returns aggregate bar objects.

## Convert Aggregates to pandas DataFrame

The exact object format can depend on the client version. A safe approach is to convert each item to a dictionary.

```python
records = []

for bar in aggs:
    if hasattr(bar, "__dict__"):
        records.append(bar.__dict__)
    else:
        records.append(dict(bar))

df = pd.DataFrame(records)

df.head()
```

If your client already returns dictionaries, this can be simpler:

```python
df = pd.DataFrame(aggs)
```

## Understand Aggregate Columns

Depending on the client version, columns may be long names or short names.

Old JSON-style columns often looked like this:

| Column | Meaning |
|---|---|
| `v` | volume |
| `vw` | volume weighted average price |
| `o` | open |
| `c` | close |
| `h` | high |
| `l` | low |
| `t` | timestamp in milliseconds |
| `n` | number of transactions |

A cleaned version can use readable column names:

```python
rename_map = {
    "v": "volume",
    "vw": "vwap",
    "o": "open",
    "c": "close",
    "h": "high",
    "l": "low",
    "t": "timestamp",
    "n": "transactions",
}

df = df.rename(columns=rename_map)
```

## Convert Timestamp to Datetime

Polygon-style aggregate timestamps are often in milliseconds.

```python
df["datetime"] = pd.to_datetime(
    df["timestamp"],
    unit="ms",
    utc=True
)

df = df.sort_values("datetime")

df.head()
```

If you want New York time:

```python
df["datetime_ny"] = df["datetime"].dt.tz_convert("America/New_York")
```

Timezone handling is important in market data because exchanges operate in specific timezones.

## Select Useful Columns

```python
columns = [
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "transactions"
]

df = df[[column for column in columns if column in df.columns]]

df.head()
```

Now the data is ready for analysis.

## Plot Close Price

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.plot(df["datetime"], df["close"])

plt.title("AAPL 1-Minute Close Price")
plt.xlabel("Time")
plt.ylabel("Close Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

This gives a simple line plot of close prices.

## Save Data to CSV

```python
df.to_csv("aapl_1min_bars.csv", index=False)
```

Read it later:

```python
df = pd.read_csv("aapl_1min_bars.csv")
```

Saving data is useful for debugging, analysis, and backtesting.

## Raw REST Request Alternative

You can also use the REST API manually with `requests`.

```python
import os

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.environ["POLYGON_API_KEY"]

ticker = "AAPL"

url = (
    "https://api.polygon.io/v2/aggs/ticker/"
    f"{ticker}/range/1/minute/2022-06-10/2022-06-21"
)

params = {
    "adjusted": "true",
    "sort": "asc",
    "limit": 50000,
    "apiKey": API_KEY,
}

response = requests.get(url, params=params, timeout=30)
response.raise_for_status()

data = response.json()

df = pd.DataFrame(data.get("results", []))
```

This is useful when you want to understand what the client library is doing internally.

## WebSocket for Realtime Data

REST is good for historical data. But if we want realtime market data, we need WebSockets.

WebSockets keep a connection open and send new events as they arrive.

In the old 2022 code, the WebSocket usage looked like this:

```python
from polygon import WebSocketClient, STOCKS_CLUSTER, RESTClient
import json
```

Then:

```python
symbols = ["AAPL"]
my_client = WebSocketClient(STOCKS_CLUSTER, key, close_handler)
my_client.run_async()
my_client.subscribe(*[f"AM.{s}" for s in symbols])
```

This was based on the older client style.

For newer projects, check the current WebSocket client examples in the official client repository because the API style may change between versions.

## WebSocket Event Types

For stock streams, common channels include:

| Channel | Meaning |
|---|---|
| `T.SYMBOL` | trades |
| `Q.SYMBOL` | quotes |
| `AM.SYMBOL` | minute aggregates |
| `A.SYMBOL` | second aggregates, depending on plan/support |

For example:

```text
AM.AAPL
```

means minute aggregate bars for Apple.

## Simple WebSocket Handler Concept

A WebSocket handler receives messages from the stream.

Example concept:

```python
def handle_message(message):
    """Handle incoming WebSocket messages."""
    print(message)
```

In practice, the message format depends on the client version.

A response may contain several events at once, so it is common to loop through messages.

```python
def handle_events(events):
    for event in events:
        event_type = getattr(event, "event_type", None)

        if event_type == "AM":
            print(event)
```

If the client gives dictionaries:

```python
def handle_events(events):
    for event in events:
        if event.get("ev") == "AM":
            print(event)
```

## Store Realtime Bars

For a real project, printing is not enough. We may want to store bars.

Possible storage options:

- CSV file
- SQLite
- PostgreSQL
- Redis
- TimescaleDB
- Parquet files
- message queue

For a simple CSV example:

```python
def append_bar_to_csv(bar, file_path="bars.csv"):
    row = pd.DataFrame([bar])
    row.to_csv(
        file_path,
        mode="a",
        header=not Path(file_path).exists(),
        index=False
    )
```

Remember to import `Path`:

```python
from pathlib import Path
```

## Realtime Data Pipeline Idea

A simple realtime pipeline can look like this:

```text
WebSocket stream
      |
      v
message handler
      |
      v
clean event
      |
      v
append to database
      |
      v
dashboard or alert system
```

This is useful for:

- live dashboards
- alert systems
- paper trading experiments
- market monitoring
- data collection
- strategy research

## REST vs WebSocket Example Use Cases

| Task | Better Choice |
|---|---|
| Download last 6 months of 1-minute bars | REST |
| Show live AAPL updates in dashboard | WebSocket |
| Backtest a strategy | REST |
| Monitor realtime trade events | WebSocket |
| Build a daily report | REST |
| Trigger alert when price moves | WebSocket |

## Common Problems and Fixes

### Problem 1: Authentication Error

Check:

- API key is correct
- API key is active
- environment variable is loaded
- subscription plan supports the endpoint

### Problem 2: No Data Returned

Possible reasons:

- ticker is wrong
- market was closed
- date range has no trading data
- plan does not support the data
- endpoint parameters are wrong

### Problem 3: WebSocket Connects but No Messages Arrive

Possible reasons:

- market is closed
- subscribed channel is not supported by your plan
- wrong ticker symbol
- wrong channel name
- connection is blocked
- you are using old client code with a newer package

### Problem 4: Too Many Requests

REST APIs can have rate limits.

Solutions:

- cache results
- reduce request frequency
- use pagination properly
- use a paid plan if needed
- use flat files for large historical downloads

### Problem 5: Timestamp Looks Wrong

Check the timestamp unit.

If timestamp is in milliseconds:

```python
pd.to_datetime(df["timestamp"], unit="ms", utc=True)
```

If timestamp is in seconds:

```python
pd.to_datetime(df["timestamp"], unit="s", utc=True)
```

## Good Practices

Here are some good practices when working with stock market APIs:

- keep API keys in environment variables
- do not commit keys to GitHub
- understand your data plan and limits
- store raw data before cleaning
- convert timestamps carefully
- use UTC internally
- convert to exchange timezone for display
- handle empty API responses
- add retry logic for network errors
- log WebSocket disconnects
- do not make trading decisions from untested data

## Minimal Project Structure

A small project can look like this:

```text
polygon-stock-data/
│
├── rest_bars.py
├── websocket_stream.py
├── config.py
├── requirements.txt
├── .env
├── .gitignore
└── data/
```

Example `requirements.txt`:

```text
massive
pandas
python-dotenv
requests
matplotlib
```

If using old code:

```text
polygon-api-client==0.2.11
pandas
python-dotenv
requests
matplotlib
```

## Full REST Example

```python
import os

import pandas as pd
from dotenv import load_dotenv
from massive import RESTClient


load_dotenv()

API_KEY = os.environ["POLYGON_API_KEY"]

client = RESTClient(api_key=API_KEY)

ticker = "AAPL"

aggs = []

for bar in client.list_aggs(
    ticker=ticker,
    multiplier=1,
    timespan="minute",
    from_="2022-06-10",
    to="2022-06-21",
    limit=50000
):
    aggs.append(bar)

records = []

for bar in aggs:
    if hasattr(bar, "__dict__"):
        records.append(bar.__dict__)
    else:
        records.append(dict(bar))

df = pd.DataFrame(records)

rename_map = {
    "v": "volume",
    "vw": "vwap",
    "o": "open",
    "c": "close",
    "h": "high",
    "l": "low",
    "t": "timestamp",
    "n": "transactions",
}

df = df.rename(columns=rename_map)

if "timestamp" in df.columns:
    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True
    )

df = df.sort_values("datetime")

df.to_csv("data/aapl_1min_bars.csv", index=False)

print(df.head())
print(df.shape)
```

## Final Thoughts

In this post, we learned how to use **Polygon.io stock market data in Python**. We covered REST aggregate bars, pandas conversion, timestamp cleaning, and the idea of WebSocket realtime streams.

The main ideas are:

- use REST API for historical data
- use WebSocket API for realtime data
- keep API keys secure
- convert JSON responses into pandas DataFrames
- clean timestamps carefully
- check the current client library before copying old code

The original 2022 code used `polygon-api-client==0.2.11`, which was useful at that time. For new projects, start from the current official client and documentation, then adapt the code based on your package version and subscription plan.
