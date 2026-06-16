---
layout: single
title: "Working with Datetime in Python: datetime, timedelta, Timezones, zoneinfo, pytz, Pendulum, and pandas"
date: 2022-10-02 01:29:17 +0545
last_modified_at: 2026-06-16
categories:
  - Python
  - DateTime
  - Tutorial
tags:
  - "Python"
  - "datetime"
  - "timedelta"
  - "timezone"
  - "zoneinfo"
  - "pytz"
  - "pendulum"
  - "pandas"
  - "date parsing"
description: "A beginner-friendly guide to working with datetime in Python, including date parsing, formatting, timedelta, timezones, zoneinfo, pytz, pendulum, pandas, and real-world examples."
excerpt: "Learn how to work with dates and times in Python using datetime, timedelta, strptime, strftime, timestamps, timezones, zoneinfo, pytz, pendulum, and pandas."
header:
  teaser: assets/python/datetime.png
  og_image: assets/python/datetime.png
  image_description: "Python datetime tutorial cover image"
toc: true
toc_label: "Datetime in Python"
toc_icon: "clock"
toc_sticky: true
read_time: true
share: true
related: true
classes: wide
---

Working with **datetime in Python** can be confusing at first. Dates and times look simple, but real projects often involve different formats, timezones, timestamps, daylight saving time, scheduling, and data from multiple sources.

In this tutorial, I will explain the most useful parts of working with datetime in Python. Instead of covering every function in the documentation, I will focus on practical examples.

We will cover:

- creating date and datetime objects
- parsing strings into datetime
- formatting datetime into strings
- getting the current time
- adding and subtracting time
- comparing dates and times
- working with timestamps
- handling timezones
- using `zoneinfo`, `pytz`, and `pendulum`
- working with datetime in pandas
- solving real-world datetime problems

## Why Datetime Can Be Difficult

Datetime becomes difficult because different systems store time differently.

For example:

- one API may return `2022-10-01`
- another may return `10/01/2022`
- another may return `2022-10-01T13:25:13Z`
- one database may store UTC time
- another may store local time
- one country may use daylight saving time
- another country may not

Because of this, it is important to understand the basics clearly.

## Import the datetime Module

Python has a built-in module called `datetime`.

```python
import datetime

print(datetime)
```

The `datetime` module includes useful classes such as:

- `datetime.date`
- `datetime.time`
- `datetime.datetime`
- `datetime.timedelta`
- `datetime.timezone`

The naming can be confusing because the module is called `datetime`, and one of the classes inside it is also called `datetime`.

A common import style is:

```python
from datetime import datetime, date, time, timedelta, timezone
```

In this post, I will show both styles where useful.

## Date, Time, and Datetime

Python separates date and time concepts.

### Date Object

A `date` object stores only year, month, and day.

```python
from datetime import date

my_date = date(2022, 10, 1)

print(my_date)
print(my_date.year)
print(my_date.month)
print(my_date.day)
```

Output:

```text
2022-10-01
2022
10
1
```

### Time Object

A `time` object stores time without a date.

```python
from datetime import time

my_time = time(hour=13, minute=25, second=13)

print(my_time)
```

Output:

```text
13:25:13
```

### Datetime Object

A `datetime` object stores both date and time.

```python
from datetime import datetime

my_datetime = datetime(
    year=2022,
    month=10,
    day=1,
    hour=13,
    minute=25,
    second=13,
    microsecond=100
)

print(my_datetime)
```

Output:

```text
2022-10-01 13:25:13.000100
```

## Create a Datetime from Integers

The simplest way to create a datetime is to pass year, month, and day.

```python
import datetime

print(datetime.datetime(year=1990, month=12, day=11))
print(
    datetime.datetime(
        year=2022,
        month=10,
        day=1,
        hour=13,
        minute=25,
        second=13,
        microsecond=100
    )
)
```

Output:

```text
1990-12-11 00:00:00
2022-10-01 13:25:13.000100
```

The year, month, and day are required. Hour, minute, second, and microsecond are optional.

## Parse String to Datetime with strptime

Very often, dates come as strings. We can convert a string into a datetime object using `strptime`.

```python
from datetime import datetime

dt1 = datetime.strptime("2022-01-23", "%Y-%m-%d")
dt2 = datetime.strptime("2022/01/23", "%Y/%m/%d")

print(dt1)
print(dt2)
```

Output:

```text
2022-01-23 00:00:00
2022-01-23 00:00:00
```

The second argument tells Python the format of the string.

### Parse Date and Time

```python
dt = datetime.strptime("2022-01-23 12:23:30", "%Y-%m-%d %H:%M:%S")

print(dt)
```

Output:

```text
2022-01-23 12:23:30
```

## Common Datetime Format Codes

Here are some common format codes used with `strptime` and `strftime`.

| Code | Meaning | Example |
|---|---|---|
| `%Y` | Four-digit year | `2022` |
| `%y` | Two-digit year | `22` |
| `%m` | Month number | `10` |
| `%B` | Full month name | `October` |
| `%b` | Short month name | `Oct` |
| `%d` | Day of month | `09` |
| `%A` | Full weekday name | `Sunday` |
| `%a` | Short weekday name | `Sun` |
| `%H` | Hour, 24-hour clock | `21` |
| `%I` | Hour, 12-hour clock | `09` |
| `%M` | Minute | `33` |
| `%S` | Second | `01` |
| `%p` | AM or PM | `PM` |
| `%z` | UTC offset | `+0100` |
| `%Z` | Timezone name | `CET` |

## Convert Datetime to String with strftime

The `strftime` method converts a datetime object into a formatted string.

```python
from datetime import datetime

dt = datetime.now()

print(f"Original: {dt}")
print(dt.strftime("%Y/%m/%d %H:%M:%S"))
print(dt.strftime("%Y.%m.%d %H:%M:%S"))
print(dt.strftime("%d/%m/%Y %H.%M.%S"))
```

Example output:

```text
Original: 2022-11-09 19:38:02.419127
2022/11/09 19:38:02
2022.11.09 19:38:02
09/11/2022 19.38.02
```

This is useful when different data sources or databases need different date formats.

## Get Current Date and Time

There are different ways to get the current date and time.

```python
from datetime import datetime

print(datetime.now())
print(datetime.utcnow())
```

`datetime.now()` gives the current local datetime.

`datetime.utcnow()` gives the current UTC datetime, but it returns a timezone-naive datetime. In modern Python code, it is usually better to use timezone-aware UTC time.

```python
from datetime import datetime, timezone

now_utc = datetime.now(timezone.utc)

print(now_utc)
```

## Naive vs Aware Datetime

A datetime object can be **naive** or **aware**.

### Naive Datetime

A naive datetime has no timezone information.

```python
from datetime import datetime

dt = datetime.now()

print(dt)
print(dt.tzinfo)
```

Output:

```text
2022-11-09 20:13:57.279373
None
```

### Aware Datetime

An aware datetime has timezone information.

```python
from datetime import datetime, timezone

dt = datetime.now(timezone.utc)

print(dt)
print(dt.tzinfo)
```

Output:

```text
2022-11-09 20:13:57.279373+00:00
UTC
```

In real applications, timezone-aware datetime is usually safer.

## Get Date and Time Values

We can extract values such as year, month, day, hour, minute, and second.

```python
from datetime import datetime

dt = datetime.now()

print(f"Year: {dt.year}")
print(f"Month: {dt.month}")
print(f"Day: {dt.day}")
print(f"Hour: {dt.hour}")
print(f"Minute: {dt.minute}")
print(f"Second: {dt.second}")

print(f"Date: {dt.date()}")
print(f"Time: {dt.time()}")
```

## Add and Subtract Time with timedelta

We can use `timedelta` to add or subtract time.

```python
from datetime import datetime, timedelta

dt = datetime.now()

print(f"Current datetime: {dt}")
print(f"Add one day: {dt + timedelta(days=1)}")
print(f"Subtract one day: {dt - timedelta(days=1)}")
print(f"Add one hour: {dt + timedelta(hours=1)}")
print(f"Add 30 minutes: {dt + timedelta(minutes=30)}")
```

Output:

```text
Current datetime: 2022-11-09 19:15:38.568075
Add one day: 2022-11-10 19:15:38.568075
Subtract one day: 2022-11-08 19:15:38.568075
Add one hour: 2022-11-09 20:15:38.568075
Add 30 minutes: 2022-11-09 19:45:38.568075
```

### Important Note About Months

`timedelta(days=30)` is not the same as adding one calendar month.

Months have different lengths:

- February can have 28 or 29 days
- April has 30 days
- May has 31 days

For calendar month operations, use packages such as `dateutil` or `pendulum`.

## Get Day and Month Names

We can use `strftime` to get day and month names.

```python
from datetime import datetime

dt = datetime.now()

print(dt.strftime("%A"))
print(dt.strftime("%B"))
print(dt.strftime("%Y-%B-%A"))
print(dt.strftime("%Y-%m-%A"))
```

Example output:

```text
Wednesday
November
2022-November-Wednesday
2022-11-Wednesday
```

This is useful when writing reports, emails, dashboards, or logs.

## Find Time Difference

To find the difference between two datetime values, subtract one from the other.

```python
from datetime import datetime, timedelta

dt1 = datetime.now()
dt2 = dt1 + timedelta(minutes=5.5, seconds=40)

difference = dt2 - dt1

print(f"DT1: {dt1}")
print(f"DT2: {dt2}")
print(f"Time difference: {difference}")
print(f"Difference in seconds: {difference.total_seconds()}")
```

Output:

```text
Time difference: 0:06:10
Difference in seconds: 370.0
```

## Compare Datetime Objects

Datetime objects support comparison operators.

```python
dt1 < dt2
dt1 == dt2
dt1 > dt2
dt1 != dt2
```

Example result:

```text
(True, False, False, True)
```

This is useful for filtering data, checking deadlines, scheduling tasks, and comparing events.

## Work with Timestamps

A timestamp usually represents seconds since the Unix epoch:

```text
1970-01-01 00:00:00 UTC
```

We can convert datetime to timestamp.

```python
from datetime import datetime

dt = datetime.strptime("2022-01-23 12:23:30", "%Y-%m-%d %H:%M:%S")

timestamp = dt.timestamp()

print(timestamp)
```

We can also convert a timestamp back to datetime.

```python
print(datetime.fromtimestamp(timestamp))
print(datetime.utcfromtimestamp(timestamp))
```

For timezone-aware code, prefer:

```python
from datetime import datetime, timezone

dt_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc)

print(dt_utc)
```

## Working with Timezones

Timezones are one of the most difficult parts of datetime work.

A common mistake is to use `replace(tzinfo=...)` incorrectly. Replacing timezone does not convert the actual time. It only attaches timezone information to the datetime object.

For timezone conversion, use `astimezone()`.

## Using zoneinfo for Timezones

For modern Python versions, `zoneinfo` is the standard library way to work with IANA timezones.

```python
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

now_utc = datetime.now(timezone.utc)

berlin_time = now_utc.astimezone(ZoneInfo("Europe/Berlin"))
kathmandu_time = now_utc.astimezone(ZoneInfo("Asia/Kathmandu"))

print(f"UTC: {now_utc}")
print(f"Berlin: {berlin_time}")
print(f"Kathmandu: {kathmandu_time}")
```

This correctly handles timezone offsets and daylight saving time where applicable.

## Adding Timezone to a Naive Datetime

If you already have a naive datetime that represents local Berlin time, you can attach the timezone like this:

```python
from datetime import datetime
from zoneinfo import ZoneInfo

naive_berlin_time = datetime(2022, 11, 9, 21, 2, 46)

aware_berlin_time = naive_berlin_time.replace(
    tzinfo=ZoneInfo("Europe/Berlin")
)

print(aware_berlin_time)
```

But be careful. This does not convert time. It only says, "this datetime should be interpreted as Berlin time."

## Convert Between Timezones

If you have an aware datetime, use `astimezone()` to convert it.

```python
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

now_utc = datetime.now(timezone.utc)

berlin_time = now_utc.astimezone(ZoneInfo("Europe/Berlin"))
kathmandu_time = berlin_time.astimezone(ZoneInfo("Asia/Kathmandu"))

print(berlin_time)
print(kathmandu_time)
```

Nepal does not use daylight saving time, so its offset is usually `+05:45`.

## Using pytz for Timezones

Older Python projects often use `pytz`.

Install it with:

```bash
pip install pytz
```

Import it:

```python
import datetime
import pytz
```

### Get a Timezone

```python
berlin = pytz.timezone("Europe/Berlin")
kathmandu = pytz.timezone("Asia/Kathmandu")
```

### Localize a Naive Datetime with pytz

With `pytz`, use `localize()` instead of directly using `replace()`.

```python
naive_dt = datetime.datetime(2022, 11, 9, 21, 2, 46)

berlin_dt = berlin.localize(naive_dt)

print(berlin_dt)
```

### Convert Timezone with pytz

```python
kathmandu_dt = berlin_dt.astimezone(kathmandu)

print(kathmandu_dt)
```

### Why Not Use replace with pytz?

This can produce strange historical offsets such as `+00:53` for Berlin:

```python
bad_dt = naive_dt.replace(tzinfo=pytz.timezone("Europe/Berlin"))

print(bad_dt)
```

This is why `pytz.localize()` is recommended for naive datetime objects.

For new projects, I prefer `zoneinfo` when possible because it is part of the Python standard library.

## Using Pendulum for Easier Datetime Work

[`pendulum`](https://pypi.org/project/pendulum/) is a third-party datetime library that makes many datetime operations easier.

Install it with:

```bash
pip install pendulum
```

Import it:

```python
import pendulum
```

Get the current time:

```python
now = pendulum.now()

print(now)
```

Get current time in a specific timezone:

```python
berlin_now = pendulum.now("Europe/Berlin")
kathmandu_now = pendulum.now("Asia/Kathmandu")

print(berlin_now)
print(kathmandu_now)
```

## Find the Next Friday with Pendulum

Pendulum makes date navigation simple.

```python
import pendulum

now = pendulum.now()

next_friday = now.next(pendulum.FRIDAY)

print(next_friday)
```

This is useful for scheduling tasks.

## Add Months with Pendulum

`timedelta` does not handle calendar months directly, but Pendulum can.

```python
import pendulum

dt = pendulum.datetime(2022, 1, 31)

print(dt.add(months=1))
print(dt.add(months=2))
```

Pendulum understands calendar-aware operations better than simple day-based addition.

## Human-Friendly Time Difference with Pendulum

Pendulum can also show differences in a human-readable way.

```python
import pendulum

past = pendulum.now().subtract(days=3)
now = pendulum.now()

print(past.diff_for_humans(now))
```

Example:

```text
3 days before
```

## Working with Datetime in pandas

In data science and analytics, datetime often appears in pandas DataFrames.

```python
import pandas as pd

df = pd.DataFrame({
    "created_at": [
        "2022-10-01 10:00:00",
        "2022-10-02 15:30:00",
        "2022-10-03 21:45:00"
    ]
})

df["created_at"] = pd.to_datetime(df["created_at"])

print(df)
```

## Extract Date Parts in pandas

```python
df["date"] = df["created_at"].dt.date
df["year"] = df["created_at"].dt.year
df["month"] = df["created_at"].dt.month
df["day"] = df["created_at"].dt.day
df["hour"] = df["created_at"].dt.hour

print(df)
```

This is useful for grouping, filtering, and plotting time-based data.

## Filter Last 7 Days in pandas

```python
now = pd.Timestamp.now()

last_7_days = df[df["created_at"] >= now - pd.Timedelta(days=7)]

print(last_7_days)
```

For timezone-aware data, make sure both sides of the comparison use compatible timezones.

## Real-World Example 1: Run a Task Every Saturday

Suppose we want to run stock backtesting every Saturday.

A simple scheduling idea is:

- get the current time
- check the current day
- if today is Saturday, run the task
- otherwise calculate seconds until next Saturday

```python
import datetime
import pendulum
from zoneinfo import ZoneInfo

current_time = datetime.datetime.now(ZoneInfo("Europe/Berlin"))
day = current_time.strftime("%A").upper()

print(f"Current time: {current_time}")
print(f"Day: {day}")

if day != "SATURDAY":
    now = pendulum.now("Europe/Berlin")
    next_saturday = now.next(pendulum.SATURDAY)

    sleep_till = (next_saturday - now).total_seconds()

    print(f"Next Saturday: {next_saturday}")
    print(f"Sleep for: {sleep_till} seconds")

    # await asyncio.sleep(sleep_till)
else:
    # perform backtesting here
    pass
```

For production, I would usually use a scheduler such as cron, APScheduler, Celery Beat, GitHub Actions, or a cloud scheduler instead of writing a never-ending loop manually.

## Real-World Example 2: Email Customers Before Month End

Suppose customers are charged on the last day of every month, and we want to email them one week before that.

We can calculate the last day of the current month.

```python
import datetime

today = datetime.date.today()

if today.month == 12:
    first_day_next_month = datetime.date(today.year + 1, 1, 1)
else:
    first_day_next_month = datetime.date(today.year, today.month + 1, 1)

last_day_current_month = first_day_next_month - datetime.timedelta(days=1)
email_day = last_day_current_month - datetime.timedelta(days=7)

print(f"Last day of current month: {last_day_current_month}")
print(f"Email customers on: {email_day}")

if today == email_day:
    print("Send email reminder.")
```

This handles different month lengths.

## Real-World Example 3: Store Datetime from Multiple Sources

Suppose we collect datetime values from multiple sources. Each source may use a different format and timezone.

A good approach is:

1. define the expected datetime format for each source
2. define the timezone for each source
3. parse the datetime
4. convert it to UTC
5. store UTC in the database
6. convert to local timezone only when displaying to users

Example configuration:

```python
source_datetime_format = {
    "source1": "%Y-%m-%d",
    "source2": "%m-%d-%Y %H.%M.%S",
    "source3": "%m-%d-%Y %H:%M:%S",
}

source_timezone = {
    "source1": "Europe/Berlin",
    "source2": "Asia/Kathmandu",
    "source3": "UTC",
}
```

Example parser:

```python
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def parse_source_datetime(source_name, datetime_text):
    dt_format = source_datetime_format[source_name]
    tz_name = source_timezone[source_name]

    naive_dt = datetime.strptime(datetime_text, dt_format)

    aware_dt = naive_dt.replace(tzinfo=ZoneInfo(tz_name))

    return aware_dt.astimezone(timezone.utc)
```

Then:

```python
utc_dt = parse_source_datetime("source1", "2022-10-01")

print(utc_dt)
```

For serious systems, always test timezone assumptions carefully.

## Best Practices for Datetime in Python

Here are some useful rules:

- Store datetime in UTC in databases.
- Convert to user timezone only for display.
- Prefer timezone-aware datetime for real applications.
- Use `zoneinfo` for modern Python timezone handling.
- Avoid mixing naive and aware datetime objects.
- Do not use `timedelta(days=30)` as a calendar month.
- Be careful with daylight saving time.
- Use `pd.to_datetime()` when working with pandas.
- Use clear datetime formats when parsing strings.
- Do not assume all APIs return the same timezone.

## Common Mistakes

Some common datetime mistakes are:

- comparing naive and aware datetime objects
- using `replace(tzinfo=...)` when conversion is needed
- forgetting daylight saving time
- storing local time in a database without timezone
- using inconsistent formats across sources
- assuming every month has 30 days
- ignoring UTC offsets in API responses
- using string comparison instead of datetime comparison
- parsing dates without knowing the source format

## datetime vs pytz vs zoneinfo vs pendulum vs pandas

| Tool | Best For |
|---|---|
| `datetime` | Basic date and time work |
| `timedelta` | Adding or subtracting days, seconds, minutes, hours |
| `zoneinfo` | Modern timezone handling in standard Python |
| `pytz` | Older projects that already use pytz |
| `pendulum` | Easier human-friendly datetime operations |
| `pandas` | Datetime columns in datasets and time series |

For normal Python applications, I usually start with `datetime` and `zoneinfo`.

For data analysis, I use pandas.

For cleaner scheduling or calendar operations, Pendulum can be helpful.

## Final Thoughts

In this post, we explored how to work with **datetime in Python**. We created datetime objects, parsed strings, formatted dates, calculated time differences, compared datetime values, worked with timestamps, and handled timezones.

The most important lesson is to be careful with timezones. If your application receives datetime values from different sources, convert them to a common timezone such as UTC before storing them. Then convert to local time only when showing the result to users.

Datetime work can be tricky, but once you understand parsing, formatting, timezone awareness, and conversion, it becomes much easier to manage real-world time data in Python.
