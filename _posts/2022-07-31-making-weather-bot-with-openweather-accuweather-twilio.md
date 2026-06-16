---
layout: single
title: "Build a Weather SMS Bot in Python with AccuWeather, OpenWeather, Twilio, and Cron"
date: 2022-07-31 09:29:17 +0545
last_modified_at: 2026-06-16
categories:
  - Python
  - API
  - Bot
  - Weather
tags:
  - "Weather Bot"
  - "Python"
  - "Twilio"
  - "AccuWeather"
  - "OpenWeather"
  - "SMS Bot"
  - "Cron Job"
  - "API"
  - "Automation"
description: "A beginner-friendly tutorial on building a Python weather SMS bot using AccuWeather, OpenWeather, Twilio, environment variables, and cron scheduling."
excerpt: "Learn how to build a Python weather SMS bot that fetches forecast data from AccuWeather and OpenWeather, formats it into a short SMS, and sends it with Twilio on a schedule."
header:
  teaser: assets/weather_bot/flow.png
  og_image: assets/weather_bot/flow.png
  image_description: "Flow diagram of a Python weather SMS bot using weather APIs and Twilio"
toc: true
toc_label: "Weather SMS Bot"
toc_icon: "cloud-sun-rain"
toc_sticky: true
read_time: true
share: true
related: true
classes: wide
---

In this tutorial, we will build a **weather SMS bot in Python** using AccuWeather, OpenWeather, Twilio, and a scheduler such as cron.

The goal is simple:

> Every day, the bot fetches the weather forecast and sends an SMS alert.

I first built this idea during the beginning of the COVID period, when I was living in my village. Internet access was mostly through cellular 3G data, and the weather from June to October was often unpredictable. One day, I saw a tomato farm nearly damaged by wind, and the electricity was already gone. My parents' corn farm was also affected.

That made me think:

> What if farmers knew in the morning that heavy rain or strong wind could come later in the day?

So I wrote a small Python script to fetch weather forecasts and send SMS alerts. It helped me prepare for power cuts and backup battery charging. The same idea can be extended for farmers, remote communities, field workers, and anyone who needs important weather alerts without opening a weather app.

## What This Weather Bot Does

The weather bot will:

1. get location information
2. fetch a short forecast from AccuWeather
3. fetch a longer forecast from OpenWeather
4. format the forecast into a readable SMS
5. keep the message within SMS size limits
6. send the SMS using Twilio
7. run automatically every day using cron or another scheduler

## Project Flow

When I first built this project in 2020, I had AWS EC2 credits from the GitHub Student Developer Pack. I hosted a Python app on a small EC2 instance and scheduled it with cron.

Other scheduling options are also possible, such as:

- cron job
- Apache Airflow
- APScheduler
- Celery Beat
- GitHub Actions
- systemd timer
- cloud scheduler
- asyncio loop for small experiments

The project flow is shown below.

![]({{site.url}}/assets/weather_bot/flow.png)

## Tools Used

We will use:

- **Twilio** for sending SMS
- **AccuWeather API** for short hourly forecast
- **OpenWeather API** for hourly forecast
- **Python requests** for API calls
- **python-dotenv** for loading secrets locally
- **cron** for scheduling

## Important Safety Note About API Keys

The original version of this blog used API keys directly in the code for simplicity.

That is not safe.

Do not write API keys, auth tokens, or phone numbers directly in your source code. Instead, store them in environment variables.

Bad:

```python
account_sid = "your sid"
auth_token = "your auth"
```

Better:

```python
import os

account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]
```

If you accidentally commit secrets to GitHub, rotate those keys immediately.

## Project Structure

A simple project structure can look like this:

```text
weather-sms-bot/
│
├── main.py
├── weather.py
├── sms.py
├── config.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

Here:

- `main.py` runs the full bot
- `weather.py` fetches weather data
- `sms.py` sends SMS
- `config.py` loads environment variables
- `.env` stores local secrets
- `.gitignore` prevents secrets from being committed

## Install Required Packages

Create `requirements.txt`.

```text
requests
twilio
python-dotenv
```

Install packages:

```bash
pip install -r requirements.txt
```

## Create a .env File

Create a `.env` file for local development.

```text
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_FROM_NUMBER=your_twilio_phone_number
TWILIO_TO_NUMBER=your_verified_or_target_phone_number

ACCUWEATHER_API_KEY=your_accuweather_api_key
OPENWEATHER_API_KEY=your_openweather_api_key

WEATHER_LOCATION=Hetauda
WEATHER_TIMEZONE=Asia/Kathmandu
```

Add `.env` to `.gitignore`.

```text
.env
__pycache__/
*.pyc
```

This keeps secrets out of Git.

## Prepare Twilio Account

First, sign up for Twilio:

- [Twilio](https://www.twilio.com)

After sign-up, verify your email address and phone number.

During the trial period, Twilio may require sending only to verified recipient numbers. So, if you are using a trial account, verify the phone number that should receive the weather SMS.

In the Twilio console, you can get:

- Account SID
- Auth Token
- Twilio phone number

The trial setup looks like this.

![]({{site.url}}/assets/weather_bot/trial.png)

Then choose the type of app and coding setup.

![]({{site.url}}/assets/weather_bot/plan.png)

After that, the Twilio dashboard is shown.

![]({{site.url}}/assets/weather_bot/dashboard.png)

## Send First SMS from Twilio

Before building the full weather bot, test Twilio SMS sending.

```python
import os

from dotenv import load_dotenv
from twilio.rest import Client


load_dotenv()

account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]
from_number = os.environ["TWILIO_FROM_NUMBER"]
to_number = os.environ["TWILIO_TO_NUMBER"]

client = Client(account_sid, auth_token)

message = client.messages.create(
    body="Hey, this SMS is for testing the weather bot.",
    from_=from_number,
    to=to_number
)

print(message.sid)
```

If there is no error, the SMS should be delivered.

For trial accounts, Twilio may add trial text to the message.

## Get Weather Data from AccuWeather

AccuWeather provides APIs for location search and forecasts.

In this project, we use AccuWeather for a short hourly forecast.

First, we search for the location and get the location key.

```python
import os
from datetime import datetime

import requests
from dotenv import load_dotenv


load_dotenv()

ACCUWEATHER_API_KEY = os.environ["ACCUWEATHER_API_KEY"]
LOCATION = os.getenv("WEATHER_LOCATION", "Hetauda")


def get_accuweather_location(location_name):
    """Search AccuWeather location and return location details."""
    url = "http://dataservice.accuweather.com/locations/v1/cities/search"

    params = {
        "apikey": ACCUWEATHER_API_KEY,
        "q": location_name
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    results = response.json()

    if not results:
        raise ValueError(f"No AccuWeather location found for: {location_name}")

    location = results[0]

    return {
        "key": location["Key"],
        "name": location["LocalizedName"],
        "lat": location["GeoPosition"]["Latitude"],
        "lon": location["GeoPosition"]["Longitude"],
    }
```

Now fetch 12-hour forecast.

```python
def get_accuweather_12_hour_forecast(location_key):
    """Get 12-hour forecast from AccuWeather."""
    url = (
        "http://dataservice.accuweather.com/"
        f"forecasts/v1/hourly/12hour/{location_key}"
    )

    params = {
        "apikey": ACCUWEATHER_API_KEY
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    return response.json()
```

## Format AccuWeather Forecast for SMS

```python
def format_accuweather_sms(hourly_data):
    """Format AccuWeather hourly forecast into SMS text."""
    lines = ["AccuWeather 12hrs"]

    for item in hourly_data:
        epoch_time = item["EpochDateTime"]
        weather = item["IconPhrase"]

        dt = datetime.fromtimestamp(epoch_time)

        day = dt.strftime("%a")
        time_text = dt.strftime("%I:%M %p").lstrip("0")

        lines.append(f"{day}, {time_text}: {weather}")

    return "\n".join(lines)
```

Example output:

```text
AccuWeather 12hrs
Sat, 11:00 AM: Thunderstorms
Sat, 12:00 PM: Cloudy
Sat, 1:00 PM: Thunderstorms
```

This format is easier to read than raw JSON.

## Get Weather Data from OpenWeather

OpenWeather's One Call API can provide hourly forecast data when we have latitude and longitude.

The older code used the `/data/2.5/onecall` endpoint. If you are using the newer One Call API version, check your OpenWeather subscription and endpoint format.

Example with One Call API 3.0-style endpoint:

```python
OPENWEATHER_API_KEY = os.environ["OPENWEATHER_API_KEY"]


def get_openweather_hourly_forecast(lat, lon):
    """Get hourly forecast from OpenWeather."""
    url = "https://api.openweathermap.org/data/3.0/onecall"

    params = {
        "lat": lat,
        "lon": lon,
        "exclude": "current,minutely,daily,alerts",
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    return response.json()
```

The `exclude` parameter removes parts of the response that we do not need.

## Format OpenWeather Forecast for SMS

```python
def format_openweather_sms(openweather_data, max_hours=24):
    """Format OpenWeather hourly forecast into SMS text."""
    lines = [f"OpenWeather next {max_hours}hrs"]

    hourly = openweather_data.get("hourly", [])[:max_hours]

    for item in hourly:
        epoch_time = item["dt"]
        weather = item["weather"][0]["description"]

        dt = datetime.fromtimestamp(epoch_time)

        day = dt.strftime("%a")
        time_text = dt.strftime("%I:%M %p").lstrip("0")

        temp = item.get("temp")

        if temp is not None:
            lines.append(f"{day}, {time_text}: {weather}, {temp:.1f}°C")
        else:
            lines.append(f"{day}, {time_text}: {weather}")

    return "\n".join(lines)
```

Instead of sending all 48 hours, this version sends the next 24 hours by default. This helps keep the SMS shorter.

## Combine AccuWeather and OpenWeather Forecasts

```python
def build_weather_message():
    """Build full weather SMS message."""
    location = get_accuweather_location(LOCATION)

    acc_data = get_accuweather_12_hour_forecast(location["key"])
    opw_data = get_openweather_hourly_forecast(location["lat"], location["lon"])

    acc_sms = format_accuweather_sms(acc_data)
    opw_sms = format_openweather_sms(opw_data, max_hours=24)

    full_sms = (
        f"Weather forecast for {location['name']}\n\n"
        f"{acc_sms}\n\n"
        f"{opw_sms}"
    )

    return full_sms
```

This function creates the final SMS content.

## Keep SMS Within a Safe Length

SMS messages have length limits. If the message is too long, it may be split into multiple segments or fail depending on the provider and region.

The original code used a maximum of 1600 characters.

```python
def crop_sms(text, max_length=1500):
    """Crop SMS to a safe length."""
    if len(text) <= max_length:
        return text

    return text[:max_length - 20] + "\n...message cropped"
```

This is simple and safe for a beginner project.

For a production system, it is better to summarize only important weather events such as rain, storm, wind, or alerts.

## Send Weather SMS with Twilio

```python
def send_sms(body):
    """Send SMS using Twilio."""
    account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    from_number = os.environ["TWILIO_FROM_NUMBER"]
    to_number = os.environ["TWILIO_TO_NUMBER"]

    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body=body,
        from_=from_number,
        to=to_number
    )

    return message.sid
```

## Full main.py Example

```python
import os
from datetime import datetime

import requests
from dotenv import load_dotenv
from twilio.rest import Client


load_dotenv()

ACCUWEATHER_API_KEY = os.environ["ACCUWEATHER_API_KEY"]
OPENWEATHER_API_KEY = os.environ["OPENWEATHER_API_KEY"]
LOCATION = os.getenv("WEATHER_LOCATION", "Hetauda")


def get_accuweather_location(location_name):
    url = "http://dataservice.accuweather.com/locations/v1/cities/search"

    params = {
        "apikey": ACCUWEATHER_API_KEY,
        "q": location_name
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    results = response.json()

    if not results:
        raise ValueError(f"No AccuWeather location found for: {location_name}")

    location = results[0]

    return {
        "key": location["Key"],
        "name": location["LocalizedName"],
        "lat": location["GeoPosition"]["Latitude"],
        "lon": location["GeoPosition"]["Longitude"],
    }


def get_accuweather_12_hour_forecast(location_key):
    url = (
        "http://dataservice.accuweather.com/"
        f"forecasts/v1/hourly/12hour/{location_key}"
    )

    params = {
        "apikey": ACCUWEATHER_API_KEY
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    return response.json()


def get_openweather_hourly_forecast(lat, lon):
    url = "https://api.openweathermap.org/data/3.0/onecall"

    params = {
        "lat": lat,
        "lon": lon,
        "exclude": "current,minutely,daily,alerts",
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    return response.json()


def format_accuweather_sms(hourly_data):
    lines = ["AccuWeather 12hrs"]

    for item in hourly_data:
        epoch_time = item["EpochDateTime"]
        weather = item["IconPhrase"]

        dt = datetime.fromtimestamp(epoch_time)

        day = dt.strftime("%a")
        time_text = dt.strftime("%I:%M %p").lstrip("0")

        lines.append(f"{day}, {time_text}: {weather}")

    return "\n".join(lines)


def format_openweather_sms(openweather_data, max_hours=24):
    lines = [f"OpenWeather next {max_hours}hrs"]

    hourly = openweather_data.get("hourly", [])[:max_hours]

    for item in hourly:
        epoch_time = item["dt"]
        weather = item["weather"][0]["description"]

        dt = datetime.fromtimestamp(epoch_time)

        day = dt.strftime("%a")
        time_text = dt.strftime("%I:%M %p").lstrip("0")

        temp = item.get("temp")

        if temp is not None:
            lines.append(f"{day}, {time_text}: {weather}, {temp:.1f}°C")
        else:
            lines.append(f"{day}, {time_text}: {weather}")

    return "\n".join(lines)


def crop_sms(text, max_length=1500):
    if len(text) <= max_length:
        return text

    return text[:max_length - 20] + "\n...message cropped"


def send_sms(body):
    account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    from_number = os.environ["TWILIO_FROM_NUMBER"]
    to_number = os.environ["TWILIO_TO_NUMBER"]

    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body=body,
        from_=from_number,
        to=to_number
    )

    return message.sid


def build_weather_message():
    location = get_accuweather_location(LOCATION)

    acc_data = get_accuweather_12_hour_forecast(location["key"])
    opw_data = get_openweather_hourly_forecast(location["lat"], location["lon"])

    acc_sms = format_accuweather_sms(acc_data)
    opw_sms = format_openweather_sms(opw_data, max_hours=24)

    return (
        f"Weather forecast for {location['name']}\n\n"
        f"{acc_sms}\n\n"
        f"{opw_sms}"
    )


if __name__ == "__main__":
    sms = build_weather_message()
    sms = crop_sms(sms)

    sid = send_sms(sms)

    print(f"SMS sent successfully. SID: {sid}")
```

## Example SMS Output

The SMS may look like this:

```text
Weather forecast for Hetauda

AccuWeather 12hrs
Sat, 11:00 AM: Thunderstorms
Sat, 12:00 PM: Cloudy
Sat, 1:00 PM: Thunderstorms
Sat, 2:00 PM: Cloudy

OpenWeather next 24hrs
Sat, 11:45 AM: overcast clouds, 27.2°C
Sat, 12:45 PM: light rain, 26.8°C
Sat, 1:45 PM: moderate rain, 25.9°C
```

This is shorter and more useful than sending too many hourly rows.

## Scheduling the Weather Bot with Cron

Now we need to run the bot automatically every day.

On Linux, open crontab:

```bash
crontab -e
```

Run the bot every morning at 6 AM:

```cron
0 6 * * * /usr/bin/python3 /home/ubuntu/weather-sms-bot/main.py >> /home/ubuntu/weather-sms-bot/weather.log 2>&1
```

This runs the script daily at 6:00 AM.

Make sure to use the correct Python path and project path.

## Scheduling Alternatives

Cron is simple, but there are other options.

### Apache Airflow

Good for complex workflows and data pipelines.

### APScheduler

Good for scheduling inside Python applications.

### GitHub Actions

Good for simple scheduled scripts if secrets are stored in GitHub Actions secrets.

### Cloud Scheduler

Good for production cloud environments.

### systemd Timer

Good for Linux servers where you want better service management than cron.

## Running on AWS EC2

When I first built this project, I used AWS EC2 credits from GitHub Student Developer Pack. I hosted the Python script on a small EC2 instance and ran it with cron.

The basic steps are:

1. create a small EC2 instance
2. install Python
3. clone your project
4. create `.env`
5. install requirements
6. test the script manually
7. add cron job
8. check logs

Example:

```bash
git clone https://github.com/your-username/weather-sms-bot.git
cd weather-sms-bot

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python main.py
```

Then add the cron job.

## Improve the Bot for Real Use

The current bot sends a forecast summary. We can make it better.

Possible improvements:

- send SMS only when rain is expected
- send SMS only when wind speed is above a threshold
- include temperature
- include humidity
- include severe weather alerts
- send messages in Nepali
- send WhatsApp instead of SMS
- store forecasts in a database
- compare forecasts from multiple providers
- add retry logic
- add logging
- add error notification
- send to multiple phone numbers

## Example: Alert Only When Rain Is Expected

Instead of sending the full forecast every day, we can send an SMS only if rain appears in the forecast.

```python
def should_send_rain_alert(message):
    keywords = ["rain", "thunderstorm", "storm", "shower"]

    text = message.lower()

    return any(keyword in text for keyword in keywords)
```

Use it:

```python
sms = build_weather_message()

if should_send_rain_alert(sms):
    send_sms(crop_sms(sms))
else:
    print("No rain alert needed today.")
```

This can reduce SMS cost and make alerts more useful.

## Common Problems and Fixes

### Problem 1: Twilio SMS Is Not Delivered

Check:

- recipient number is verified if using a trial account
- phone numbers are in E.164 format
- Twilio account has enough credit
- `from_` number is a valid Twilio number
- country/region is supported

### Problem 2: API Key Error

Check:

- API key is correct
- API key is active
- correct endpoint is used
- subscription supports that endpoint
- daily or monthly API limit is not exceeded

### Problem 3: OpenWeather One Call API Fails

OpenWeather has different API products and versions. Make sure your API key has access to the endpoint you are calling.

If `/data/3.0/onecall` does not work for your account, check your OpenWeather dashboard and documentation.

### Problem 4: Cron Job Runs but SMS Is Not Sent

Check the cron log file.

```bash
cat /home/ubuntu/weather-sms-bot/weather.log
```

Also make sure environment variables are available to cron. If using `.env`, the script should call:

```python
load_dotenv()
```

### Problem 5: Message Is Too Long

Reduce the number of forecast hours.

```python
format_openweather_sms(opw_data, max_hours=12)
```

Or send only important alerts.

## Final Thoughts

In this post, we built a **weather SMS bot in Python** using AccuWeather, OpenWeather, Twilio, and cron. The bot fetches weather data, formats it into a readable SMS, and sends it automatically.

This project is small but practical. It can be useful in areas where internet is limited, where people need early weather warnings, or where opening a weather app every day is not convenient.

The most important improvement over the original version is secure handling of API keys and phone numbers. Always use environment variables and never commit secrets to GitHub.
