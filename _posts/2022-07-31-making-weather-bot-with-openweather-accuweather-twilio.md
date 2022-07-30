---
title:  "Making Weather Bot with Accuweather, OpenWeather and Twilio"
date:   2022-07-31 09:29:17 +0545
categories:
    - urllib3
    - weather-bot
tags:
    - accuweather
    - openweather
header:
  teaser: assets/weather_bot/flow.png
---

To make a Weather Bot we are going to use [Twilio](https://www.twilio.com), it is a platform which allows us to send SMS, make calls using methods like API calls. It was the beginning of COVID crisis and I was living in my village where internet was available by only cellular 3G data. But it is a difficult thing to stay in my village from June to October because the weather is pretty unexpected. One day, I was walking by the village and I saw a huge tomato farm nearly damaged due to wind and the electricity was already gone. Also, my parent's corn farm was highly affected by it. The tomato farm being damaged has something to do with our error like not locking or tightening the farm wires/doors. I thought what if the farmer knew in the morning that the wind or rain is coming in the day. I wrote some codes for myself to backup charge and batteries in the time of electricity cut off.


## Flow of the Project
I had a free credits of AWS EC2 that I received from GitHub Student Developer Pack and I used that to host a Python app, for scheduling, I used Cron Job but we can use Cron Job, Apache Airflow, Asyncio too. We run this app each day to get forecast of 12 hours (from AccuWeather) and 48 hours (from OpenWeather). Below is the typical flow of the Weather Bot Project that I followed in 2020.

![]({{site.url}}/assets/weather_bot/flow.png)





## Prepare Twilio Account
First step is to sign up to [twilio](https://www.twilio.com/try-twilio). Once completed, a verification mail might be sent to the address. Lets verify that too. 

Next, We choose the phone number for the trial period.

![]({{site.url}}/assets/weather_bot/trial.png)

Then we need to choose the type of app, code we are willing to use. Lets chose something like below:

![]({{site.url}}/assets/weather_bot/plan.png)

Next a dashboard is shown and we can see how much of credit is remaining.

![]({{site.url}}/assets/weather_bot/dashboard.png)


## Send First SMS from Twilio

And if we scroll to the bottom of dashboard or [twilio.com/console](https://twilio.com/console), we can see the Account SID and Auth Token, we need that to send sms. Next, we will send a sample SMS using this info. But we need to have installed Twilio's helper python library as `pip install twilio`. Or follow the [official instruction](https://www.twilio.com/docs/python/install). Documentation of twilio have recommended to use [secure way (as environment variable)](http://twil.io/secure) of using Auth token and account sid but for the sake of simplicity, I am using those from plain text. Also we need to have phone number to send a sms from. On the console itself, we can get a phone number by simply clicking in `Get a Phone Number`. **Please note that the trial version can only send sms to verified phone number that we've used earlier**.


```python
from twilio.rest import Client

account_sid = "your sid"
auth_token = "your auth"
client = Client(account_sid, auth_token)

full_sms = "Hey, this sms is for testing only. But did you get it?"

message = client.messages \
                .create(
                     body=full_sms,
                     from_='+12283356824',
                     to='+9779864031167'
                 )

print(message.sid)

```

    SM9b4ca6b7d86a447fae4fea41c9cd7a44
    

If no error comes after the execution of above block of the code then the sms must be delivered. It will be something like, `Sent from your Twilio trial account - Hey, this sms is for testing only. But did you get it?`

Now that we have successfully sent a sms, lets actually send weather data to make a Weather Bot.

## Get Weather Data from Accu Weather

In our Weather Bot, we will use Accuweather's API to get 12hrs forecasted data. Accuweather gives us free API key that allows us to get weather data by simply calling the api. The more info about APIs can be found in the [official site](https://developer.accuweather.com).

Please keep patience and create a API key before following below.

We will use the API to get weather of 12 hours from now and write that in a string.


```python
# accu weather
import requests
from datetime import datetime, date


ACC_KEY = "your acc key from accuweather"
LOCATION = "Hetauda"
LOCATION_API = f"http://dataservice.accuweather.com/locations/v1/cities/search?apikey={ACC_KEY}&q={LOCATION}"
lresponse = requests.get(LOCATION_API)

if lresponse.status_code == 200:
    
    locationKey = lresponse.json()[0]["Key"]
    geo = lresponse.json()[0]["GeoPosition"]
    lon = geo["Longitude"]
    lat = geo["Latitude"]
    

HOURLY_ACC_API = f"http://dataservice.accuweather.com/forecasts/v1/hourly/12hour/{locationKey}?apikey={ACC_KEY}"


aresponse = requests.get(HOURLY_ACC_API)
ajson = aresponse.json()
ajson


sms = ""
for aj in ajson:
    date_time = aj["EpochDateTime"]   
    weather = aj["IconPhrase"]
    
    dtm = datetime.fromtimestamp(date_time).date().strftime('%Y %B %d %A')
    hour = datetime.fromtimestamp(date_time).time().hour
    minute = time = datetime.fromtimestamp(date_time).time().minute
    am="am"
    if hour>=12:
        am="pm"

    message = f"""{dtm.split(" ")[-1][:3]}, {hour}:{minute} {am}: {weather}\n"""
    sms += message
print(sms)

```

    Sat, 11:0 am: Thunderstorms
    Sat, 12:0 pm: Cloudy
    Sat, 13:0 pm: Thunderstorms
    Sat, 14:0 pm: Cloudy
    Sat, 15:0 pm: Cloudy
    Sat, 16:0 pm: Cloudy
    Sat, 17:0 pm: Cloudy
    Sat, 18:0 pm: Cloudy
    Sat, 19:0 pm: Cloudy
    Sat, 20:0 pm: Rain
    Sat, 21:0 pm: Cloudy
    Sat, 22:0 pm: Cloudy
    
    

## Get Weather Data with Open Weather
Now lets use OpenWeather's forecast in our existing part of Weather Bot because it gives us data of 48 hours in the future. So I preferred it in the end. Please get he key from the openweather's developer portal before following below. In the below code, I have combined the 12hrs forecast data of Accuweather and 48hrs forecast of Openweather.


```python

# open weather
# onecall api for hourly 48 hours forecast
part = "current,minutely,hourly,daily,alerts"
part = "current,minutely,daily,alerts"
OPW_KEY = "your open weather key"

ONECALL_API = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude={part}&appid={OPW_KEY}"

response = requests.get(ONECALL_API)

sms1 = ""
if response.status_code:
    rjson = response.json()

    content = rjson["hourly"]

    for l in content:
        date_time = l["dt"]
        
        weather = l["weather"][0]

        dtm = datetime.fromtimestamp(date_time).date().strftime('%Y %B %d %A')
        hour = datetime.fromtimestamp(date_time).time().hour
        minute = time = datetime.fromtimestamp(date_time).time().minute
        am="am"
        if hour>=12:
            am="pm"
        message = f"""{dtm.split(" ")[-1][:3]}, {hour}:{minute}, {weather["description"]}\n"""
        sms1 += message
full_sms = "AccuWeather 12hrs\n"+sms+"\n"+"OpenWeather 48hrs\n"+sms1
print(full_sms)

```

    AccuWeather 12hrs
    Sat, 11:0 am: Thunderstorms
    Sat, 12:0 pm: Cloudy
    Sat, 13:0 pm: Thunderstorms
    Sat, 14:0 pm: Cloudy
    Sat, 15:0 pm: Cloudy
    Sat, 16:0 pm: Cloudy
    Sat, 17:0 pm: Cloudy
    Sat, 18:0 pm: Cloudy
    Sat, 19:0 pm: Cloudy
    Sat, 20:0 pm: Rain
    Sat, 21:0 pm: Cloudy
    Sat, 22:0 pm: Cloudy
    
    OpenWeather 48hrs
    Sat, 9:45, overcast clouds
    Sat, 10:45, overcast clouds
    Sat, 11:45, overcast clouds
    Sat, 12:45, light rain
    Sat, 13:45, moderate rain
    Sat, 14:45, moderate rain
    Sat, 15:45, moderate rain
    Sat, 16:45, moderate rain
    Sat, 17:45, moderate rain
    Sat, 18:45, light rain
    Sat, 19:45, moderate rain
    Sat, 20:45, light rain
    Sat, 21:45, light rain
    Sat, 22:45, moderate rain
    Sat, 23:45, moderate rain
    Sun, 0:45, moderate rain
    Sun, 1:45, heavy intensity rain
    Sun, 2:45, heavy intensity rain
    Sun, 3:45, heavy intensity rain
    Sun, 4:45, heavy intensity rain
    Sun, 5:45, moderate rain
    Sun, 6:45, light rain
    Sun, 7:45, light rain
    Sun, 8:45, light rain
    Sun, 9:45, light rain
    Sun, 10:45, moderate rain
    Sun, 11:45, moderate rain
    Sun, 12:45, moderate rain
    Sun, 13:45, moderate rain
    Sun, 14:45, moderate rain
    Sun, 15:45, moderate rain
    Sun, 16:45, moderate rain
    Sun, 17:45, light rain
    Sun, 18:45, light rain
    Sun, 19:45, light rain
    Sun, 20:45, light rain
    Sun, 21:45, light rain
    Sun, 22:45, light rain
    Sun, 23:45, light rain
    Mon, 0:45, light rain
    Mon, 1:45, light rain
    Mon, 2:45, light rain
    Mon, 3:45, light rain
    Mon, 4:45, light rain
    Mon, 5:45, light rain
    Mon, 6:45, light rain
    Mon, 7:45, light rain
    Mon, 8:45, light rain
    
    

Now lets send the sms to my registered phone number. The maximum amount of character in sms allowed is 1600 so lets crop it too.


```python

if len(full_sms)>1600:
    nfull_sms = full_sms[:1600]
else:
    nfull_sms=full_sms[:]

message = client.messages \
                .create(
                     body=nfull_sms,
                     from_='+12283356824',
                     to='+9779864031167'
                 )

print(message.sid)

```

    SMec07a8a5fe064060af48db85b65bec9e
    

Now, it might take some more time to receive this sms because of its size. Once received, it should look something like below:

```
Sent from your Twilio trial account - AccuWeather 12hrs
Sat, 11:0 am: Thunderstorms
Sat, 12:0 pm: Cloudy
Sat, 13:0 pm: Thunderstorms
Sat, 14:0 pm: Cloudy
Sat, 15:0 pm: Cloudy
Sat, 16:0 pm: Cloudy
Sat, 17:0 pm: Cloudy
Sat, 18:0 pm: Cloudy
Sat, 19:0 pm: Cloudy
Sat, 20:0 pm: Rain
Sat, 21:0 pm: Cloudy
Sat, 22:0 pm: Cloudy

OpenWeather 48hrs
Sat, 9:45, overcast clouds
Sat, 10:45, overcast clouds
Sat, 11:45, overcast clouds
Sat, 12:45, light rain
Sat, 13:45, moderate rain
Sat, 14:45, moderate rain
Sat, 15:45, moderate rain
Sat, 16:45, moderate rain
Sat, 17:45, moderate rain
Sat, 18:45, light rain
Sat, 19:45, moderate rain
Sat, 20:45, light rain
Sat, 21:45, light rain
Sat, 22:45, moderate rain
Sat, 23:45, moderate rain
Sun, 0:45, moderate rain
Sun, 1:45, heavy intensity rain
Sun, 2:45, heavy intensity rain
Sun, 3:45, heavy intensity rain
Sun, 4:45, heavy intensity rain
Sun, 5:45, moderate rain
Sun, 6:45, light rain
Sun, 7:45, light rain
Sun, 8:45, light rain
Sun, 9:45, light rain
Sun, 10:45, moderate rain
Sun, 11:45, moderate rain
Sun, 12:45, moderate rain
Sun, 13:45, moderate rain
Sun, 14:45, moderate rain
Sun, 15:45, moderate rain
Sun, 16:45, moderate rain
Sun, 17:45, light rain
Sun, 18:45, light rain
Sun, 19:45, light rain
Sun, 20:45, light rain
Sun, 21:45, light rain
Sun, 22:45, light rain
Sun, 23:45, light rain
Mon, 0:45, light rain
Mon, 1:45, light rain
Mon, 2:45, light rain
Mon, 3:45, light rain
Mon, 4:45, light rain
Mon, 5:45, light rain
Mon, 6:45, light rain
Mon, 7:45, light rain
Mon, 8:45, light rain
```

Now comes the part where we need to schedule our weather bot and run it. Since my problem was with electricity and internet, I had no other options than to choose the cloud services. I once had received a GitHub's Student Developer Pack and it provided me with some free credits to AWS and I choose the simplest instance in EC2. I hosted this python app in a CRON Job and used to receive sms everyday until the credit in Twilio ran out. There are other ways to schedule this job like [using Airflow](https://dataqoil.com/?s=airflow) and [Asyncio](https://dataqoil.com/?s=asyncio).
