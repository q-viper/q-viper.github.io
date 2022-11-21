---
title:  Working with datetime in Python
date:   2022-10-02 01:29:17 +0545
categories:
    - Python
    - datetime
tags:
    - python
    - datetime
header:
  teaser: assets/python/datetime.png
---
Working with DateTime in Python can be a challenging job if we do not know the right module to do the right thing. Here we will explore some of the useful modules based on their purpose and application rather than exploring the module as a whole.

## Using `datetime`
Datetime in Python can be done using various ways and one of the popular is using the standard library [datetime](https://docs.python.org/3/library/datetime.html). In general, we can use datetime datatype as datetime object in Python. We start by importing the library.



```python
import datetime

# what is it??
print(datetime)
```

    <module 'datetime' from '/usr/lib/python3.7/datetime.py'>
    


```python

```

### Making a Datetime
Before working with DateTime, we need to have one :). There are many ways to create DateTime and let's start with some.


#### Integer to Datetime 
We use `datetime.datetime`. We must pass year, month, and day to make datetime and hour, minute, and second optional to add time as well.


```python
print(datetime.datetime(year=1990, month=12, day=11))
print(datetime.datetime(year=2022, month=10, day=1, hour=13, minute=25, second=13, microsecond=100))
```

    1990-12-11 00:00:00
    2022-10-01 13:25:13.000100
    

#### Using String to datetime

We can use `datetime.strptime` which needs datetime in string and its format as the second parameter.


```python
# Date only but contains hour and minute as 0, 0
datetime.datetime.strptime("2022-01-23", "%Y-%m-%d"), datetime.datetime.strptime("2022/01/23", "%Y/%m/%d")

```




    (datetime.datetime(2022, 1, 23, 0, 0), datetime.datetime(2022, 1, 23, 0, 0))




```python
# passing time with seconds
datetime.datetime.strptime("2022-01-23 12:23:30", "%Y-%m-%d %H:%M:%S")

```




    datetime.datetime(2022, 1, 23, 12, 23, 30)



#### Using Timestamp

We can get datetime from timestamp as well. But for that we need timestamp. We can get timestamp from datetime by simply calling `timestamp()` of datetime object. 


```python
ts = datetime.datetime.strptime("2022-01-23 12:23:30", "%Y-%m-%d %H:%M:%S").timestamp()
print(ts)
print(f"From ts: {datetime.datetime.fromtimestamp(ts)}")
print(f"UTC From ts: {datetime.datetime.utcfromtimestamp(ts)}")
```

    1642940610.0
    From ts: 2022-01-23 12:23:30
    UTC From ts: 2022-01-23 12:23:30
    

#### Current Time


```python
## Getting UTC datetime
datetime.datetime.utcnow()
```




    datetime.datetime(2022, 11, 9, 20, 12, 15, 351996)




```python
# Getting current local datetime
datetime.datetime.now()
```




    datetime.datetime(2022, 11, 9, 20, 12, 23, 759600)



### Working with Datetime
Lets try to work with datetime.

#### Formatting Datetime
There are many formats of datetime and some are:
* YYYY-MM-DD
* YYYY-DD-MM
* DD/MM/YYYY

And we can format from one to another.


```python
dt = datetime.datetime.now()
print(f"DT: {dt}")
print(f"DT1: {dt.strftime('%Y/%m/%d %H:%M:%S')}")
print(f"DT1: {dt.strftime('%Y.%m.%d %H:%M:%S')}")
print(f"DT1: {dt.strftime('%d/%m/%Y %H.%M.%S')}")
```

    DT: 2022-11-09 19:38:02.419127
    DT1: 2022/11/09 19:38:02
    DT1: 2022.11.09 19:38:02
    DT1: 09/11/2022 19.38.02
    

Changing time from one to another has to be done when we are working with a different source of data and each has a different format.

#### Getting date and time values
We can get values of date, time, day, month, year, and so on.


```python
dt = datetime.datetime.now()
print(f"DT: {dt}")

print(f"Year: {dt.year}, Month: {dt.month}, Day: {dt.day}, Hour: {dt.hour}, Minute: {dt.minute}, Seconds: {dt.second}")
print(f"Date: {dt.date()}, Time: {dt.time()}")
```

    DT: 2022-11-09 20:13:57.279373
    Year: 2022, Month: 11, Day: 9, Hour: 20, Minute: 13, Seconds: 57
    Date: 2022-11-09, Time: 20:13:57.279373
    

#### Adding/Subtracting datetime
We can add or subtract any unit of time including year, month, day, hour, minute etc. But we can add/subtract from the day and below it only. For this, we can use `timedelta()`.


```python
print(f"DT: {dt}")
print(f"Timedelta day: {datetime.timedelta(days=1)}")
print(f"Added day: {dt+datetime.timedelta(days=1)}")
print(f"Added Month: {dt+datetime.timedelta(days=30)}")
print(f"Added Hour: {dt+datetime.timedelta(hours=1)}")
```

    DT: 2022-11-09 19:15:38.568075
    Timedelta day: 1 day, 0:00:00
    Added day: 2022-11-10 19:15:38.568075
    Added Month: 2022-12-09 19:15:38.568075
    Added Hour: 2022-11-09 20:15:38.568075
    

#### Day in Words
Simply using `strftime` we can get day in words. What if we want to send datetime but with day as word and month, like November 13 Sunday.


```python
dt.strftime('%A')
```




    'Wednesday'




```python

```




    '2022-11-Wednesday'



And to get it on datetime as well.


```python
dt.strftime("%Y-%B-%A"), dt.strftime("%Y-%m-%A")
```




    ('2022-November-Wednesday', '2022-11-Wednesday')



#### Finding Time Difference 

Let's create a datetime of now and add 5.5 minutes then 40 seconds to it. We will be adding 5.5*60+40=370 seconds.


```python
dt1 = datetime.datetime.now()
dt2 = dt1+datetime.timedelta(minutes=5.5, seconds=40)
print(f"DT1: {dt1} | DT2: {dt2}")
print(f"Time diff: {dt2-dt1}")
print(f"Diff in Secs: {(dt2-dt1).total_seconds()}")

```

    DT1: 2022-11-09 19:34:40.199437 | DT2: 2022-11-09 19:40:50.199437
    Time diff: 0:06:10
    Diff in Secs: 370.0
    

#### Boolean Operation
We can do simple boolean operations within datetime too. One example can be: we need to find the data for the last 7 days only from the dataframe. Let's use date-times from the last example.


```python
dt1<dt2, dt1==dt2, dt1>dt2, dt1!=dt2
```




    (True, False, False, True)



### Working with Time Zones
One of the difficult things while working with time-related data is timezone and in addition to that, daylight saving.

The Default DateTime object will have no timezone info. We can add one using replace.



```python
dt1.tzinfo
```

Working with timezone with a standard library is quite difficult but there is one package `pytz` which removes the complexity of it.

## Using `pytz` for timezones
[`pytz`](https://pypi.org/project/pytz/) is a library that is great to work with timezones and it can be easily implemented on the standard library `datetime`. We can install it like the below.


```python
!pip install pytz
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: pytz in /usr/local/lib/python3.7/dist-packages (2022.6)
    

### Getting Timezones
Let's see available time zones. It has a lot of time zones.


```python
import pytz
```


```python
pytz.all_timezones
```




    ['Africa/Abidjan',
     'Africa/Accra',
     'Africa/Addis_Ababa',
     'Africa/Algiers',
     'Africa/Asmara',
     'Africa/Asmera',
     'Africa/Bamako',
     'Africa/Bangui',
     'Africa/Banjul',
     'Africa/Bissau',
     'Africa/Blantyre',
     'Africa/Brazzaville',
     'Africa/Bujumbura',
     'Africa/Cairo',
     'Africa/Casablanca',
     'Africa/Ceuta',
     'Africa/Conakry',
     'Africa/Dakar',
     'Africa/Dar_es_Salaam',
     'Africa/Djibouti',
     'Africa/Douala',
     'Africa/El_Aaiun',
     'Africa/Freetown',
     'Africa/Gaborone',
     'Africa/Harare',
     'Africa/Johannesburg',
     'Africa/Juba',
     'Africa/Kampala',
     'Africa/Khartoum',
     'Africa/Kigali',
     'Africa/Kinshasa',
     'Africa/Lagos',
     'Africa/Libreville',
     'Africa/Lome',
     'Africa/Luanda',
     'Africa/Lubumbashi',
     'Africa/Lusaka',
     'Africa/Malabo',
     'Africa/Maputo',
     'Africa/Maseru',
     'Africa/Mbabane',
     'Africa/Mogadishu',
     'Africa/Monrovia',
     'Africa/Nairobi',
     'Africa/Ndjamena',
     'Africa/Niamey',
     'Africa/Nouakchott',
     'Africa/Ouagadougou',
     'Africa/Porto-Novo',
     'Africa/Sao_Tome',
     'Africa/Timbuktu',
     'Africa/Tripoli',
     'Africa/Tunis',
     'Africa/Windhoek',
     'America/Adak',
     'America/Anchorage',
     'America/Anguilla',
     'America/Antigua',
     'America/Araguaina',
     'America/Argentina/Buenos_Aires',
     'America/Argentina/Catamarca',
     'America/Argentina/ComodRivadavia',
     'America/Argentina/Cordoba',
     'America/Argentina/Jujuy',
     'America/Argentina/La_Rioja',
     'America/Argentina/Mendoza',
     'America/Argentina/Rio_Gallegos',
     'America/Argentina/Salta',
     'America/Argentina/San_Juan',
     'America/Argentina/San_Luis',
     'America/Argentina/Tucuman',
     'America/Argentina/Ushuaia',
     'America/Aruba',
     'America/Asuncion',
     'America/Atikokan',
     'America/Atka',
     'America/Bahia',
     'America/Bahia_Banderas',
     'America/Barbados',
     'America/Belem',
     'America/Belize',
     'America/Blanc-Sablon',
     'America/Boa_Vista',
     'America/Bogota',
     'America/Boise',
     'America/Buenos_Aires',
     'America/Cambridge_Bay',
     'America/Campo_Grande',
     'America/Cancun',
     'America/Caracas',
     'America/Catamarca',
     'America/Cayenne',
     'America/Cayman',
     'America/Chicago',
     'America/Chihuahua',
     'America/Coral_Harbour',
     'America/Cordoba',
     'America/Costa_Rica',
     'America/Creston',
     'America/Cuiaba',
     'America/Curacao',
     'America/Danmarkshavn',
     'America/Dawson',
     'America/Dawson_Creek',
     'America/Denver',
     'America/Detroit',
     'America/Dominica',
     'America/Edmonton',
     'America/Eirunepe',
     'America/El_Salvador',
     'America/Ensenada',
     'America/Fort_Nelson',
     'America/Fort_Wayne',
     'America/Fortaleza',
     'America/Glace_Bay',
     'America/Godthab',
     'America/Goose_Bay',
     'America/Grand_Turk',
     'America/Grenada',
     'America/Guadeloupe',
     'America/Guatemala',
     'America/Guayaquil',
     'America/Guyana',
     'America/Halifax',
     'America/Havana',
     'America/Hermosillo',
     'America/Indiana/Indianapolis',
     'America/Indiana/Knox',
     'America/Indiana/Marengo',
     'America/Indiana/Petersburg',
     'America/Indiana/Tell_City',
     'America/Indiana/Vevay',
     'America/Indiana/Vincennes',
     'America/Indiana/Winamac',
     'America/Indianapolis',
     'America/Inuvik',
     'America/Iqaluit',
     'America/Jamaica',
     'America/Jujuy',
     'America/Juneau',
     'America/Kentucky/Louisville',
     'America/Kentucky/Monticello',
     'America/Knox_IN',
     'America/Kralendijk',
     'America/La_Paz',
     'America/Lima',
     'America/Los_Angeles',
     'America/Louisville',
     'America/Lower_Princes',
     'America/Maceio',
     'America/Managua',
     'America/Manaus',
     'America/Marigot',
     'America/Martinique',
     'America/Matamoros',
     'America/Mazatlan',
     'America/Mendoza',
     'America/Menominee',
     'America/Merida',
     'America/Metlakatla',
     'America/Mexico_City',
     'America/Miquelon',
     'America/Moncton',
     'America/Monterrey',
     'America/Montevideo',
     'America/Montreal',
     'America/Montserrat',
     'America/Nassau',
     'America/New_York',
     'America/Nipigon',
     'America/Nome',
     'America/Noronha',
     'America/North_Dakota/Beulah',
     'America/North_Dakota/Center',
     'America/North_Dakota/New_Salem',
     'America/Nuuk',
     'America/Ojinaga',
     'America/Panama',
     'America/Pangnirtung',
     'America/Paramaribo',
     'America/Phoenix',
     'America/Port-au-Prince',
     'America/Port_of_Spain',
     'America/Porto_Acre',
     'America/Porto_Velho',
     'America/Puerto_Rico',
     'America/Punta_Arenas',
     'America/Rainy_River',
     'America/Rankin_Inlet',
     'America/Recife',
     'America/Regina',
     'America/Resolute',
     'America/Rio_Branco',
     'America/Rosario',
     'America/Santa_Isabel',
     'America/Santarem',
     'America/Santiago',
     'America/Santo_Domingo',
     'America/Sao_Paulo',
     'America/Scoresbysund',
     'America/Shiprock',
     'America/Sitka',
     'America/St_Barthelemy',
     'America/St_Johns',
     'America/St_Kitts',
     'America/St_Lucia',
     'America/St_Thomas',
     'America/St_Vincent',
     'America/Swift_Current',
     'America/Tegucigalpa',
     'America/Thule',
     'America/Thunder_Bay',
     'America/Tijuana',
     'America/Toronto',
     'America/Tortola',
     'America/Vancouver',
     'America/Virgin',
     'America/Whitehorse',
     'America/Winnipeg',
     'America/Yakutat',
     'America/Yellowknife',
     'Antarctica/Casey',
     'Antarctica/Davis',
     'Antarctica/DumontDUrville',
     'Antarctica/Macquarie',
     'Antarctica/Mawson',
     'Antarctica/McMurdo',
     'Antarctica/Palmer',
     'Antarctica/Rothera',
     'Antarctica/South_Pole',
     'Antarctica/Syowa',
     'Antarctica/Troll',
     'Antarctica/Vostok',
     'Arctic/Longyearbyen',
     'Asia/Aden',
     'Asia/Almaty',
     'Asia/Amman',
     'Asia/Anadyr',
     'Asia/Aqtau',
     'Asia/Aqtobe',
     'Asia/Ashgabat',
     'Asia/Ashkhabad',
     'Asia/Atyrau',
     'Asia/Baghdad',
     'Asia/Bahrain',
     'Asia/Baku',
     'Asia/Bangkok',
     'Asia/Barnaul',
     'Asia/Beirut',
     'Asia/Bishkek',
     'Asia/Brunei',
     'Asia/Calcutta',
     'Asia/Chita',
     'Asia/Choibalsan',
     'Asia/Chongqing',
     'Asia/Chungking',
     'Asia/Colombo',
     'Asia/Dacca',
     'Asia/Damascus',
     'Asia/Dhaka',
     'Asia/Dili',
     'Asia/Dubai',
     'Asia/Dushanbe',
     'Asia/Famagusta',
     'Asia/Gaza',
     'Asia/Harbin',
     'Asia/Hebron',
     'Asia/Ho_Chi_Minh',
     'Asia/Hong_Kong',
     'Asia/Hovd',
     'Asia/Irkutsk',
     'Asia/Istanbul',
     'Asia/Jakarta',
     'Asia/Jayapura',
     'Asia/Jerusalem',
     'Asia/Kabul',
     'Asia/Kamchatka',
     'Asia/Karachi',
     'Asia/Kashgar',
     'Asia/Kathmandu',
     'Asia/Katmandu',
     'Asia/Khandyga',
     'Asia/Kolkata',
     'Asia/Krasnoyarsk',
     'Asia/Kuala_Lumpur',
     'Asia/Kuching',
     'Asia/Kuwait',
     'Asia/Macao',
     'Asia/Macau',
     'Asia/Magadan',
     'Asia/Makassar',
     'Asia/Manila',
     'Asia/Muscat',
     'Asia/Nicosia',
     'Asia/Novokuznetsk',
     'Asia/Novosibirsk',
     'Asia/Omsk',
     'Asia/Oral',
     'Asia/Phnom_Penh',
     'Asia/Pontianak',
     'Asia/Pyongyang',
     'Asia/Qatar',
     'Asia/Qostanay',
     'Asia/Qyzylorda',
     'Asia/Rangoon',
     'Asia/Riyadh',
     'Asia/Saigon',
     'Asia/Sakhalin',
     'Asia/Samarkand',
     'Asia/Seoul',
     'Asia/Shanghai',
     'Asia/Singapore',
     'Asia/Srednekolymsk',
     'Asia/Taipei',
     'Asia/Tashkent',
     'Asia/Tbilisi',
     'Asia/Tehran',
     'Asia/Tel_Aviv',
     'Asia/Thimbu',
     'Asia/Thimphu',
     'Asia/Tokyo',
     'Asia/Tomsk',
     'Asia/Ujung_Pandang',
     'Asia/Ulaanbaatar',
     'Asia/Ulan_Bator',
     'Asia/Urumqi',
     'Asia/Ust-Nera',
     'Asia/Vientiane',
     'Asia/Vladivostok',
     'Asia/Yakutsk',
     'Asia/Yangon',
     'Asia/Yekaterinburg',
     'Asia/Yerevan',
     'Atlantic/Azores',
     'Atlantic/Bermuda',
     'Atlantic/Canary',
     'Atlantic/Cape_Verde',
     'Atlantic/Faeroe',
     'Atlantic/Faroe',
     'Atlantic/Jan_Mayen',
     'Atlantic/Madeira',
     'Atlantic/Reykjavik',
     'Atlantic/South_Georgia',
     'Atlantic/St_Helena',
     'Atlantic/Stanley',
     'Australia/ACT',
     'Australia/Adelaide',
     'Australia/Brisbane',
     'Australia/Broken_Hill',
     'Australia/Canberra',
     'Australia/Currie',
     'Australia/Darwin',
     'Australia/Eucla',
     'Australia/Hobart',
     'Australia/LHI',
     'Australia/Lindeman',
     'Australia/Lord_Howe',
     'Australia/Melbourne',
     'Australia/NSW',
     'Australia/North',
     'Australia/Perth',
     'Australia/Queensland',
     'Australia/South',
     'Australia/Sydney',
     'Australia/Tasmania',
     'Australia/Victoria',
     'Australia/West',
     'Australia/Yancowinna',
     'Brazil/Acre',
     'Brazil/DeNoronha',
     'Brazil/East',
     'Brazil/West',
     'CET',
     'CST6CDT',
     'Canada/Atlantic',
     'Canada/Central',
     'Canada/Eastern',
     'Canada/Mountain',
     'Canada/Newfoundland',
     'Canada/Pacific',
     'Canada/Saskatchewan',
     'Canada/Yukon',
     'Chile/Continental',
     'Chile/EasterIsland',
     'Cuba',
     'EET',
     'EST',
     'EST5EDT',
     'Egypt',
     'Eire',
     'Etc/GMT',
     'Etc/GMT+0',
     'Etc/GMT+1',
     'Etc/GMT+10',
     'Etc/GMT+11',
     'Etc/GMT+12',
     'Etc/GMT+2',
     'Etc/GMT+3',
     'Etc/GMT+4',
     'Etc/GMT+5',
     'Etc/GMT+6',
     'Etc/GMT+7',
     'Etc/GMT+8',
     'Etc/GMT+9',
     'Etc/GMT-0',
     'Etc/GMT-1',
     'Etc/GMT-10',
     'Etc/GMT-11',
     'Etc/GMT-12',
     'Etc/GMT-13',
     'Etc/GMT-14',
     'Etc/GMT-2',
     'Etc/GMT-3',
     'Etc/GMT-4',
     'Etc/GMT-5',
     'Etc/GMT-6',
     'Etc/GMT-7',
     'Etc/GMT-8',
     'Etc/GMT-9',
     'Etc/GMT0',
     'Etc/Greenwich',
     'Etc/UCT',
     'Etc/UTC',
     'Etc/Universal',
     'Etc/Zulu',
     'Europe/Amsterdam',
     'Europe/Andorra',
     'Europe/Astrakhan',
     'Europe/Athens',
     'Europe/Belfast',
     'Europe/Belgrade',
     'Europe/Berlin',
     'Europe/Bratislava',
     'Europe/Brussels',
     'Europe/Bucharest',
     'Europe/Budapest',
     'Europe/Busingen',
     'Europe/Chisinau',
     'Europe/Copenhagen',
     'Europe/Dublin',
     'Europe/Gibraltar',
     'Europe/Guernsey',
     'Europe/Helsinki',
     'Europe/Isle_of_Man',
     'Europe/Istanbul',
     'Europe/Jersey',
     'Europe/Kaliningrad',
     'Europe/Kiev',
     'Europe/Kirov',
     'Europe/Kyiv',
     'Europe/Lisbon',
     'Europe/Ljubljana',
     'Europe/London',
     'Europe/Luxembourg',
     'Europe/Madrid',
     'Europe/Malta',
     'Europe/Mariehamn',
     'Europe/Minsk',
     'Europe/Monaco',
     'Europe/Moscow',
     'Europe/Nicosia',
     'Europe/Oslo',
     'Europe/Paris',
     'Europe/Podgorica',
     'Europe/Prague',
     'Europe/Riga',
     'Europe/Rome',
     'Europe/Samara',
     'Europe/San_Marino',
     'Europe/Sarajevo',
     'Europe/Saratov',
     'Europe/Simferopol',
     'Europe/Skopje',
     'Europe/Sofia',
     'Europe/Stockholm',
     'Europe/Tallinn',
     'Europe/Tirane',
     'Europe/Tiraspol',
     'Europe/Ulyanovsk',
     'Europe/Uzhgorod',
     'Europe/Vaduz',
     'Europe/Vatican',
     'Europe/Vienna',
     'Europe/Vilnius',
     'Europe/Volgograd',
     'Europe/Warsaw',
     'Europe/Zagreb',
     'Europe/Zaporozhye',
     'Europe/Zurich',
     'GB',
     'GB-Eire',
     'GMT',
     'GMT+0',
     'GMT-0',
     'GMT0',
     'Greenwich',
     'HST',
     'Hongkong',
     'Iceland',
     'Indian/Antananarivo',
     'Indian/Chagos',
     'Indian/Christmas',
     'Indian/Cocos',
     'Indian/Comoro',
     'Indian/Kerguelen',
     'Indian/Mahe',
     'Indian/Maldives',
     'Indian/Mauritius',
     'Indian/Mayotte',
     'Indian/Reunion',
     'Iran',
     'Israel',
     'Jamaica',
     'Japan',
     'Kwajalein',
     'Libya',
     'MET',
     'MST',
     'MST7MDT',
     'Mexico/BajaNorte',
     'Mexico/BajaSur',
     'Mexico/General',
     'NZ',
     'NZ-CHAT',
     'Navajo',
     'PRC',
     'PST8PDT',
     'Pacific/Apia',
     'Pacific/Auckland',
     'Pacific/Bougainville',
     'Pacific/Chatham',
     'Pacific/Chuuk',
     'Pacific/Easter',
     'Pacific/Efate',
     'Pacific/Enderbury',
     'Pacific/Fakaofo',
     'Pacific/Fiji',
     'Pacific/Funafuti',
     'Pacific/Galapagos',
     'Pacific/Gambier',
     'Pacific/Guadalcanal',
     'Pacific/Guam',
     'Pacific/Honolulu',
     'Pacific/Johnston',
     'Pacific/Kanton',
     'Pacific/Kiritimati',
     'Pacific/Kosrae',
     'Pacific/Kwajalein',
     'Pacific/Majuro',
     'Pacific/Marquesas',
     'Pacific/Midway',
     'Pacific/Nauru',
     'Pacific/Niue',
     'Pacific/Norfolk',
     'Pacific/Noumea',
     'Pacific/Pago_Pago',
     'Pacific/Palau',
     'Pacific/Pitcairn',
     'Pacific/Pohnpei',
     'Pacific/Ponape',
     'Pacific/Port_Moresby',
     'Pacific/Rarotonga',
     'Pacific/Saipan',
     'Pacific/Samoa',
     'Pacific/Tahiti',
     'Pacific/Tarawa',
     'Pacific/Tongatapu',
     'Pacific/Truk',
     'Pacific/Wake',
     'Pacific/Wallis',
     'Pacific/Yap',
     'Poland',
     'Portugal',
     'ROC',
     'ROK',
     'Singapore',
     'Turkey',
     'UCT',
     'US/Alaska',
     'US/Aleutian',
     'US/Arizona',
     'US/Central',
     'US/East-Indiana',
     'US/Eastern',
     'US/Hawaii',
     'US/Indiana-Starke',
     'US/Michigan',
     'US/Mountain',
     'US/Pacific',
     'US/Samoa',
     'UTC',
     'Universal',
     'W-SU',
     'WET',
     'Zulu']



### Adding timezone
It seems like `pytz` uses standard datetime library under the hood as well.


```python
now = pytz.datetime.datetime.now()
print(f"Now: {now}")
print(f"Now added Berlin Time Zone: {now.replace(tzinfo=pytz.timezone('Europe/Berlin'))}")
print(f"Now Changed to Berlin Time: {now.astimezone(pytz.timezone('Europe/Berlin'))}")

```

    Now: 2022-11-09 20:02:46.248376
    Now added Berlin Time Zone: 2022-11-09 20:02:46.248376+00:53
    Now Changed to Berlin Time: 2022-11-09 21:02:46.248376+01:00
    

In the above example, the first time is the local time printed by Python. And we replaced its timezone with Europe/Berlin which added the +00:53 on the last of time. Then we changed the actual time to Europe/Berlin time which added one hour and can be seen on the +01:00 as well. Looking into the now time in the first line, it's 20:02:46 but the current time is 21:02:46 here in Germany so daylight saving is not working properly. Then on the last line, we can see that daylight saving is working properly. Now I want to see what time is it in Kathmandu because Nepal does not follow daylight savings.


```python
print(f"Now Changed to Kathmandu Time: {now.astimezone(pytz.timezone('Asia/Kathmandu'))}")
```

    Now Changed to Kathmandu Time: 2022-11-10 01:47:46.248376+05:45
    

It's the correct time but what if we changed it from the Berlin time which is daylight saving?


```python
print(f"Now Changed to Kathmandu Time from Berlin: {now.astimezone(pytz.timezone('Europe/Berlin')).astimezone(pytz.timezone('Asia/Kathmandu'))}")
```

    Now Changed to Kathmandu Time from Berlin: 2022-11-10 01:47:46.248376+05:45
    

It is being handled. How ironic!!

After adding the timezone, datetime object will be little different.


```python
now.replace(tzinfo=pytz.timezone('Europe/Berlin'))
```




    datetime.datetime(2022, 11, 9, 20, 2, 46, 248376, tzinfo=<DstTzInfo 'Europe/Berlin' LMT+0:53:00 STD>)




```python
# UTC time
now.replace(tzinfo=pytz.timezone('UTC'))
```




    datetime.datetime(2022, 11, 9, 20, 2, 46, 248376, tzinfo=<UTC>)



## Using `pendulum` for clever works
If we need to do more datetime operations then we can use [`pendulum`](https://pypi.org/project/pendulum/). We can install it simply by `pip install pendulum`.


```python
!pip install pendulum

```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting pendulum
      Downloading pendulum-2.1.2-cp37-cp37m-manylinux1_x86_64.whl (155 kB)
    [K     |████████████████████████████████| 155 kB 5.1 MB/s 
    [?25hCollecting pytzdata>=2020.1
      Downloading pytzdata-2020.1-py2.py3-none-any.whl (489 kB)
    [K     |████████████████████████████████| 489 kB 44.8 MB/s 
    [?25hRequirement already satisfied: python-dateutil<3.0,>=2.6 in /usr/local/lib/python3.7/dist-packages (from pendulum) (2.8.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil<3.0,>=2.6->pendulum) (1.15.0)
    Installing collected packages: pytzdata, pendulum
    Successfully installed pendulum-2.1.2 pytzdata-2020.1
    


```python
import pendulum
now = pendulum.now()
print(now)
```

    2022-11-09T20:19:39.486885+00:00
    

### When will next Friday be?
The pendulum tells us this with a simple call.


```python
now.next(pendulum.FRIDAY)
```




    DateTime(2022, 11, 11, 0, 0, 0, tzinfo=Timezone('Etc/UTC'))



## Problem Answers
Here I will write about some cases where I had to use datetime above-mentioned packages on real-world.

### 1. Run stock backtesting every Saturday.

* We can use Asyncio to sleep the system.
* Run everything in a never-ending loop and check the current day. If the current day is Saturday then perform backtesting.
* If the current day is not Friday then sleep until Saturday.



```python
import datetime, pytz, pendulum
curr_time = datetime.datetime.now().astimezone(pytz.timezone('Europe/Berlin'))
day = curr_time.strftime('%A').upper()
print(f"Curr Time: {curr_time}, Day: {day}")
if day!='SATURDAY':
  pnow = pendulum.now()
  next_sat = pnow.astimezone(pytz.timezone('Europe/Berlin')).next(pendulum.SATURDAY)
  print(f"Next SAT: {next_sat}")
  sleep_till = (next_sat-pnow).total_seconds()
  print(f"Sleep until: {sleep_till}secs.")
  # await asyncio.sleep(sleep_till)
else:
  pass
  # perform backtesting here
```

    Curr Time: 2022-11-09 21:33:01.985186+01:00, Day: WEDNESDAY
    Next SAT: 2022-11-12T00:00:00+01:00
    Sleep until: 181618.014127secs.
    

### 2. Email customers that they will be charged a fee on the last day of the month.
Let's say there is a platform where subscribed customers will be charged every month's end regardless of days in a month. And we will mail them a week before the end of the month. This is a quite tricky and fun thing to do.


```python
next_month = datetime.datetime(year=curr_time.year, month=curr_time.month+1, day=1)
print(f"Curr Time: {curr_time}, Next Month: {next_month}")
last_month_day = next_month-datetime.timedelta(days=1)
print(f"Last Month Day: {last_month_day}")

if curr_time.date()==last_month_day.date():
  print("Send emails.")


```

    Curr Time: 2022-11-09 21:33:01.985186+01:00, Next Month: 2022-12-01 00:00:00
    Last Month Day: 2022-11-30 00:00:00
    

### 3. Store datetime from multiple sources in a central database
Here datetime format and timezones could be different for different sources. And there will not be a quiet solution for this problem because there is not any information about what are the formats of datetime in different sources. But what we can do is:
* Prepare config for datetime formats from each source. Assuming that format won't change once set. Example
```python
source_datetime_format = {"source1":"%Y-%m-%d", "source2":"%m-%d-%Y %H.%M.%S", "source2":"%m-%d-%Y %H:%M:%S"}
destination_datetime_format = "%m/%d/%Y %H:%M:%S"
```
* Prepare config of timezone for each source's DateTime.



## Conclusion
Here I tried to use some of datetime packages in Python and did some examples as well there are a lot more to come. Stay tuned :)
