---
title:  Finding DateTime in Text Using Python
date:   2022-10-23 01:29:17 +0545
categories:
    - Python
    - DateTime
    - Text Processing
tags:
    - python
    - Text Processing
    - DateTime
header:
  teaser: assets/python/date_parser.png
---

Why do we need to find DateTime in the text? In the field of data science, we often have to deal with various kinds of data and one of the common is Text data but sometimes datetime in the text has to be extracted. Before jumping into a topic let's first start with a problem I recently encountered. I was given a task to extract sent and received email messages from a long thread of multipart emails. The email I used to get would be something like the below:
 
 
 
 
 
```python
 
eml = """Re: Documents Received
 
 
 
John Doe <john@doe.org>
Wed, Jun 1, 2011, 9:39 PM
to Emma, Don, Bucky
 
 
 
Lorem
Ipsum
Dorem
 
 
On 01/06/2011, at 7:57 PM, "Emma" <emma@thompson.com> wrote:
 
 
Lorem Ipsum?
 
Thanks John
 
On 1 June 2011 13:43, Bucky Hallam <bucky@barnes.com> wrote:
 
 
Lorem Ipsum is Dorem.
 
 
 
Thanks Emma"""
 
```
 
The above text is modified from an original multipart email. My goal was to separate received and sent emails from one another. I have written a blog about how to retrieve emails as well, please give it a try if you are interested. I split the entire text by `wrote:` and then have to split parts again using `On sent date`. The first part was easy but due to different variants of dates, the second part got terribly hard. Some of the first emails  I worked on had dates like `On Sun 11, 2022`, and I created a list like below
 
 
```python
[f'On {d},' for d in ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']]
```
 
 
 
 
    ['On Sun,', 'On Mon,', 'On Tue,', 'On Wed,', 'On Thu,', 'On Fri,', 'On Sat,']
 
 
 
It worked for some but when sent dates were in a different format based on mail servers, this failed. Now there is a number of ways one could do it. But all are based on finding the pattern of DateTime in the text. Usually, DateTime in the text has patterns like YYYY/MM/DD HH:MM:SS, DD/MM/YYYY HH:MM:SS and so on we could prepare regex for that and find where it matched.
 
 
```python
import re
```
 
## Finding Date Using Regex
 
### Format 1
Let's try to find the datetime in the text from the format YYYY/MM/DD without any time.
 
 
```python
pattern = r'\d{4}/\d{2}/\d{2}'
txt = "This is 2022/11/11 and we are waiting for 2022/11/12."
print(re.findall(pattern, txt, re.DOTALL))
```
 
    ['2022/11/11', '2022/11/12']
   
 
This works well but not in the case when another format like `-` is used instead of `/`.
 
 
```python
pattern = r'\d{4}/\d{2}/\d{2}'
txt = "This is 2022-11-11 and we are waiting for 2022/11/12."
print(re.findall(pattern, txt, re.DOTALL))
```
 
    ['2022/11/12']
   
 
It missed date here. We can simply use the or operator to add another format there.
 
 
```python
pattern = r'(\d{4}-\d{2}-\d{2}|\d{4}/\d{2}/\d{2})'
txt = "This is 2022-11-11 and we are waiting for 2022/11/12."
print(re.findall(pattern, txt, re.DOTALL))
```
 
    ['2022-11-11', '2022/11/12']
   
 
### Format 2
 
Let's use time too.
 
 
```python
pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}|\d{4}/\d{2}/\d{2})'
txt = "This is 2022-11-11 14:23:19 and we are waiting for 2022/11/12."
print(re.findall(pattern, txt, re.DOTALL))
```
 
    ['2022-11-11 14:23:19', '2022/11/12']
   
 
It worked but not much in the cases as we have in email. But we can split our text based on a found date too and it's very useful above.
 
 
```python
pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}|\d{4}/\d{2}/\d{2})'
txt = "This is 2022-11-11 14:23:19 and we are waiting for 2022/11/12."
print(re.split(pattern, txt, re.DOTALL))
```
 
    ['This is ', '2022-11-11 14:23:19', ' and we are waiting for ', '2022/11/12', '.']
   
 
## Our Email
 
There are different formats of datetime in the text in our above email.
* Wed, Jun 1, 2011, 9:39 PM
* 01/06/2011, at 7:57 PM
* 1 June 2011 13:43
 
And all of the above 3 requires different pattern as well so it's little tricky and more hard work to find them.
 
### For Jun 1, 2011, 9:39 PM
 
 
 
```python
pattern = r'([0-3]?[0-9], \d{4}, [0-2]?[0-9]:[0-5][0-9] [AaPp][Mm])'
txt = 'This is Wed, Jun 1, 2011, 9:29 AM and Wed, Jun 1, 2011, 19:39 PM'
print(re.findall(pattern, txt, re.DOTALL))
 
```
 
    ['1, 2011, 9:29 AM', '1, 2011, 19:39 PM']
   
 
### For 01/06/2011, at 7:57 PM
 
 
```python
pattern = r'([0-1]?[0-2]/[0-3]?[0-9]/\d{4}, at [0-2]?[0-9]:[0-5][0-9] [AaPp][Mm])'
txt = 'This is 01/06/2011, at 7:57 PM and 01/06/2011, at 19:57 PM'
print(re.findall(pattern, txt, re.DOTALL))
 
```
 
    ['01/06/2011, at 7:57 PM', '01/06/2011, at 19:57 PM']
   
 
### For 1 June 2011 13:43
 
 
```python
pattern = r'([0-1]?[0-2] \w{3,} \d{4} [0-2]?[0-9]:[0-5][0-9])'
txt = 'This is 1 June 2011 13:43'
print(re.findall(pattern, txt, re.DOTALL))
 
```
 
    ['1 June 2011 13:43']
   
 
But finding the pattern for each format is not a good solution. And there is not a golden pattern either. However, there are some Python packages that can help us in these cases.
 
## Using `dateutil`
If this package is not installed, please do it by `pip install dateutil`.
 
 
 
```python
from dateutil.parser import parse
```
 
By simply calling the parse method, we can get datetime.
 
 
```python
parse('This is 1 June 2011 13:43', fuzzy_with_tokens=True)
```
 
 
 
 
    (datetime.datetime(2011, 6, 1, 13, 43), ('This is ', ' '))
 
 
 
But this doesn't work always.
 
 
```python
parse('This is Wed, Jun 1, 2011, 9:29 AM and Wed, Jun 1, 2011, 19:39 PM', fuzzy_with_tokens=True)
```
 
 
    ---------------------------------------------------------------------------
 
    ParserError                               Traceback (most recent call last)
 
    <ipython-input-147-1f921311ad5f> in <module>
    ----> 1 parse('This is Wed, Jun 1, 2011, 9:29 AM and Wed, Jun 1, 2011, 19:39 PM', fuzzy_with_tokens=True)
   
 
    C:\ProgramData\Anaconda3\lib\site-packages\dateutil\parser\_parser.py in parse(timestr, parserinfo, **kwargs)
       1372         return parser(parserinfo).parse(timestr, **kwargs)
       1373     else:
    -> 1374         return DEFAULTPARSER.parse(timestr, **kwargs)
       1375
       1376
   
 
    C:\ProgramData\Anaconda3\lib\site-packages\dateutil\parser\_parser.py in parse(self, timestr, default, ignoretz, tzinfos, **kwargs)
        647
        648         if res is None:
    --> 649             raise ParserError("Unknown string format: %s", timestr)
        650
        651         if len(res) == 0:
   
 
    ParserError: Unknown string format: This is Wed, Jun 1, 2011, 9:29 AM and Wed, Jun 1, 2011, 19:39 PM
 
 
 
```python
parse('This is 01/06/2011, at 7:57 PM and 01/06/2011, at 19:57 PM', fuzzy_with_tokens=True)
```
 
 
    ---------------------------------------------------------------------------
 
    ParserError                               Traceback (most recent call last)
 
    <ipython-input-146-f7324a8c6b10> in <module>
    ----> 1 parse('This is 01/06/2011, at 7:57 PM and 01/06/2011, at 19:57 PM')
   
 
    C:\ProgramData\Anaconda3\lib\site-packages\dateutil\parser\_parser.py in parse(timestr, parserinfo, **kwargs)
       1372         return parser(parserinfo).parse(timestr, **kwargs)
       1373     else:
    -> 1374         return DEFAULTPARSER.parse(timestr, **kwargs)
       1375
       1376
   
 
    C:\ProgramData\Anaconda3\lib\site-packages\dateutil\parser\_parser.py in parse(self, timestr, default, ignoretz, tzinfos, **kwargs)
        647
        648         if res is None:
    --> 649             raise ParserError("Unknown string format: %s", timestr)
        650
        651         if len(res) == 0:
   
 
    ParserError: Unknown string format: This is 01/06/2011, at 7:57 PM and 01/06/2011, at 19:57 PM
 
 
## Using `dateparser`
 
I found this package to be more effective than `dateutil`. Please install it using `pip install dateparser`.
 
 
```python
from dateparser.search import search_dates
 
search_dates(eml)
```
 
 
 
 
    [('Wed, Jun 1, 2011, 9:39 PM', datetime.datetime(2011, 6, 1, 21, 39)),
     ('On 01/06/2011, at 7:57 PM', datetime.datetime(2011, 1, 6, 19, 57)),
     ('On 1 June 2011 13:43', datetime.datetime(2011, 6, 1, 13, 43))]
 
 
 
We can see that it found all the date times. And it also returns in the native python DateTime object. Isn't it awesome?
 
## Using `datefinder`
 
This is another package that can find dates from the text. Please install it using `pip install datefinder`
 
 
```python
!pip install datefinder
```
 
    Collecting datefinder
      Downloading datefinder-0.7.3-py2.py3-none-any.whl (10 kB)
    Requirement already satisfied: pytz in c:\programdata\anaconda3\lib\site-packages (from datefinder) (2020.1)
    Requirement already satisfied: python-dateutil>=2.4.2 in c:\programdata\anaconda3\lib\site-packages (from datefinder) (2.8.1)
    Requirement already satisfied: regex>=2017.02.08 in c:\programdata\anaconda3\lib\site-packages (from datefinder) (2020.10.15)
    Requirement already satisfied: six>=1.5 in c:\programdata\anaconda3\lib\site-packages (from python-dateutil>=2.4.2->datefinder) (1.15.0)
    Installing collected packages: datefinder
    Successfully installed datefinder-0.7.3
   
 
 
```python
from datefinder import find_dates
 
list(find_dates(eml))
```
 
 
 
 
    [datetime.datetime(2011, 6, 1, 21, 39),
     datetime.datetime(2011, 1, 6, 19, 57),
     datetime.datetime(2011, 6, 1, 13, 43)]
 
 
 
This also gets our job done but we are more concerned about the original date format.
 
That's all for now and for my use case, I found `date parser` to be best. What is yours?

For more content like this one, please stay exploring our site or signup for the [newsletter](https://dataqoil.com/newsletter/).



