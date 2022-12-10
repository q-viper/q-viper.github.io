---
title:  "Text Analysis with WordCloud in Python"
date:   2022-12-06 09:29:17
categories:
    - Project
tags:
    - sentiment analysis
    - python
    - WordCloud
header:
  teaser: assets/twitter_bot/output_21_0.png
---
WordCloud in Python can be done in different ways but one of the most popular and easier ones is using the package `wordcloud`. We can install it using the following way.


```python
!pip install wordcloud
```

    Requirement already satisfied: wordcloud in c:\programdata\anaconda3\lib\site-packages (1.8.1)
    Requirement already satisfied: pillow in c:\programdata\anaconda3\lib\site-packages (from wordcloud) (8.0.1)
    Requirement already satisfied: numpy>=1.6.1 in c:\programdata\anaconda3\lib\site-packages (from wordcloud) (1.19.2)
    Requirement already satisfied: matplotlib in c:\users\viper\appdata\roaming\python\python38\site-packages (from wordcloud) (3.5.3)
    Requirement already satisfied: pyparsing>=2.2.1 in c:\users\viper\appdata\roaming\python\python38\site-packages (from matplotlib->wordcloud) (3.0.9)
    Requirement already satisfied: fonttools>=4.22.0 in c:\programdata\anaconda3\lib\site-packages (from matplotlib->wordcloud) (4.37.1)
    Requirement already satisfied: cycler>=0.10 in c:\programdata\anaconda3\lib\site-packages (from matplotlib->wordcloud) (0.10.0)
    Requirement already satisfied: python-dateutil>=2.7 in c:\programdata\anaconda3\lib\site-packages (from matplotlib->wordcloud) (2.8.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\programdata\anaconda3\lib\site-packages (from matplotlib->wordcloud) (1.3.0)
    Requirement already satisfied: packaging>=20.0 in c:\programdata\anaconda3\lib\site-packages (from matplotlib->wordcloud) (20.4)
    Requirement already satisfied: six in c:\programdata\anaconda3\lib\site-packages (from cycler>=0.10->matplotlib->wordcloud) (1.15.0)
    

WordCloud simply is the words scattered in an image and the word's size differs based on different properties. Here in this blog, I will plot a word cloud for based on a tweet created with keywords `['worldcup', 'world cup', 'wcup', 'football', 'qatar worldcup prediction']`. You can read about [how to scrape tweets from this blog](https://dataqoil.com/2022/06/05/scraping-tweets-with-tweepy/) and [how to perform sentiment analysis from this blog](https://dataqoil.com/2022/11/29/worldcup-tweet-sentiment-analysis-in-python/).

## Importing Packages


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
sns.set()
```

## Reading CSV File
Let's read our CSV file containing tweets.


```python
df = pd.read_csv('en1670670038.8448372.csv')
df.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>tweet_created_at</th>
      <th>text</th>
      <th>user</th>
      <th>bio</th>
      <th>location</th>
      <th>hashtags</th>
      <th>user_mentions</th>
      <th>in_reply</th>
      <th>protected</th>
      <th>...</th>
      <th>verified</th>
      <th>statuses_count</th>
      <th>coordinates</th>
      <th>is_quote_status</th>
      <th>retweet_count</th>
      <th>retweeted</th>
      <th>lang</th>
      <th>source</th>
      <th>place</th>
      <th>kwd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1601532325241950209</td>
      <td>2022-12-10 11:00:36+00:00</td>
      <td>@mcbenwell @TheTotallyShow Unsurprisingly, thi...</td>
      <td>DrLouiseClare1</td>
      <td>Historian looking at Argentine, British and US...</td>
      <td>NaN</td>
      <td>[]</td>
      <td>2</td>
      <td>1.601523e+18</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>1187</td>
      <td>NaN</td>
      <td>False</td>
      <td>0</td>
      <td>False</td>
      <td>en</td>
      <td>Twitter for iPhone</td>
      <td>NaN</td>
      <td>w</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1601532314613334018</td>
      <td>2022-12-10 11:00:33+00:00</td>
      <td>All the best to England playing in the quarter...</td>
      <td>bookajet</td>
      <td>Enjoy freedom without responsibility and let y...</td>
      <td>Farnborough Airport</td>
      <td>[{'text': 'worldcup', 'indices': [61, 70]}, {'...</td>
      <td>0</td>
      <td>NaN</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>959</td>
      <td>NaN</td>
      <td>False</td>
      <td>0</td>
      <td>False</td>
      <td>en</td>
      <td>Hootsuite Inc.</td>
      <td>NaN</td>
      <td>w</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1601532312696811520</td>
      <td>2022-12-10 11:00:33+00:00</td>
      <td>1/It's Matchday⚽️\r\n\r\nShow support to your ...</td>
      <td>0xNeverWinn</td>
      <td>@GaHunter688 suspended</td>
      <td>NaN</td>
      <td>[{'text': 'worldcup', 'indices': [61, 70]}, {'...</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>8442</td>
      <td>NaN</td>
      <td>False</td>
      <td>4472</td>
      <td>False</td>
      <td>en</td>
      <td>Twitter Web App</td>
      <td>NaN</td>
      <td>w</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1601532303762677762</td>
      <td>2022-12-10 11:00:30+00:00</td>
      <td>Good Luck England ⚽⚽⚽\r\n #Itscominghome #Worl...</td>
      <td>3LionsOnMaShirt</td>
      <td>Sharing the latest #ThreeLions news and fan ta...</td>
      <td>Manchester, England</td>
      <td>[{'text': 'Itscominghome', 'indices': [40, 54]...</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>157</td>
      <td>NaN</td>
      <td>False</td>
      <td>1</td>
      <td>False</td>
      <td>en</td>
      <td>VillaBotMan</td>
      <td>NaN</td>
      <td>w</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1601532286507552768</td>
      <td>2022-12-10 11:00:26+00:00</td>
      <td>Guess the Quarter Final Winners ⚽️🥂\r\n\r\nThe...</td>
      <td>EdehRonald</td>
      <td>crypto enthusiast/trader</td>
      <td>NaN</td>
      <td>[{'text': 'WorldcupQatar2022', 'indices': [64,...</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>92</td>
      <td>NaN</td>
      <td>False</td>
      <td>24</td>
      <td>False</td>
      <td>en</td>
      <td>Twitter for Android</td>
      <td>NaN</td>
      <td>w</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>



Two columns, text and bio can be used to plot a word cloud.

## Simple WordCloud
Let's plot a simple WordCloud of the following text:


```python
txt = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla fringilla ex nec massa sollicitudin, et condimentum mi vehicula. Integer enim urna, pellentesque a augue sed, malesuada ornare enim. Integer at ullamcorper tellus. Cras condimentum orci ac enim egestas, nec elementum dolor varius. Vestibulum molestie magna vel sapien tristique dictum. Nam auctor vitae enim vitae lacinia. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus mollis in est vitae dictum. Duis et mauris dui. Etiam aliquam in leo vitae placerat. Cras tincidunt neque id lectus tincidunt accumsan. Donec ut dignissim mi, at consequat elit.

Suspendisse vel vestibulum lorem, vel aliquam justo. Praesent hendrerit, est et lobortis condimentum, elit augue bibendum velit, sed volutpat purus tortor maximus nisi. In sed volutpat lectus. Aenean at turpis vel nisl egestas mollis at sit amet dolor. Nullam semper dapibus orci, facilisis tempor nisl volutpat consectetur. Curabitur elit est, vehicula venenatis interdum at, suscipit et magna. Vestibulum a pretium felis. Curabitur tristique euismod laoreet. Aliquam erat volutpat. Sed luctus nulla sed posuere mattis. Vivamus ligula turpis, sollicitudin non rutrum non, consequat sodales diam. Donec dapibus nec ligula eu tincidunt. Maecenas risus massa, malesuada eu lorem a, fringilla imperdiet leo.
"""
wc = WordCloud(max_words=500, width=1000, height=500)
wcimg=wc.generate(txt)
plt.figure(figsize=(15,10))
plt.imshow(wcimg)
plt.title('WordCloud Test')
plt.show()
```


    
![png]({{site.url}}/assets/twitter_bot/output_9_0.png)
    


WordCloud accepts some parameters which can be found in docstring as well.


```python
txt = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla fringilla ex nec massa sollicitudin, et condimentum mi vehicula. Integer enim urna, pellentesque a augue sed, malesuada ornare enim. Integer at ullamcorper tellus. Cras condimentum orci ac enim egestas, nec elementum dolor varius. Vestibulum molestie magna vel sapien tristique dictum. Nam auctor vitae enim vitae lacinia. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus mollis in est vitae dictum. Duis et mauris dui. Etiam aliquam in leo vitae placerat. Cras tincidunt neque id lectus tincidunt accumsan. Donec ut dignissim mi, at consequat elit.

Suspendisse vel vestibulum lorem, vel aliquam justo. Praesent hendrerit, est et lobortis condimentum, elit augue bibendum velit, sed volutpat purus tortor maximus nisi. In sed volutpat lectus. Aenean at turpis vel nisl egestas mollis at sit amet dolor. Nullam semper dapibus orci, facilisis tempor nisl volutpat consectetur. Curabitur elit est, vehicula venenatis interdum at, suscipit et magna. Vestibulum a pretium felis. Curabitur tristique euismod laoreet. Aliquam erat volutpat. Sed luctus nulla sed posuere mattis. Vivamus ligula turpis, sollicitudin non rutrum non, consequat sodales diam. Donec dapibus nec ligula eu tincidunt. Maecenas risus massa, malesuada eu lorem a, fringilla imperdiet leo.
"""
wc = WordCloud(max_words=500, width=1000, height=500, background_color='red')
wcimg=wc.generate(txt)
plt.figure(figsize=(15,10))
plt.imshow(wcimg)
plt.title('WordCloud Test')
plt.show()
```


    
![png]({{site.url}}/assets/twitter_bot/output_11_0.png)
    


## WordCloud of Bio

We will plot WordCloud of only those bios where there is actually a text!


```python
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
                            collocations=False).generate(" ".join(df[df.bio.isna()==False].bio))

plt.figure(figsize=(15,10))
plt.imshow(wc)
plt.title('WordCloud of Bio')
plt.show()
    
```


    
![png]({{site.url}}/assets/twitter_bot/output_13_0.png)
    


Looking over the above WordCloud, we can see that the word `https` is also there, it is a noise and we need to clear it.

## Clearing Noise


```python
def remove_noise(tweet):
        '''
        To remove noise
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
                            collocations=False).generate(" ".join(df[df.bio.isna()==False].bio.apply(remove_noise)))

plt.figure(figsize=(15,10))
plt.imshow(wc)
plt.title('WordCloud of Bio')
plt.show()
       
```


    
![png]({{site.url}}/assets/twitter_bot/output_16_0.png)
    


It worked!

## WordCloud of Tweet Text
Let's plot WordCloud in Python for our tweet's text.

```python
def remove_noise(tweet):
        '''
        To remove noise
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
                            collocations=False).generate(" ".join(df[df.text.isna()==False].text.apply(remove_noise)))

plt.figure(figsize=(15,10))
plt.imshow(wc)
plt.title('WordCloud of Text')
plt.show()
       
```


    
![png]({{site.url}}/assets/twitter_bot/output_19_0.png)
    


It's obvious that our most repeated word is the world cup!

And we could simply plot like below.


```python
wc.to_image()
```




    
![png]({{site.url}}/assets/twitter_bot/output_21_0.png)
    



If we want to plot wordcloud in python but with different font of the text, we can pass the font file and plot as well.


