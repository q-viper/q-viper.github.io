---
title:  "Tweet Scraping with Tweepy"
date:   2022-06-04 09:29:17 +0545
categories:
    - Tweets
    - Data Science
    
tags:
    - Pandas
    - Python
    - tweepy

header:
  teaser: assets/twitter_bot/twitter_app.png
---

Tweet Scraping can be done with different ways but one of the reliable ones is using Tweepy and Tweeter's developer API. In this blog we are going to explore how we can do Tweet Scraping using Twitter's API and Tweepy. The API calls are handled by Tweepy and we only need to give it Keys.
 
 
### Getting API Keys
First we need before doing tweet scraping is to have a Twitter Developer Account and only with it, we can get keys to scrape tweets.
* Visit [developer.twitter.com](https://developer.twitter.com/en/docs/projects/overview). Please read carefully how much request is possible to which type of accounts.
* Go to sign in and fill in the credentials.
* Then create an app.
 
![](https://q-viper.github.io/assets/twitter_bot/bot_name.png)
 
* Generate and save API Keys.
 
![](https://q-viper.github.io/assets/twitter_bot/apikey.png)
 
 
### Installing Tweepy
 
 
```python
!pip install git+https://github.com/tweepy/tweepy.git
```
 
    Collecting git+https://github.com/tweepy/tweepy.git
      Cloning https://github.com/tweepy/tweepy.git to c:\users\dell\appdata\local\temp\pip-req-build-div0g8k4
    Requirement already satisfied: oauthlib<4,>=3.2.0 in c:\users\dell\appdata\roaming\python\python38\site-packages (from tweepy==4.10.0) (3.2.0)
    Requirement already satisfied: requests<3,>=2.27.0 in c:\users\dell\appdata\roaming\python\python38\site-packages (from tweepy==4.10.0) (2.27.1)
    Requirement already satisfied: requests-oauthlib<2,>=1.2.0 in c:\programdata\anaconda3\lib\site-packages (from tweepy==4.10.0) (1.3.0)
    Requirement already satisfied: charset-normalizer~=2.0.0 in c:\users\dell\appdata\roaming\python\python38\site-packages (from requests<3,>=2.27.0->tweepy==4.10.0) (2.0.7)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\programdata\anaconda3\lib\site-packages (from requests<3,>=2.27.0->tweepy==4.10.0) (1.26.4)
    Requirement already satisfied: idna<4,>=2.5 in c:\programdata\anaconda3\lib\site-packages (from requests<3,>=2.27.0->tweepy==4.10.0) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in c:\programdata\anaconda3\lib\site-packages (from requests<3,>=2.27.0->tweepy==4.10.0) (2020.12.5)
    Building wheels for collected packages: tweepy
      Building wheel for tweepy (setup.py): started
      Building wheel for tweepy (setup.py): finished with status 'done'
      Created wheel for tweepy: filename=tweepy-4.10.0-py3-none-any.whl size=95239 sha256=21aef993404498afa5f673cece5e8ee6a9c82b6b89c7b063c963858021692b03
      Stored in directory: C:\Users\Dell\AppData\Local\Temp\pip-ephem-wheel-cache-yi90v1ry\wheels\ad\05\51\a78f66d15b87f9c623d2f3afc4401660ac4219e526c787fb8b
    Successfully built tweepy
    Installing collected packages: tweepy
      Attempting uninstall: tweepy
        Found existing installation: tweepy 4.8.0
        Uninstalling tweepy-4.8.0:
          Successfully uninstalled tweepy-4.8.0
    Successfully installed tweepy-4.10.0
    
 
      Running command git clone -q https://github.com/tweepy/tweepy.git 'C:\Users\Dell\AppData\Local\Temp\pip-req-build-div0g8k4'
    
 
### Preparing Keys
 
 
```python
api_key="api_key here"
secret="secret key here"
bearer="bearer here"
access_token="access_token here"
access_token_secret="access_token_secret here"
```
 
### Import and prepare API Object
Pass api_key, secret, access_token, access_token_secret into the OAuthHandler.
 
 
```python
import tweepy as tw
 
api_key= api_key
api_secret= secret
 
 
auth = tw.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)
 
```
 
### Making First API Call
 
 
```python
key_word="Funny"
max_items=5
for tweet in tw.Cursor(api.search_tweets,
        q=key_word, count=max_items).items(max_items):
    print(tweet.text)
```
 
    RT @SocDoneLeft: still support the Socialist Agenda? https://t.co/qWYlwscPuD
    RT @hiraiboostan: i know it might sound weird but twice really did save me in some ways. they helped me feel less lonely and opened a door…
    RT @memorytrain2012: 下北沢クラブキューに誘っていただきました。近藤さん(ex.PEALOUT)率いるmy funny hitchhiker、知り合いからずっと話を聞いていたthe MADRASとの3マンライブ。大先輩に囲まれて緊張感ありますが突然少年は遠慮…
    I mean...
    
    (https://t.co/cOvCZLjJVs)
    #meme #memes #funny #joke #memebot https://t.co/HEi2bsOytd
    RT @IcyJaime: “are you ok?” nah but l’m funny
    
 
In the above example, we searched for the keyword Funny and took only 5 items.
 
### Storing Tweets as CSV and DataFrame
 
Since we have already made some calls, let's create DataFrame where all the tweets will be stored. 
 
 
```python
import json,csv,time,os
def get_related_tweets(key_words, language="en", max_tweets=50, max_items=10):
    fname=language+str(time.time())+".csv"
    print(f"Filename {fname}")
    
    count=0
    tweets=max_tweets
    
    for key_word in key_words:
        print(f"Current Keyword: {key_word}")
        for tweet in tw.Cursor(api.search_tweets,
                               q=key_word, count=max_items).items(max_items):
            
            tweet_created_at = []
            text = []
            user=[]
            hashtags = []
            user_mentions = []
            in_reply = []
            protected = []
            followers_count = [] 
            friends_count = []
            listed_count = []
            created_at = []
            favourites_count = []
            geo_enabled = []
            verified =[]
            statuses_count=[]
            coordinates=[]
            is_quote_status=[]
            retweet_count=[]
            favorited=[]
            retweeted=[]
            source = []
            place=[]
            lang=[]
            kwd=[]
            ids=[]
            locations=[]
            description=[]
 
            if tweet.lang!=language:
                continue
            count+=1
            try:
              tweet_created_at.append(tweet.created_at)
              status = api.get_status(tweet.id, tweet_mode="extended")
              try:
                  txt = status.retweeted_status.full_text
              except AttributeError:  
                  txt = status.full_text
 
              description.append(tweet.user.description)
              locations.append(tweet.user.location)
              ids.append(tweet.id)
              text.append(txt)
              user.append(tweet.user.screen_name)
              hashtags.append(tweet.entities["hashtags"])
              user_mentions.append(len(tweet.entities["user_mentions"]))
              in_reply.append(tweet.in_reply_to_status_id)
              protected.append(tweet.user.protected)
              followers_count.append(tweet.user.followers_count)
              friends_count.append(tweet.user.friends_count)
              listed_count.append(tweet.user.listed_count)
              created_at.append(tweet.user.created_at)
              favourites_count.append(tweet.user.favourites_count)
              geo_enabled.append(tweet.user.geo_enabled)
              verified.append(tweet.user.verified)
              statuses_count.append(tweet.user.statuses_count)
              coordinates.append(tweet.coordinates)
              is_quote_status.append(tweet.is_quote_status)
              retweet_count.append(tweet.retweet_count)
              favorited.append(tweet.favorited)
              retweeted.append(tweet.retweeted)
              source.append(tweet.source)
              place.append(tweet.place)
              lang.append(tweet.lang)
              kwd.append(key_word)
 
              dict_data={"id":ids,'tweet_created_at':tweet_created_at, 
                                    'text': text, 'user': user, "bio":description,"location":locations,
                                    "hashtags":hashtags, "user_mentions":user_mentions,
                                    "in_reply":in_reply, "protected":protected, "followers_count":followers_count,
                                    "friends_count":friends_count, "listed_count":listed_count, "created_at":created_at,
                                    "favourites_count":favourites_count, "geo_enabled":geo_enabled, "verified":verified,
                                    "statuses_count":statuses_count, "coordinates":coordinates, "is_quote_status":is_quote_status,
                                    "retweet_count":retweet_count,
                                    "retweeted":retweeted,"lang":lang,
                                    "source":source,"place":place,"kwd":key_word}
              csv_columns=list(dict_data.keys())
              dict_data = {k:v[0] for k,v in dict_data.items()}
              if os.path.isfile(fname):  
                # print("File Exists")
                pass
              else:
                # print("File does not exist")
                with open(fname, 'a', encoding='utf-8') as csvfile:
                  writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                  writer.writeheader()
 
              with open(fname, 'a', encoding='utf-8') as csvfile:
                  writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                  for data in [dict_data]:
                      writer.writerow(data)
 
              if count>=tweets:
                break
            except:
              print("Something is wrong. Skipping this tweet.")
    
    return pd.read_csv(fname, parse_dates=["tweet_created_at","created_at"])
```
 
Above function takes keywords, language and number of max_tweets and max_items which will be used in the API. 
* Prepare a file name based on the current timestamp, so that there will be no repeat.
* Prepare count and max tweets number so that we will be getting only max_tweets number of tweets.
* We loop onto the keyword.
* Loop into the results given by Cursor. We pass `api.search_tweets`, which is called by the `Cursor`.
* Prepare lists to hold the important fields like tweet date, account date, user, tweet text and so on.
* If the current language of the tweet is not the language we wanted, then we will skip the current result.
* We append all the important values into their respective lists and finally create a dictionary. 
* Then write that dictionary using `csv.DictWriter`
* Finally read that csv file as a dataframe and return it.
 
 
```python
 
keywords=["climate","funny"]
tweets_df = get_related_tweets(keywords)
tweets_df
```
 
    Filename en1654358920.3398395.csv
    Current Keyword: climate
    Current Keyword: funny
    
 
 
## What's next?
We completed the tweet scraping part but what next?
* Performing EDA based on some topic like COVID, College and so on.
* We could use tweets from tweet scraping to do Sentiment Analysis based on the tweet text.
* Performing Tweet Generation using Some Markov Models.
 
 
For more contents like this, please subscribe to our [news letter](https://dataqoil.com/newsletter/).
 
 

