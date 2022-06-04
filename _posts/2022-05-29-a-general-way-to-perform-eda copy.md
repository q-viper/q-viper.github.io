---
title:  "Scraping Tweets with Tweepy"
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

## Tweets Scraping using Tweepy
Hello and welcome back everyone, in this blog we are going to explore how we can scrape tweets using Twitter's API and Tweepy. The API calls are handled by Tweepy and we only need to give it Keys.

### Getting API Keys
First we need to have a Twitter Developer Account and only with it, we can get keys to scrape tweets.
* Visit [developer.twitter.com](https://developer.twitter.com/en/docs/projects/overview). Please read carefully how much request is possible to which type of accounts.
* Go to sign in and fill in the credentials.
* Then create an app.

![]({{site.url}}/assets/twitter_bot/tweet_scraping/bot_name.png)

* Generate and save API Keys.

![]({{site.url}}/assets/twitter_bot/tweet_scraping/apikey.png)


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
    

In above example, we searched for the keyword Funny and took only 5 items.

### Storing Tweets as CSV and DataFrame

Since we have already made some calls, lets create DataFrame where all the tweets will be stored. 


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

Above function takes keywords, language and number of max_tweets and max_items which will be used in API. 
* Prepare a file name based on current timestamp, so that there will be no repeat.
* Prepare count and max tweets number so that we will be getting only max_tweets number of tweets.
* We loop onto keyword.
* Loop into the results given by Cursor. We pass `api.search_tweets`, which is called by the `Cursor`.
* Prepare lists to hold the important fields like tweet date, account date, user, tweet text and so on.
* If current language of the tweet is not language we wanted, then we will skip the current result.
* We append all the important values into their respective lists and the finally create a dictionary. 
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
    




<div><div id=5ac3434b-fba6-4b09-8a10-4531f3212092 style="display:none; background-color:#9D6CFF; color:white; width:200px; height:30px; padding-left:5px; border-radius:4px; flex-direction:row; justify-content:space-around; align-items:center;" onmouseover="this.style.backgroundColor='#BA9BF8'" onmouseout="this.style.backgroundColor='#9D6CFF'" onclick="window.commands?.execute('create-mitosheet-from-dataframe-output');">See Full Dataframe in Mito</div> <script> if (window.commands.hasCommand('create-mitosheet-from-dataframe-output')) document.getElementById('5ac3434b-fba6-4b09-8a10-4531f3212092').style.display = 'flex' </script> <table border="1" class="dataframe">
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
      <td>1533118562847670279</td>
      <td>2022-06-04 16:08:44+00:00</td>
      <td>Households and businesses rely on the Met Office to give good weather advice. With climate change, this becomes even more important.\r\n\r\nCutting the Met Office would be an absurd and ridiculous decision. https://t.co/kQOnGIQwhQ</td>
      <td>RobertCHeale</td>
      <td>Local Man in the Brighton and Hove area who is practical and caring. Interested in sport, current affairs and music. Generally safe, sound and steady!</td>
      <td>NaN</td>
      <td>[]</td>
      <td>0</td>
      <td>NaN</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>38237</td>
      <td>NaN</td>
      <td>True</td>
      <td>0</td>
      <td>False</td>
      <td>en</td>
      <td>Twitter Web App</td>
      <td>NaN</td>
      <td>c</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1533118548859572230</td>
      <td>2022-06-04 16:08:40+00:00</td>
      <td>So read my last 5 tweets.\r\nThat's my outline for trading stocks these days.\r\n\r\n If you think you have the skills and knowledge to be successful in this volatile trading climate then go right ahead.</td>
      <td>MasterBJones</td>
      <td>MASTER STOCK TRADER/100% FREE!\r\nDisclaimer: My tweets are 100% ONLY for educational/entertainment purposes. NOT a licenced financial professional. NOT Advice.</td>
      <td>NaN</td>
      <td>[]</td>
      <td>0</td>
      <td>NaN</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>54293</td>
      <td>NaN</td>
      <td>False</td>
      <td>0</td>
      <td>False</td>
      <td>en</td>
      <td>Twitter Web App</td>
      <td>NaN</td>
      <td>c</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1533118545495658496</td>
      <td>2022-06-04 16:08:39+00:00</td>
      <td>New York's Democratic supermajority just refused to pass climate policy for the 3rd year in a row. National solar industry lobbyists teamed up with fossil fueled merchant power producers to kill a bill because they were scared of losing market share to  public renewables</td>
      <td>asgardwalathor</td>
      <td>I used to rule Asgard, but Ragnarok happened.\r\nNow I am jobless, so I tweet on Indian politics.</td>
      <td>Asgard</td>
      <td>[]</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>7004</td>
      <td>NaN</td>
      <td>False</td>
      <td>128</td>
      <td>False</td>
      <td>en</td>
      <td>Twitter Web App</td>
      <td>NaN</td>
      <td>c</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1533118545143287808</td>
      <td>2022-06-04 16:08:39+00:00</td>
      <td>Nothing is more backwards than Leftists policies… https://t.co/hgKsT0vvGa</td>
      <td>ijaredtaylor</td>
      <td>Investing in our youth and community....</td>
      <td>Gilbert, Arizona</td>
      <td>[]</td>
      <td>0</td>
      <td>NaN</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>1519</td>
      <td>NaN</td>
      <td>False</td>
      <td>0</td>
      <td>False</td>
      <td>en</td>
      <td>Twitter for iPad</td>
      <td>NaN</td>
      <td>c</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1533118544208076800</td>
      <td>2022-06-04 16:08:39+00:00</td>
      <td>For 50 years, governments have failed to act on climate change. No more excuses | Christiana Figueres et al https://t.co/aActtKN7M1 https://t.co/EQw4tY3cOl</td>
      <td>rubyjean72802</td>
      <td>BLUE to the bone. Nana to 3 beautiful grandchildren who deserve the opportunity to live in a truly free country ruled by its citizens rather than the elite few.</td>
      <td>NaN</td>
      <td>[]</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>26375</td>
      <td>NaN</td>
      <td>False</td>
      <td>13</td>
      <td>False</td>
      <td>en</td>
      <td>Twitter Web App</td>
      <td>NaN</td>
      <td>c</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1533118576986574851</td>
      <td>2022-06-04 16:08:47+00:00</td>
      <td>@rebeccahchase I know. Occasionally can be funny lol X. ❤️.</td>
      <td>bathb0y</td>
      <td>Name's Justin. Bsc (Hons) Degree in Social Policy and Criminology with the Open University. Loves Cricket, Rugby Union, Horse Racing mainly. Also love Yoga!</td>
      <td>Bath/Chippenham, UK</td>
      <td>[]</td>
      <td>1</td>
      <td>1.533118e+18</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>62166</td>
      <td>NaN</td>
      <td>False</td>
      <td>0</td>
      <td>False</td>
      <td>en</td>
      <td>Twitter for Android</td>
      <td>NaN</td>
      <td>f</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1533118576000790529</td>
      <td>2022-06-04 16:08:47+00:00</td>
      <td>OP "I got scolded by kpopapi today"\r\n💎appa i'm really late right?\r\n🐢i told you to be early, didn't i?!\r\n\r\n🐢are you studying hard?\r\n💎but i'm an office worker\r\nVernon suddenly back to being an idol instead of his dad act. Op found it v funny.\r\n🐢ah i thought you were a student</td>
      <td>17_dlwlrma</td>
      <td>@pledis_17 @_iuofficial</td>
      <td>NaN</td>
      <td>[]</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>300573</td>
      <td>NaN</td>
      <td>False</td>
      <td>294</td>
      <td>False</td>
      <td>en</td>
      <td>Twitter for iPhone</td>
      <td>NaN</td>
      <td>f</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1533118568358985729</td>
      <td>2022-06-04 16:08:45+00:00</td>
      <td>@Accraaaaaa_ Brother, you’re indeed funny</td>
      <td>OforiSah1</td>
      <td>I Love Reggae.</td>
      <td>Kpone Shanghai-High Tension</td>
      <td>[]</td>
      <td>1</td>
      <td>1.532977e+18</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>290</td>
      <td>NaN</td>
      <td>False</td>
      <td>0</td>
      <td>False</td>
      <td>en</td>
      <td>Twitter for iPhone</td>
      <td>NaN</td>
      <td>f</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1533118567486345217</td>
      <td>2022-06-04 16:08:45+00:00</td>
      <td>Twitter is so funny \r\nThe impact and reach of JD  is far more than any character designed by Loki till date : Wallpapers Whatsapp statuses ellathalyum JD irkar even today.\r\nLoki will soon give another massy character in #Thalapathy67 💥 https://t.co/Fhqmar2gi4</td>
      <td>Manoj_vj_</td>
      <td>ÄrÐêñ† †hålåþå†h¥ £åñ//SuRiYa // #Beast // Thala MSDian🔥/💪\r\nMech boy</td>
      <td>murugagoundampalayam, Namakkal</td>
      <td>[]</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>217497</td>
      <td>NaN</td>
      <td>False</td>
      <td>45</td>
      <td>False</td>
      <td>en</td>
      <td>Twitter for Android</td>
      <td>NaN</td>
      <td>f</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1533118567108911106</td>
      <td>2022-06-04 16:08:45+00:00</td>
      <td>jungkook's reaction when he remembered they had a song titled 'heartbeat' is still so funny like😭 https://t.co/jGhLLYP5Hg</td>
      <td>ning061313</td>
      <td>마리즈</td>
      <td>manila</td>
      <td>[]</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>239433</td>
      <td>NaN</td>
      <td>False</td>
      <td>2784</td>
      <td>False</td>
      <td>en</td>
      <td>Twitter for iPhone</td>
      <td>NaN</td>
      <td>f</td>
    </tr>
  </tbody>
</table></div>



## Whats next?
* Performing EDA based on some topic like COVID, College and so on.
* Performing Sentiment Analysis based on the tweet text.
* Performing Tweet Generation using Some Markov Models.


```python

```
