---
title:  "Making People Finder in 2022 Using BeautifulSoup"
date:   2022-07-24 09:29:17 +0545
categories:
    - urllib3
    - BeautifulSoup
tags:
    - scraping
    - people finding
header:
  teaser: assets/people_finder/github_search.png
---

## Introduction

Hello and welcome back everyone, in this part of the blog I am going to share how can we create our own people finder tool using BeautifulSoup and Python. 

### Why do we need people finder?
There are lots of benefits of having easier way to find the person. And this might be the best thing for HR companies. Having the list of professionals and their public profile based on their expertise and the experience is one of the rich data. So lets try to make one such data for ourselves.

### How will we do it?
We will automate the search in Search Engine and some search portal like GitHub and then store the result in dataframe then into file. First, we will do google search to find the linkedin profile based on the keyword.


### Installing Dependencies
For this purpose, we are going to use BeautifulSoup, a python library.


```python
!pip install beautifulsoup4
```

## Importing Libraries


```python
import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup as BS
import time
import bs4
import warnings
warnings.filterwarnings("ignore")

```

There are bunch of libraries we will use:
* Pandas to make dataframe later on.
* Requests to make HTTP requests.
* Urllib3 to make manager and headers.
* BeautifulSoup to scrape and search over the page.
* Time to show scrape time.
* Warning to supress the warnings.

## Google Search

Lets use google search to make our first search. Head over to the Google.com and make a first search, `linkedin google engineer`.

![]({{site.url}}/assets/people_finder/google.png)

Whenever we search something in Google, it takes our query into `https://google.com/search?q=` and shows the list of results. But mostly the results are location based.


```python

query="linkedin google engineer"
url = f"https://google.com/search?q={query}"

http = urllib3.PoolManager()
http.addheaders = [('User-agent', 'Mozilla/61.0')]
# web_page = http.request('GET',url)
web_page=requests.get(url)
soup = BS(web_page.content, 'html5lib')
# soup
```

### Getting All URLs

In the search result, there will be a lot of links and we only need links at this moment. So lets find all links using the element `a`.


```python
urls = soup.find_all("a")
# urls
```

### Getting only URLs that will be relevent

There will be lots of other links which will not be relevant to us at this moment. For example the Google's Sign In page or Privacy Policy so lets do simple check. We will put the name, url of the profile and then role in a dictionary.


```python
profiles = {"names":[],"urls":[],"roles":[]}

for url in urls:
    href = url.get("href")
    
    if "/url?q=" in href and "linkedin" in href and \
        "accounts.google.com" not in href and "policies.google.com" not in href and "linkedin.com/in" in href:
        nhref=href.split("=")[1].split("&")[0]
        
        print(url.text, nhref)
        
        profiles["names"].append(url.text.split("-")[0])
        profiles["roles"].append(url.text.split("-")[1])
        profiles["urls"].append(nhref)
        
        
```

    Akshay Miterani - Software Engineer - Google - LinkedInin.linkedin.com › akshay-mite... https://in.linkedin.com/in/akshay-miterani-108827105
    David Garry - Software Engineer - Google | LinkedInwww.linkedin.com › davidgar... https://www.linkedin.com/in/davidgarry1
    Betty Chen - Software Engineer - Google | LinkedInwww.linkedin.com › bettyjxch... https://www.linkedin.com/in/bettyjxchen
    Risab Manandhar - Software Engineer - Google - LinkedInwww.linkedin.com › risab-ma... https://www.linkedin.com/in/risab-manandhar
    Hai Bi - Software Engineer - Google | LinkedInwww.linkedin.com › ... https://www.linkedin.com/in/hai-bi-b6a10010
    Sabbir Yousuf Sanny - Software Engineer - Google | LinkedInwww.linkedin.com › ... https://www.linkedin.com/in/sabbir-yousuf-sanny-11aa7a21
    Delia Lazarescu - Software Engineer - Google - LinkedInca.linkedin.com › delialazarescu https://ca.linkedin.com/in/delialazarescu
    Shailee Patel - Software Engineer - Google | LinkedInwww.linkedin.com › shailee26 https://www.linkedin.com/in/shailee26
    Sahil Gaba - Software Engineer - Google | LinkedInwww.linkedin.com › gabag26 https://www.linkedin.com/in/gabag26
    

### Dataframe of the results
Dataframes are easy to do data analysis works in Pandas. So lets make one out of above dictionary.


```python
pd.DataFrame(profiles)
```




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
      <th>names</th>
      <th>urls</th>
      <th>roles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Akshay Miterani</td>
      <td>https://in.linkedin.com/in/akshay-miterani-108...</td>
      <td>Software Engineer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>David Garry</td>
      <td>https://www.linkedin.com/in/davidgarry1</td>
      <td>Software Engineer</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Betty Chen</td>
      <td>https://www.linkedin.com/in/bettyjxchen</td>
      <td>Software Engineer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Risab Manandhar</td>
      <td>https://www.linkedin.com/in/risab-manandhar</td>
      <td>Software Engineer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hai Bi</td>
      <td>https://www.linkedin.com/in/hai-bi-b6a10010</td>
      <td>Software Engineer</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sabbir Yousuf Sanny</td>
      <td>https://www.linkedin.com/in/sabbir-yousuf-sann...</td>
      <td>Software Engineer</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Delia Lazarescu</td>
      <td>https://ca.linkedin.com/in/delialazarescu</td>
      <td>Software Engineer</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Shailee Patel</td>
      <td>https://www.linkedin.com/in/shailee26</td>
      <td>Software Engineer</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sahil Gaba</td>
      <td>https://www.linkedin.com/in/gabag26</td>
      <td>Software Engineer</td>
    </tr>
  </tbody>
</table>
</div>



### Pros and Cons of Using Google Search
* Its easier to make GET requests however it might ask us for security check if made too many request and at that time scraping fails.
* It is easier to find people as Google's Crawlers already have list of results based on our query and thus we only have to do very little to find right information. But it might be tough to get information by visiting LinkedIn profile.
* With very little luck, we could visit the person's LinkedIn profile without having to login. So relying in Google Search to find LinkedIn Profile is not much fruitful.


## GitHub Search

In the above part, we scraped some of the LinkedIn profiles from the Google Search but we were unable to get portfolio of people. It is quite common among the tech people to have a portfolio and GitHub account. Lets use GitHub's Search to find people based on the keyword. Most people often put their location, company they work for, twitter handle and the portfolio in the GitHub Profile and we are willing to scrape those.

The URL to get result is `https://github.com/search?q=[QUERY]&type=users&p=[PAGE]`. Where QUERY is the query we will search for, type is user and the p for page.


```python

query="google engineer"
url = f"https://github.com/search?q={query}&type=users"

print(f"URL : {url}")

http = urllib3.PoolManager()
http.addheaders = [('User-agent', 'Mozilla/61.0')]
# web_page = http.request('GET',url)
web_page=requests.get(url)
soup = BS(web_page.content, 'html5lib')

pages = soup.find_all("em", class_="current")[0].get("data-total-pages")

max_page = 5

pages
```

    URL : https://github.com/search?q=google engineer&type=users
    




    '100'





In above result, we did GET request and received a webpage and upon Inspecting the page, we can see the Elements. From Elements we can find the elements like `dev`, `a` and so on where our desired information will be. Like that, we searched for `em` with class as `current` and it have a `data-total-pages` attribute in it. Upon doing get, one can get the value of it. It seems that there are 100 pages with results.

![]({{site.url}}/assets/people_finder/github_search.png)

Now we will loop over to those pages to get the information of the user like name and URL of profile.


```python

github_profiles = {"name":[], "urls":[]}

if pages:
    pages=int(pages)
    print(f"Total Pages: {pages}")
    
    for page in range(1,pages):
        if page>max_page:
            break
        url = f"https://github.com/search?q={query}&type=users&p={page}"
        print(f"\n Current URL: {url} \n")
        
        http = urllib3.PoolManager()
        http.addheaders = [('User-agent', 'Mozilla/61.0')]
        
        web_page=requests.get(url)
        soup = BS(web_page.content, 'html5lib')

        for a in soup.find_all("a",class_="mr-1"):

            gurl = "https://github.com/"+a.get("href")
            gname = a.text

            print(gname, gurl)

            github_profiles["name"].append(gname)
            github_profiles["urls"].append(gurl)
```

    Total Pages: 100
    
     Current URL: https://github.com/search?q=google engineer&type=users&p=1 
    
    Seth Vargo https://github.com//sethvargo
    Kevin Naughton Jr. https://github.com//kdn251
    Miguel Ángel Durán https://github.com//midudev
    Jose Alcérreca https://github.com//JoseAlcerreca
    Shubham Mathur https://github.com//googleknight
    Dan Field https://github.com//dnfield
    Nick Bourdakos https://github.com//bourdakos1
    Mark https://github.com//MarkEdmondson1234
    
     Current URL: https://github.com/search?q=google engineer&type=users&p=2 
    
    Pierfrancesco Soffritti https://github.com//PierfrancescoSoffritti
    Parker Moore https://github.com//parkr
    Gokmen Goksel https://github.com//gokmen
    Justin Poehnelt https://github.com//jpoehnelt
    Sanket Singh https://github.com//singhsanket143
    Shanqing Cai https://github.com//caisq
    Adam Silverstein https://github.com//adamsilverstein
    Mizux https://github.com//Mizux
    Valerii Iatsko https://github.com//viatsko
    Zulkarnine Mahmud https://github.com//zulkarnine
    
     Current URL: https://github.com/search?q=google engineer&type=users&p=3 
    
    Yacine Rezgui https://github.com//yrezgui
    Zulkarnine Mahmud https://github.com//zulkarnine
    Google https://github.com//Google987
    Prateek Narang https://github.com//prateek27
    Rakina Zata Amni https://github.com//rakina
    Sriram Sundarraj https://github.com//ssundarraj
    Irene Ros https://github.com//iros
    Clément Mihailescu https://github.com//clementmihailescu
    Márton Braun https://github.com//zsmb13
    Andrey Kulikov https://github.com//andkulikov
    
     Current URL: https://github.com/search?q=google engineer&type=users&p=4 
    
    Kate Lovett https://github.com//Piinks
    Faisal Abid https://github.com//FaisalAbid
    Viktor Turskyi https://github.com//koorchik
    Milad Naseri https://github.com//mmnaseri
    Rahul Ravikumar https://github.com//tikurahul
    Robert Kubis https://github.com//hostirosti
    Corey Lynch https://github.com//coreylynch
    Emma Twersky https://github.com//twerske
    Shivam Goyal https://github.com//ShivamGoyal1899
    Abhinay Omkar https://github.com//abhiomkar
    
     Current URL: https://github.com/search?q=google engineer&type=users&p=5 
    
    Marek Siarkowicz https://github.com//serathius
    Mais Alheraki https://github.com//pr-Mais
    Abhinay Omkar https://github.com//abhiomkar
    Imaculate https://github.com//imaculate
    Zhixun Tan https://github.com//phisiart
    Jafer Haider https://github.com//itsjafer
    Christie Wilson https://github.com//bobcatfish
    Jason Feinstein https://github.com//jasonwyatt
    Ryan Sepassi https://github.com//rsepassi
    Nick Rout https://github.com//ricknout
    

In above example, we looped for 5 pages and we have stored those info in dictionary `github_profiles`.

Now we will open the profile of a person and extract information like Twitter Handle and Portfolio.
Lets select a last profile url.


```python
gurl
```




    'https://github.com//ricknout'



Lets visit that url from BS4.


```python

http = urllib3.PoolManager()
http.addheaders = [('User-agent', 'Mozilla/61.0')]
# web_page = http.request('GET',url)
web_page=requests.get(gurl)
soup = BS(web_page.content, 'html5lib')
# soup
```

Just like previous time, we should look for the class that holds our information. For headline we can do like below.


```python
headline = soup.find_all("div",class_="p-note user-profile-bio mb-3 js-user-profile-bio f4")[0].text
headline
```




    'Android Developer Relations Engineer at Google 🇿🇦'



For Followers and Following Counts we can do something like below.


```python
followers = soup.find_all("a",class_="Link--secondary no-underline no-wrap")[0].text.strip().split("\n")[0]
following = soup.find_all("a",class_="Link--secondary no-underline no-wrap")[1].text.strip().split("\n")[0]
followers,following
```




    ('510', '29')



For the information like Twitter handle and Portfolio URL we can do something like below.


```python
vcard = soup.find_all("ul",class_="vcard-details")[0].text
vcard = [v.strip() for v in vcard.strip().split("\n") if len(v.strip())>0]

vcard
```




    ['@google', 'Cape Town, South Africa', 'ricknout.dev', 'Twitter', '@ricknout']



But more easily, we can find these information using Itemprop attribute assigned.


```python
vcard = soup.find_all("ul",class_="vcard-details")[0]

portfolio=None
home=None
work=None
twitter=None
for vc in vcard.find_all("li"):
    item=vc.get("itemprop")
    if item=="url":
        portfolio=vc.text.strip()
    if item=="homeLocation":
        home=vc.text.strip()
    if item=="worksFor":
        work=vc.text.strip()
    if item=="twitter":
        twitter=vc.text.strip()

portfolio,home,work,twitter
```




    ('ricknout.dev',
     'Cape Town, South Africa',
     '@google',
     'Twitter\n\n      @ricknout')



Now let combine above codes to work as a whole.


```python

query="google engineer"
url = f"https://github.com/search?q={query}&type=users"

print(f"URL : {url}")

http = urllib3.PoolManager()
http.addheaders = [('User-agent', 'Mozilla/61.0')]
# web_page = http.request('GET',url)
web_page=requests.get(url)
soup = BS(web_page.content, 'html5lib')

pages = soup.find_all("em", class_="current")[0].get("data-total-pages")

max_page = 5

github_profiles = {"name":[], "urls":[], "portfolio":[],"headline":[],
                   "followers":[],"following":[],
                   "home":[], "work":[], "twitter":[]}

if pages:
    pages=int(pages)
    print(f"Total Pages: {pages}. Running upto {max_page}.")
    
    for page in range(1,pages):
        if page>max_page:
            break
        url = f"https://github.com/search?q={query}&type=users&p={page}"
        print(f"\n Current URL: {url} \n")
        
        http = urllib3.PoolManager()
        http.addheaders = [('User-agent', 'Mozilla/61.0')]
        
        web_page=requests.get(url)
        osoup = BS(web_page.content, 'html5lib')

        for a in osoup.find_all("a",class_="mr-1"):

            gurl = "https://github.com/"+a.get("href")
            gname = a.text

            print(f"Got: {gname}, {gurl}")

            github_profiles["name"].append(gname)
            github_profiles["urls"].append(gurl)
            
            
            http = urllib3.PoolManager()
            http.addheaders = [('User-agent', 'Mozilla/61.0')]
            web_page=requests.get(gurl)
            soup = BS(web_page.content, 'html5lib')
            
            headline = soup.find_all("div",class_="p-note user-profile-bio mb-3 js-user-profile-bio f4")[0].text
            
            github_profiles["headline"].append(headline)
            
            followers = soup.find_all("a",class_="Link--secondary no-underline no-wrap")[0].text.strip().split("\n")[0]
            following = soup.find_all("a",class_="Link--secondary no-underline no-wrap")[1].text.strip().split("\n")[0]
            
            github_profiles["followers"].append(followers)
            github_profiles["following"].append(following)
            
            vcard = soup.find_all("ul",class_="vcard-details")[0]

            portfolio=None
            home=None
            work=None
            twitter=None
            for vc in vcard.find_all("li"):
                item=vc.get("itemprop")
                if item=="url":
                    portfolio=vc.text.strip()
                if item=="homeLocation":
                    home=vc.text.strip()
                if item=="worksFor":
                    work=vc.text.strip()
                if item=="twitter":
                    twitter=vc.text.strip().split("\n")[-1].strip()
            
            github_profiles["portfolio"].append(portfolio)
            github_profiles["home"].append(home)
            github_profiles["work"].append(work)
            github_profiles["twitter"].append(twitter)
            
            


            
            
            
```

    URL : https://github.com/search?q=google engineer&type=users
    Total Pages: 100. Running upto 5.
    
     Current URL: https://github.com/search?q=google engineer&type=users&p=1 
    
    Got: Seth Vargo, https://github.com//sethvargo
    Got: Kevin Naughton Jr., https://github.com//kdn251
    Got: Miguel Ángel Durán, https://github.com//midudev
    Got: Jose Alcérreca, https://github.com//JoseAlcerreca
    Got: Shubham Mathur, https://github.com//googleknight
    Got: Nick Bourdakos, https://github.com//bourdakos1
    Got: Mark, https://github.com//MarkEdmondson1234
    Got: Dan Field, https://github.com//dnfield
    Got: Pierfrancesco Soffritti, https://github.com//PierfrancescoSoffritti
    
     Current URL: https://github.com/search?q=google engineer&type=users&p=2 
    
    Got: Parker Moore, https://github.com//parkr
    Got: Gokmen Goksel, https://github.com//gokmen
    Got: Sanket Singh, https://github.com//singhsanket143
    Got: Justin Poehnelt, https://github.com//jpoehnelt
    Got: Shanqing Cai, https://github.com//caisq
    Got: Valerii Iatsko, https://github.com//viatsko
    Got: Mizux, https://github.com//Mizux
    Got: Gabriela D'Ávila Ferrara, https://github.com//gabidavila
    Got: Zulkarnine Mahmud, https://github.com//zulkarnine
    
     Current URL: https://github.com/search?q=google engineer&type=users&p=3 
    
    Got: Adam Silverstein, https://github.com//adamsilverstein
    Got: Yacine Rezgui, https://github.com//yrezgui
    Got: Google, https://github.com//Google987
    Got: Prateek Narang, https://github.com//prateek27
    Got: Rakina Zata Amni, https://github.com//rakina
    Got: Clément Mihailescu, https://github.com//clementmihailescu
    Got: Faisal Abid, https://github.com//FaisalAbid
    Got: Kate Lovett, https://github.com//Piinks
    Got: Andrey Kulikov, https://github.com//andkulikov
    Got: Márton Braun, https://github.com//zsmb13
    
     Current URL: https://github.com/search?q=google engineer&type=users&p=4 
    
    Got: Kate Lovett, https://github.com//Piinks
    Got: Faisal Abid, https://github.com//FaisalAbid
    Got: Viktor Turskyi, https://github.com//koorchik
    Got: Milad Naseri, https://github.com//mmnaseri
    Got: Rahul Ravikumar, https://github.com//tikurahul
    Got: Robert Kubis, https://github.com//hostirosti
    Got: Corey Lynch, https://github.com//coreylynch
    Got: Emma Twersky, https://github.com//twerske
    Got: Shivam Goyal, https://github.com//ShivamGoyal1899
    Got: Abhinay Omkar, https://github.com//abhiomkar
    
     Current URL: https://github.com/search?q=google engineer&type=users&p=5 
    
    Got: Marek Siarkowicz, https://github.com//serathius
    Got: Greg Spencer, https://github.com//gspencergoog
    Got: Mais Alheraki, https://github.com//pr-Mais
    Got: Jafer Haider, https://github.com//itsjafer
    Got: Zhixun Tan, https://github.com//phisiart
    Got: Imaculate, https://github.com//imaculate
    Got: Christie Wilson, https://github.com//bobcatfish
    Got: Jason Feinstein, https://github.com//jasonwyatt
    Got: Nick Rout, https://github.com//ricknout
    Got: Joe Stanton, https://github.com//JoeStanton
    

### Turn Result into Dataframe
In order to do data analysis, it is easier to work with tabular data. So lets convert our above dictionary into dataframe.


```python
df = pd.DataFrame(github_profiles)
df
```




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
      <th>name</th>
      <th>urls</th>
      <th>portfolio</th>
      <th>headline</th>
      <th>followers</th>
      <th>following</th>
      <th>home</th>
      <th>work</th>
      <th>twitter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Seth Vargo</td>
      <td>https://github.com//sethvargo</td>
      <td>https://www.sethvargo.com</td>
      <td>Engineer @google</td>
      <td>3.2k</td>
      <td>5</td>
      <td>Pittsburgh, PA</td>
      <td>@google</td>
      <td>@sethvargo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Kevin Naughton Jr.</td>
      <td>https://github.com//kdn251</td>
      <td>youtube.com/kevinnaughtonjr</td>
      <td>Software Engineer @google</td>
      <td>3.8k</td>
      <td>9</td>
      <td>New York, New York</td>
      <td>Google</td>
      <td>@kevinnaughtonjr</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Miguel Ángel Durán</td>
      <td>https://github.com//midudev</td>
      <td>https://midu.dev</td>
      <td>Software Engineer\n\nGitHub Star 🌟\nGoogle Dev...</td>
      <td>6.3k</td>
      <td>10</td>
      <td>Barcelona</td>
      <td>@AdevintaSpain</td>
      <td>@midudev</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jose Alcérreca</td>
      <td>https://github.com//JoseAlcerreca</td>
      <td>twitter.com/ppvi</td>
      <td>Android Developer Relations Engineer @ Google</td>
      <td>2.3k</td>
      <td>0</td>
      <td>Madrid, Spain</td>
      <td>@google</td>
      <td>@ppvi</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Shubham Mathur</td>
      <td>https://github.com//googleknight</td>
      <td>https://googleknight.github.io</td>
      <td>Software engineer II @ MDL Bangalore\n</td>
      <td>29</td>
      <td>40</td>
      <td>Bangalore, India</td>
      <td>Mckinsey &amp; Company</td>
      <td>None</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Nick Bourdakos</td>
      <td>https://github.com//bourdakos1</td>
      <td>None</td>
      <td>Software Engineer @google</td>
      <td>480</td>
      <td>8</td>
      <td>New York City</td>
      <td>@google</td>
      <td>None</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Mark</td>
      <td>https://github.com//MarkEdmondson1234</td>
      <td>https://code.markedmondson.me/</td>
      <td>Data Engineer @iihnordic  \nGoogle Developer E...</td>
      <td>783</td>
      <td>117</td>
      <td>Copenhagen</td>
      <td>@iihnordic</td>
      <td>@HoloMarkeD</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Dan Field</td>
      <td>https://github.com//dnfield</td>
      <td>None</td>
      <td>Software Engineer @google for @flutter</td>
      <td>928</td>
      <td>0</td>
      <td>None</td>
      <td>@google</td>
      <td>None</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Pierfrancesco Soffritti</td>
      <td>https://github.com//PierfrancescoSoffritti</td>
      <td>https://pierfrancescosoffritti.com/</td>
      <td>Software engineer @google</td>
      <td>566</td>
      <td>40</td>
      <td>London, UK</td>
      <td>@google</td>
      <td>@psoffritti</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Parker Moore</td>
      <td>https://github.com//parkr</td>
      <td>https://byparker.com</td>
      <td>🍩 🌎 Senior Engineer. Currently: @google. Forme...</td>
      <td>1.3k</td>
      <td>316</td>
      <td>USA</td>
      <td>Google</td>
      <td>None</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Gokmen Goksel</td>
      <td>https://github.com//gokmen</td>
      <td>None</td>
      <td>Software Engineer @google</td>
      <td>396</td>
      <td>58</td>
      <td>San Francisco</td>
      <td>@google</td>
      <td>@gokmen</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Sanket Singh</td>
      <td>https://github.com//singhsanket143</td>
      <td>None</td>
      <td>SDE @google | SDE @linkedin | Google Summer Of...</td>
      <td>1.7k</td>
      <td>13</td>
      <td>India</td>
      <td>@google</td>
      <td>None</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Justin Poehnelt</td>
      <td>https://github.com//jpoehnelt</td>
      <td>https://justin.poehnelt.com</td>
      <td>@google, @googleworkspace  Developer Relations...</td>
      <td>307</td>
      <td>9</td>
      <td>United States</td>
      <td>@google</td>
      <td>@jpoehnelt</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Shanqing Cai</td>
      <td>https://github.com//caisq</td>
      <td>https://caisq.github.io/</td>
      <td>Software Engineer @ Google Research</td>
      <td>343</td>
      <td>59</td>
      <td>None</td>
      <td>Google</td>
      <td>None</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Valerii Iatsko</td>
      <td>https://github.com//viatsko</td>
      <td>None</td>
      <td>UI Engineer @ Google</td>
      <td>346</td>
      <td>55</td>
      <td>None</td>
      <td>Google</td>
      <td>None</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Mizux</td>
      <td>https://github.com//Mizux</td>
      <td>http://www.mizux.net</td>
      <td>OSS Release Engineer @google</td>
      <td>167</td>
      <td>81</td>
      <td>Tours, France</td>
      <td>@google</td>
      <td>None</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Gabriela D'Ávila Ferrara</td>
      <td>https://github.com//gabidavila</td>
      <td>https://gabi.dev</td>
      <td>Developer Relations Engineer @google</td>
      <td>247</td>
      <td>25</td>
      <td>New Jersey</td>
      <td>@google @GoogleCloudPlatform</td>
      <td>@gabidavila</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Zulkarnine Mahmud</td>
      <td>https://github.com//zulkarnine</td>
      <td>www.zulkarnine.com</td>
      <td>Software Engineer at Google</td>
      <td>564</td>
      <td>0</td>
      <td>None</td>
      <td>Google</td>
      <td>None</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Adam Silverstein</td>
      <td>https://github.com//adamsilverstein</td>
      <td>http://www.earthbound.com</td>
      <td>Developer Relations Engineer @ Google</td>
      <td>153</td>
      <td>6</td>
      <td>Colorado, USA</td>
      <td>Google</td>
      <td>None</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Yacine Rezgui</td>
      <td>https://github.com//yrezgui</td>
      <td>https://yrezgui.com</td>
      <td>Creative software engineer.\nDeveloper advocat...</td>
      <td>513</td>
      <td>143</td>
      <td>London, UK</td>
      <td>Google</td>
      <td>None</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Google</td>
      <td>https://github.com//Google987</td>
      <td>youtube.com/alittlecoding</td>
      <td>Software Engineer\nYoutube: a little coding</td>
      <td>13</td>
      <td>11</td>
      <td>India</td>
      <td>None</td>
      <td>@arif_decrypted</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Prateek Narang</td>
      <td>https://github.com//prateek27</td>
      <td>www.prateeknarang.com</td>
      <td>Software Engineer-III at Google, Udemy Instruc...</td>
      <td>2.4k</td>
      <td>4</td>
      <td>Hyderabad</td>
      <td>Google India</td>
      <td>None</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Rakina Zata Amni</td>
      <td>https://github.com//rakina</td>
      <td>None</td>
      <td>Software Engineer @google @chromium 🇮🇩 🇯🇵👩‍💻</td>
      <td>307</td>
      <td>22</td>
      <td>Tokyo, Japan</td>
      <td>@google</td>
      <td>None</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Clément Mihailescu</td>
      <td>https://github.com//clementmihailescu</td>
      <td>algoexpert.io/clem</td>
      <td>Co-Founder &amp; CEO, AlgoExpert | Ex-Google &amp; Ex-...</td>
      <td>7.5k</td>
      <td>2</td>
      <td>None</td>
      <td>AlgoExpert</td>
      <td>@clemmihai</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Faisal Abid</td>
      <td>https://github.com//FaisalAbid</td>
      <td>http://www.FaisalAbid.com</td>
      <td>@google Developer Expert, Entrepreneur, and En...</td>
      <td>505</td>
      <td>30</td>
      <td>Toronto</td>
      <td>@eirene-cremations @bitstrapped @Shopistry</td>
      <td>@FaisalAbid</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Kate Lovett</td>
      <td>https://github.com//Piinks</td>
      <td>None</td>
      <td>Software Engineer at @google for @flutter</td>
      <td>745</td>
      <td>10</td>
      <td>Nashville, TN</td>
      <td>Google</td>
      <td>@k8lovett</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Andrey Kulikov</td>
      <td>https://github.com//andkulikov</td>
      <td>http://linkedin.com/in/andkulikov/</td>
      <td>Software Engineer at Google</td>
      <td>378</td>
      <td>0</td>
      <td>London</td>
      <td>@google</td>
      <td>None</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Márton Braun</td>
      <td>https://github.com//zsmb13</td>
      <td>https://zsmb.co/</td>
      <td>Android Developer Relations Engineer @google, ...</td>
      <td>418</td>
      <td>7</td>
      <td>Budapest, Hungary</td>
      <td>@google</td>
      <td>@zsmb13</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Kate Lovett</td>
      <td>https://github.com//Piinks</td>
      <td>None</td>
      <td>Software Engineer at @google for @flutter</td>
      <td>745</td>
      <td>10</td>
      <td>Nashville, TN</td>
      <td>Google</td>
      <td>@k8lovett</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Faisal Abid</td>
      <td>https://github.com//FaisalAbid</td>
      <td>http://www.FaisalAbid.com</td>
      <td>@google Developer Expert, Entrepreneur, and En...</td>
      <td>505</td>
      <td>30</td>
      <td>Toronto</td>
      <td>@eirene-cremations @bitstrapped @Shopistry</td>
      <td>@FaisalAbid</td>
    </tr>
  </tbody>
</table>
</div>





<b>limit_output extension: Maximum message size of 10000 exceeded with 15427 characters</b>



```python
df.shape
```




    (48, 9)



We were able to scrape about 50 profiles. There are lots of rich information like portfolio of a person and his/her profile headline and twitter handle.

There are few more cleaning needed too. Like The followers and following counts.


```python
df["followers"] = df.followers.apply(lambda x: 1000*float(x.replace("k","")) if "k" in x else float(x))
df.head()
```




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
      <th>name</th>
      <th>urls</th>
      <th>portfolio</th>
      <th>headline</th>
      <th>followers</th>
      <th>following</th>
      <th>home</th>
      <th>work</th>
      <th>twitter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Seth Vargo</td>
      <td>https://github.com//sethvargo</td>
      <td>https://www.sethvargo.com</td>
      <td>Engineer @google</td>
      <td>3200.0</td>
      <td>5</td>
      <td>Pittsburgh, PA</td>
      <td>@google</td>
      <td>@sethvargo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Kevin Naughton Jr.</td>
      <td>https://github.com//kdn251</td>
      <td>youtube.com/kevinnaughtonjr</td>
      <td>Software Engineer @google</td>
      <td>3800.0</td>
      <td>9</td>
      <td>New York, New York</td>
      <td>Google</td>
      <td>@kevinnaughtonjr</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Miguel Ángel Durán</td>
      <td>https://github.com//midudev</td>
      <td>https://midu.dev</td>
      <td>Software Engineer\n\nGitHub Star 🌟\nGoogle Dev...</td>
      <td>6300.0</td>
      <td>10</td>
      <td>Barcelona</td>
      <td>@AdevintaSpain</td>
      <td>@midudev</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jose Alcérreca</td>
      <td>https://github.com//JoseAlcerreca</td>
      <td>twitter.com/ppvi</td>
      <td>Android Developer Relations Engineer @ Google</td>
      <td>2300.0</td>
      <td>0</td>
      <td>Madrid, Spain</td>
      <td>@google</td>
      <td>@ppvi</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Shubham Mathur</td>
      <td>https://github.com//googleknight</td>
      <td>https://googleknight.github.io</td>
      <td>Software engineer II @ MDL Bangalore\n</td>
      <td>29.0</td>
      <td>40</td>
      <td>Bangalore, India</td>
      <td>Mckinsey &amp; Company</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["following"] = df.following.apply(lambda x: 1000*float(x.replace("k","")) if "k" in x else float(x))
df.head()
```




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
      <th>name</th>
      <th>urls</th>
      <th>portfolio</th>
      <th>headline</th>
      <th>followers</th>
      <th>following</th>
      <th>home</th>
      <th>work</th>
      <th>twitter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Seth Vargo</td>
      <td>https://github.com//sethvargo</td>
      <td>https://www.sethvargo.com</td>
      <td>Engineer @google</td>
      <td>3200.0</td>
      <td>5.0</td>
      <td>Pittsburgh, PA</td>
      <td>@google</td>
      <td>@sethvargo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Kevin Naughton Jr.</td>
      <td>https://github.com//kdn251</td>
      <td>youtube.com/kevinnaughtonjr</td>
      <td>Software Engineer @google</td>
      <td>3800.0</td>
      <td>9.0</td>
      <td>New York, New York</td>
      <td>Google</td>
      <td>@kevinnaughtonjr</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Miguel Ángel Durán</td>
      <td>https://github.com//midudev</td>
      <td>https://midu.dev</td>
      <td>Software Engineer\n\nGitHub Star 🌟\nGoogle Dev...</td>
      <td>6300.0</td>
      <td>10.0</td>
      <td>Barcelona</td>
      <td>@AdevintaSpain</td>
      <td>@midudev</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jose Alcérreca</td>
      <td>https://github.com//JoseAlcerreca</td>
      <td>twitter.com/ppvi</td>
      <td>Android Developer Relations Engineer @ Google</td>
      <td>2300.0</td>
      <td>0.0</td>
      <td>Madrid, Spain</td>
      <td>@google</td>
      <td>@ppvi</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Shubham Mathur</td>
      <td>https://github.com//googleknight</td>
      <td>https://googleknight.github.io</td>
      <td>Software engineer II @ MDL Bangalore\n</td>
      <td>29.0</td>
      <td>40.0</td>
      <td>Bangalore, India</td>
      <td>Mckinsey &amp; Company</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



### Plotting Followers

We will use `Seaborn` a library built above Matplotlib.


```python
import seaborn as sns
sns.set()


df.followers.hist()
```




    <AxesSubplot:>




    
![]({{site.url}}/assets/people_finder/output_44_1.png)
    


It seems that most people have very less followers.

### Finding Contact Details
If a profile has portfolio, then there is high chances that the portfolio has contact page too. So again, we can scrape that portfolio and collect such information.

### Pros and Cons of Using GitHub Search
* It is quite easier to find people in tech based on the skill-set but finding people who work in tech but does not have GitHub profile is not possible.
* Getting contact details is only possible if a person has the portfolio and that portfolio has it. Either way, its easier than finding the information from LinkedIn Profile.
* Sometimes the results might not be shown once GitHub suspects something is wrong in our request.


```python

```
