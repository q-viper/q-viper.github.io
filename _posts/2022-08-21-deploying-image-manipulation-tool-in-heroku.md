---
title:  "Deploying Image Manipulation Tool in Heroku"
date:   2022-08-28 01:29:17 +0545
categories:
    - Image Processing
    - Web App
tags:
    - Image Merger
header:
  teaser: assets/image_tool/app.png
---

In this part, we explore Deploying Image Manipulation tool in Heroku because it is a cloud application platform which gives us the flexibility to deploy our web apps from our GitHub repository and makes continous integration feasible. By the end of this part, we will have a web app like in this [link](https://image-manipulation-tool.herokuapp.com). 

Until now, we have done following part:
* Image Size Reducer in Python
* Image Merger Web Tool in Python

And now is the time for us to make that app live. So lets do it.

## Create a Heroku APP
* Visit [https://id.heroku.com/login](https://id.heroku.com/login) and create one account if you do not have already.
* Create a new app from [https://dashboard.heroku.com/new-app](https://dashboard.heroku.com/new-app).

## Deploying Heroku APP
* There are multiple options to do this but one simple way is using GitHub Repo. Lets use it.
* Under the `Deployment Methods` section, select GitHub and it will ask for connection to GitHub and then we have to select repository and its branch to deploy.
* But, first step is to prepare our `requirements.txt` file. Which will include the packages and its version required. Our `requirements.txt` looks like below.
```text
streamlit==1.12.0
numpy
pillow==6.2.0
opencv-python-headless

```
* Next is to prepare `setup.sh`. Here we could run some code to do something before running our app.
```shell
mkdir -p ~/.streamlit
mkdir data

echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
```

In above shell file, we create two folders, `./streamlit` and `data`. And told streamlit to run app in headless mode on given port and enable cors. These things are written in streamlit's config file inside `.streamlit` file.

* And then `Procfile`. Its a process file. I is a Heroku's special file. Below, we are telling Heroku that the app is Web and we will run commands `sh setup.sh file` and `streamlit run app.py`
  
```shell
web: sh setup.sh && streamlit run app.py
```

* Once done, we are ready to push our repository to GitHub.
* Once pushed, we can select our repo in Heroku's Deployment Method part.
* It might take some minutes to upload the changes and once done, we could open the app using the button near top right side.

## Launching Web App
And the app should be visible like below:

![]({{site.url}}/assets/image_tool/app.png)

