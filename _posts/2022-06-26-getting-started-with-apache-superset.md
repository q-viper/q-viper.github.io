---
title:  "Getting Started with Apache Superset for Data Dashboards"
date:   2022-06-26 09:29:17 +0545
categories:
    - Data Visualization
    - Apache Superset
tags:
    - WSL
    - COVID Dashboard
    - Apache Superset
header:
  teaser: assets/apache_superset/covid-dashboard.jpg
---
## Making Data Dashboard with Apache Superset
Hello and welcome back everyone, in this blog, we will explore how we can create awesome data dashboards using Apache superset with little to no code at all. But there are few things one should do before making first dashboard, we need to have installed Superset and have some data too.

## Installing Apache Superset
This blog will be using [Apache Superset](https://superset.apache.org/docs/installation/installing-superset-from-scratch) in WSL (Windows Subsystem for Linux) because the library `apache-superset` has OS level dependency. 

* Install following packages as `sudo apt-get install build-essential libssl-dev libffi-dev python3-dev python3-pip libsasl2-dev libldap2-dev default-libmysqlclient-dev`.
* If some packages are missing, `sudo apt-get update` might help out.
* Install [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) as `pip install virtualenv`.
* Create an environment as `python3 -m venv env_name`. My env_name will be `airflow_env`.
* Activate environment as `source path/to/env_name/bin/activate`.
* Install Apache [Superset](https://superset.apache.org/docs/installation/installing-superset-from-scratch) as `pip install apache-superset`.

## Setting up Superset
* Create an admin user in your metadata database (use `admin` as username to be able to load the examples)

`
export FLASK_APP=superset
superset fab create-admin
`

* Initialize the db as `superset db upgrade`.
* Load some data to play with `superset load_examples`
* Create default roles and permissions `superset init`. If some errors like table not found is shown then thats because of database issue initialization. Check the database or re-install superset. Also make sure to export FLASK_APP first.
* To start a development web server run `superset run`. It will open on default port 5000.

## Opening First Dashboard
If everything worked fine, then by default, superset should be accessible in http://127.0.0.1:5000/. It should look like below:

![]({{site.url}}/assets/apache_superset/as_login.png)

Upon entering the password and username that we have set earlier, we could see the empty dashboard as below:

![]({{site.url}}/assets/apache_superset/as_init.png)

## Preparing Data

### Choosing a Database
We can choose our database by going into Data/Databases.

![]({{site.url}}/assets/apache_superset/add_db.png)

I am choosing a MySQL database. But mine MySQL connection will be little bit different than others because I will be using Superset running in WSL while my MySQL server will be running in Windows hence I should pass Network IP. By default, MySQL runs in 3306. For using MySQL running in Windows from WSL, please follow [this blog](https://q-viper.github.io/2022/01/13/connecting-windows-mysql-from-wsl/) of mine.
* First install mysql database server along with MySQL Workbench.
* Then run a query in it `create database COVID_DASHBOARD;` to create a new database where we will put our data.
* Create a connecction as:

![]({{site.url}}/assets/apache_superset/mysql_con.png)

* Make sure to allow data upload in this database. The settings can be found in Advanced>Security Section.

![]({{site.url}}/assets/apache_superset/allow_upload.png)

### Choosing Data
* To make our data dashboard, we should have data. Apache Superset allows us to use data in following format.

![]({{site.url}}/assets/apache_superset/as_data.png)

* For this project, I am choosing COVID-19 data (CSV format) available in [GitHub Repo](https://github.com/owid/covid-19-data/tree/master/public/data). 

* After downloading a CSV file, we will upload it into our database.
![]({{site.url}}/assets/apache_superset/upload.png)

* Uploading might take little bit more time because there are lots of columns in the data and the size of data itself is huge (196451 rows 67 columns). But we can look if the data upload is on right track or not by querying a table `SELECT * FROM covid_dashboard.covid_raw_data;`. A result must be shown. 
* Once done uploading, something like below should be shown.
![]({{site.url}}/assets/apache_superset/data.png)
* 


## Making a Chart
The hard part is completed. Now with little bit of SQL knowledge, we can create charts.
* Go to charts.
* Then add new chart.
* Choose a dataset.
* Choose one chart type. I've selected timeseries.

![]({{site.url}}/assets/apache_superset/chart.png)

* Next, rename the chart from untitled to cases trend.
![]({{site.url}}/assets/apache_superset/init_chart.png)

* Initially the date column might not be in date time type so we need to change its data type from Workbench. And then we need to sync this changes in column in Superset. Which can be done via state. Click on dots on the right side of the dataset name present in left section. Then edit dataset. Then columns and finally sync columns from source and save this.
![]({{site.url}}/assets/apache_superset/sync.png)

* In the second section, we can tweak the settings for this chart. In its data section, we should select a Time column as Date. Then Time Grain. Then in the Query Section, we need to select a metric, in our case, it will be sum of new_cases. Then in Group By section, we select location. Then run the query to see trend chart like below.
![]({{site.url}}/assets/apache_superset/trend_1.png)

* From here, we can do lot of things, we can export the result in CSV format too.
* It seems that our result needs little bit of filtering to show trends of countries only. So lets filter those which have NULL in the Continent column.
![]({{site.url}}/assets/apache_superset/filtered.png)

* While hovering over, I want to view highest value name in top so we should add sort in it.

## Bar Chart
* Next create a bar chart to show top countries with death tolls.
* Please take a careful look into the second column.
* Select **Metric** as MAX of column total_deaths. Because we want to see the latest value of it and this field is cumulative.
* In **Filters**, select continent is not equals to null because in location, continent names and some other names are also present and we do not want that.
* In **Series**, select the column name by which we want to Group Data by. Lets select location.
* In **Row Limit** select 10, as we want to show only top 10 bars.
* In **Sort By**, select max of column total_deaths.
* And then run the query to see the chart like below.
![]({{site.url}}/assets/apache_superset/death_bar.png)

    

## Map Chart
Next is, create a map chart to show total deaths across the world.

![]({{site.url}}/assets/apache_superset/death_map.png)

## Creating Dashboard

Now that we have 3 charts, lets create a dashboard by going into Dashboards>New Dashboard.
![]({{site.url}}/assets/apache_superset/dashboard.png)

Next insert charts by drag-and-drop.

![]({{site.url}}/assets/apache_superset/dashboard_added.png)



We can even download dasboard as image too.
![]({{site.url}}/assets/apache_superset/covid-dashboard.jpg)

## Finally
Thats all for this part of exploring Apache Superset and I find this tool very useful because we can create our own charts in more customized way if we are familiar with SQL. There are lots of features still to be explored in Apache Superset and I will try to make next example if time persists.


```python

```
