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
Apache Superset is a very useful and easy-to-use visualization and dashboard-making tool that can be an alternative to tools like Tableau and PowerBI.
In this blog, we will explore how we can create awesome data dashboards using Apache superset with little to no code at all. But there are a few things one should do before making the first dashboard, we need to have installed Superset and have some data too.

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

* Initialize the DB as `superset db upgrade`.
* Load some data to play with `superset load_examples`
* Create default roles and permissions `superset init`. If some errors like table not found are shown then that's because of database issue initialization. Check the database or re-install the superset. Also, make sure to export FLASK_APP first.
* To start a development web server run `superset run`. It will open on default port 5000.

## Opening First Dashboard
If everything worked fine, then by default, Apache Superset should be accessible at http://127.0.0.1:5000/. It should look like below:

![]({{site.url}}/assets/apache_superset/as_login.png)

Upon entering the password and username that we set earlier, we could see the empty dashboard as below:

![]({{site.url}}/assets/apache_superset/as_init.png)

## Preparing Data

### Choosing a Database
We can choose our database by going into Data/Databases.

![]({{site.url}}/assets/apache_superset/add_db.png)

I am choosing a MySQL database. But my MySQL connection will be a little bit different than others because I will be using Superset running in WSL while my MySQL server will be running in Windows hence I should pass Network IP. By default, MySQL runs in 3306. For using MySQL running in Windows from WSL, please follow [this blog]({{site.url}}/2022/01/13/connecting-windows-mysql-from-wsl/) of mine.
* First install MySQL database server along with MySQL Workbench.
* Then run a query in it `create database COVID_DASHBOARD;` to create a new database where we will put our data.
* Create a connection as:

![]({{site.url}}/assets/apache_superset/mysql_con.png)

* Make sure to allow data upload in this database. The settings can be found in the Advanced>Security Section.

![]({{site.url}}/assets/apache_superset/allow_upload.png)

### Choosing Data
* To make our data dashboard, we should have data. Apache Superset allows us to use data in the following format.

![]({{site.url}}/assets/apache_superset/as_data.png)

* For this project, I am choosing COVID-19 data (CSV format) available in [GitHub Repo](https://github.com/owid/covid-19-data/tree/master/public/data). 

* After downloading a CSV file, we will upload it into our database.
![]({{site.url}}/assets/apache_superset/upload.png)

* Uploading might take a little bit more time because there are lots of columns in the data and the size of the data itself is huge (196451 rows 67 columns). But we can look if the data upload is on right track or not by querying a table `SELECT * FROM covid_dashboard.covid_raw_data;`. A result must be shown. 
* Once done uploading, something like the below should be shown.
![]({{site.url}}/assets/apache_superset/data.png)
* 


## Making a Chart
The hard part is completed. Now with a little bit of SQL knowledge, we can create charts.
* Go to charts.
* Then add a new chart.
* Choose a dataset.
* Choose one chart type. I've selected a time-series.

![]({{site.url}}/assets/apache_superset/chart.png)

* Next, rename the chart from untitled to cases trend.
![]({{site.url}}/assets/apache_superset/init_chart.png)

* Initially the date column might not be in date time type so we need to change its data type from Workbench. And then we need to sync these changes in a column in Superset. Which can be done via the state. Click on the dots on the right side of the dataset name present in the left section. Then edit the dataset. Then columns and finally sync columns from source and save this.
![]({{site.url}}/assets/apache_superset/sync.png)

* In the second section, we can tweak the settings for this chart. In its data section, we should select a Time column as a Date. Then Time Grain. Then in the Query Section, we need to select a metric, in our case, it will be some of the new_cases. Then in Group By section, we select a location. Then run the query to see the trend chart like below.
![]({{site.url}}/assets/apache_superset/trend_1.png)

* From here, we can do a lot of things, and we can export the result in CSV format too.
* It seems that our result needs a little bit of filtering to show trends of countries only. So let's filter those which have NULL in the Continent column.
![]({{site.url}}/assets/apache_superset/filtered.png)

* While hovering over, I want to view the highest value name at the top so we should add sort in it.

## Bar Chart
* Next create a bar chart to show the top countries with death tolls.
* Please take a careful look at the second column.
* Select **Metric** as MAX of column total_deaths. Because we want to see the latest value of it and this field is cumulative.
* In **Filters**, the select continent is not equal to null because in location, continent names and some other names are also present and we do not want that.
* In **Series**, select the column name by which we want to Group Data. Let's select a location.
* In **Row Limit** select 10, as we want to show only the top 10 bars.
* In **Sort By**, select max of column total_deaths.
* And then run the query to see the chart like below.
![]({{site.url}}/assets/apache_superset/death_bar.png)

    

## Map Chart
Next, create a map chart to show total deaths across the world.

![]({{site.url}}/assets/apache_superset/death_map.png)

## Creating Dashboard

Now that we have 3 charts, let's create a dashboard by going into Dashboards>New Dashboard.
![]({{site.url}}/assets/apache_superset/dashboard.png)

Next insert charts by drag-and-drop.

![]({{site.url}}/assets/apache_superset/dashboard_added.png)



We can even download the dashboard as an image too.
![]({{site.url}}/assets/apache_superset/covid-dashboard.jpg)

## Finally
That's all for this part of exploring Apache Superset and I find this tool very useful because we can create our own charts in the more customized way if we are familiar with SQL. There are lots of features still to be explored in Apache Superset and I will try to make the next example if time persists. Until then, please stay exploring our [site](https://dataqoil.com).