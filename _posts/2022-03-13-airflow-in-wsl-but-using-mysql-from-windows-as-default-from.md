---
title:  "Running Airflow in WSL but Using MySQL Server from Windows as Default"
date:   2022-03-13 9:29:17 +0545
last_modified_at: 2022-03-13 12:29:17 +0545
categories:
    - apache airflow
    - data engineering
    - data pipelining
    - mysql
tags:
    - data pipelining
    - ubuntu and windows
    - windows subsystem for linux
    - airflow for beginners
    - tutorial
    - mysql
header:
  teaser: assets/airflow_blog/scheduler.png
---

## Introduction
There are lots of benefit of using MySQL as a backend database in Airflow. Main reason is that MySQL is widely used in production instead of SQlite. Also, we can have scalable database system where we can have concurrent requests, high security and well defined permission and roles.

From the last few blogs, I've share how can we fully schedule DAGs written in Windows Machine from WSL. And we were using default Database, SQlite on those blogs but later we did connect to MySQL running in Windows from WSL. But in this blog, we will do clean install of Airflow in WSL and we will use MySQL as a Database. This blog is related to some of the previous blogs in some way but it is not required to go there.
* [Running Airflow in WSL and Getting Started with it](https://q-viper.github.io/2021/12/01/running-airflow-in-wsl-and-getting-started-with-it/)
* [Dynamic Tasks in Airflow](https://q-viper.github.io/2022/01/09/airflow-dynamic-tasks/)
* [Connecting MySQL Running in Windows from WSL](https://q-viper.github.io/2022/01/13/connecting-windows-mysql-from-wsl/)
* [Branching Task in Airflow](https://q-viper.github.io/2022/01/23/branching-task-in-airflow/)


## Installing WSL
Using airflow in Windows machine is hard way to go but with the use of Docker one can do it easily. But I am using [Ubuntu in WSL](https://www.microsoft.com/store/productId/9NBLGGH4MSV6) (Windows Subsystem for Linux) to use Airflow in my Windows.

## Installing Airflow
(Referenced from [here](https://towardsdatascience.com/run-apache-airflow-on-windows-10-without-docker-3c5754bb98b4).)
* Open the Ubuntu.
* Update system packages.
    ```bash
    sudo apt update
    sudo apt upgrade
    ```

* Installing PIP.
    ```bash
    sudo apt-get install software-properties-common
    sudo apt-add-repository universe
    sudo apt-get update
    sudo apt-get install python-setuptools
    sudo apt install python3-pip
    ```

* Run `sudo nano /etc/wsl.conf` then, insert the block below, save and exit with `ctrl+s` `ctrl+x`
```
[automount]
root = /
options = "metadata"
```

* To setup a airflow home, first make sure where to install it. Run `nano ~/.bashrc`, insert the line below, save and exit with `ctrl+s` `ctrl+x`

    ```export AIRFLOW_HOME=c/users/YOURNAME/airflowhome```

    Mine is, `/mnt/c/users/dell/myName/documents/airflow`

* Install virtualenv to create environment.
    ```
    sudo apt install python3-virtualenv
    ```

* Create and activate environment.
    ```
    virtualenv airflow_env
    source airflow_env/bin/activate
    ```

* Install airflow
    ```
    pip install apache-airflow
    ```

* Make sure if Airflow is installed properly.
    ```
    airflow info
    ```

    If no error pops up, proceed else install missing packages.


## Making Connection to MySQL Running in Windows From WSL

### MySQL Client in WSL
First install MySQL client in WSL using below command which can be seen once we type `mysql` in WSL terminal.

```shell
sudo apt install mysql-client-core-8.0     # version 8.0.27-0ubuntu0.20.04.1, or
sudo apt install mariadb-client-core-10.3  # version 1:10.3.31-0ubuntu0.20.04.1
```

For me, I did first one.

### Find IPv4 Adress of WSL
* Go to Settings -> Network and Internet -> Status -> View Hardware and connection properties. Look for WSL.
* My looks like below. But I've shaded the adresses.

![]({{site.url}}/assets/wsl_mysql/ipv4_address.png)

Now try to connect to MySQL from WSL using below command:

```shell
mysql -u wsl_root -p -h 192.168.xxx.xxx
```

Please remember that in above command xxx is just a placeholder. Also, `root` is just a username that we tried to login with. **We will get an error right now with above command and we will fix it.**

### Making New User in MySQL to make a Call from WSL

```sql
CREATE USER 'wsl_root'@'localhost' IDENTIFIED BY 'password';
GRANT ALL PRIVILEGES ON *.* TO 'wsl_root'@'localhost' WITH GRANT OPTION;
CREATE USER 'wsl_root'@'%' IDENTIFIED BY 'password';
GRANT ALL PRIVILEGES ON *.* TO 'wsl_root'@'%' WITH GRANT OPTION;
FLUSH PRIVILEGES;
```

In above query, 
* `wsl_root` is a name of user that we will use from WSL. `localhost` is a adress where MySQL is and `password` is password. :)
* We have granted all privileges to that user and it will be just another admin.

### From WSL
Now running the command `mysql -u wsl_root -p -h 192.168.xxx.xxx` and giving password after it asked, we could connect to the MySQL server.

### References
* [StackOverflow](https://stackoverflow.com/questions/1559955/host-xxx-xx-xxx-xxx-is-not-allowed-to-connect-to-this-mysql-server)

## Install MySQL Connector
Now we need to [install MySQL Connection Provider](https://airflow.apache.org/docs/apache-airflow-providers-mysql/stable/index.html) for Airflow as:

```
pip install apache-airflow-providers-mysql
```

If error pops up, it might be because of our MySQL client's version. The fix in that case ([from here](https://stackoverflow.com/a/67605701)):
* For Debian 8 or older,

```
sudo apt-get install libmysqlclient-dev
```

* For Debian > 8

```
sudo apt-get install default-libmysqlclient-dev
```

## Creating Airflow Database
Now, lets go to our MySQL Workbench in Windows side and run below queries to setup our [MySQL database](https://airflow.apache.org/docs/apache-airflow/stable/howto/set-up-database.html):

```sql
CREATE DATABASE airflow_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'airflow_user' IDENTIFIED BY 'airflow_pass';
GRANT ALL PRIVILEGES ON airflow_db.* TO 'airflow_user';
```

In above code, we are creating a database `airflow_db` and also created user as airflow_user and granted all privileges.

## Initializing Database,
First we need to edit our Airflow's configuration file. We should change the value of `sql_alchemy_conn` in `airflow.cfg`. This file is located at Airflow's home directory.

```
sql_alchemy_conn = mysql+mysqldb://wsl_root:password@my_ip:3306/airflow_db
```

**Please change my_ip by the IPv4 Address we found in above step.**

Now from WSL, do `airflow db init`. If no error pops up, we are good to go.

* **If a error comes saying "Operation is not Permitted" make sure you have write access to the $AIRFLOW_HOME folder from WSL. So do something like below**:

    ```
    sudo chmod -R 777 /mnt/c/Users/Dell/Documents/airflow/
    ```

* **Create airflow user.**
    ```
    airflow users create [-h] -e EMAIL -f FIRSTNAME -l LASTNAME [-p PASSWORD] -r
                         ROLE [--use-random-password] -u USERNAME
    ```

## Run Webserver and Scheduler
Now lets open another Ubuntu terminal and run `airflow webserver` in it. Also run `airflow scheduler` in another terminal. 

Next, open the Airflow's Web URl which must be `http://localhost:8080` then sigin using the credentials that we just created in above step. If it works, we could try scheduling some of example DAGs shown there. And those should be running without any errors.


If we head over to the Workbench, we can see the tables being created and populated in `airflow_db`.

Thank you for your time.


```python

```
