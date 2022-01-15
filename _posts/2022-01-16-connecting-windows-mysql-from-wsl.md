---
title:  "Connecting to MySQL Server in Windows Machine from WSL"
date:   2022-01-13 10:29:17 +0545
last_modified_at: 2022-01-16 12:29:17 +0545
categories:
    - apache airflow
    - data engineering
    - data pipelining
    - MySQL
tags:
    - data pipelining
    - tasks
    - branching
    - airflow for beginners
    - tutorial
header:
  teaser: assets/wsl_mysql/thumbnail.png
---

## Connecting MySQL Server in Windows Machine from WSL
What does this mean? In simple sentence, how do we connect to a MySQL server which is hosted in Windows from WSL. It might sound easy but let me tell you, IT IS NOT!!!!

I was trying to connect (from WSL) to my local MySQL which was installed on Windows Machine while using Airflow because my Airflow was installed in WSL. But it took me long to figure out the best way to do it. I hope it helps you too.

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

## References
* https://stackoverflow.com/questions/1559955/host-xxx-xx-xxx-xxx-is-not-allowed-to-connect-to-this-mysql-server
