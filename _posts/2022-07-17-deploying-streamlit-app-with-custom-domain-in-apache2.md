---
title:  "Deploying Streamlit App with Custom Domain in Apache2"
date:   2022-07-17 09:29:17 +0545
categories:
    - streamlit
    - apache2
tags:
    - streamlit
    - apache2
header:
  teaser: assets/streamlit_live/apached.png
---
## Introduction

Hello and welcome back everyone in this new blog of ours where I will be sharing how one could host a streamlit app in own domain using Apache2. From the last few blogs, I have written various projects in Streamlit and in this blog I am going to share how can we make those apps live in domain name.

## Have a Domain Name
First step is to have a domain name. And I took one from Godaddy. Once done, we should change the the DNS records especially A and CNAME records. **For the simplicity, I once had one [VPS in Contabo](https://contabo.com/en/) and thus I changed the nameserver of the domain to contabo's nameservers but it is not necessary.** One can use Cloudfare to manage DNS too if using the AWS's EC2 Instance. How to setup DNS is written below.

![]({{site.url}}/assets/streamlit_live/godaddy_ns.png)

### Manage the DNS records
DNS (Domain Name Server) records are kind of like address of a site pointing to some location. In simple terms, its made up of two part, Domain Name and Server, we point Domain Name to a Server having the contents to show upon visiting the Domain Name.

![]({{site.url}}/assets/streamlit_live/dns.png)

In above DNS record image, there are few records:
* **A** record is all about pointing domain name to certain IPv4 address. The address will be the IPv4 of my Contabo VPS. If you are using AWS's EC2 Instance, then the public IP will be in that field.
* With `*.domainname.com`, we will be pointing all the subdomain to that IP and I have also done the same. 

## Installing Ubuntu in Server

I am focusing on Ubuntu because of its simplicity and the way I could tweak things easily. While using Contabo, we can have a option to choose OS for a VPS and in EC2, we can choose the server OS too. So chosing Ubuntu might be a great idea.

## Installing and Configuring Apache2
Next we need to have Apache2 installed. One can use Ngnix too but for the simplicity, I am using Apache2. [Apache2](https://ubuntu.com/tutorials/install-and-configure-apache#1-overview) allows us to use multiple websites and using very minimal configurations and works to publish.

```bash
sudo apt update
sudo apt install apache2
```

Once done, it is worth looking into firewalls too. One can enable firewall by doing, `ufw enable`. If ufw is not installed, we can do it by, 
```bash
sudo apt update
sudo apt install ufw
```

One can enable app or port access by `ufw allow port`. To allow Apache2 in firewall, `ufw allow apache2`. Then checking if those are in list by `ufw status`.

One should do `ufw allow 22` if willing to do ssh. And it is necessary to allow all the ports be allowed those are going to be used later on. For example 8501 for streamlit's default port. Port 80 to open TCP connection usually to access website. And so on.

Once ports and apache are allowed we can restart apache2 and visit the default page by entering the server's IPv4 address. The following page should be visible.

![]({{site.url}}/assets/streamlit_live/apached.png)

If something like above is not visible then we have to check if the site is enabled. We could do that by `apache2ctl -S`.

![]({{site.url}}/assets/streamlit_live/list_site.png)

If nothing is on the list then we have to enable the default site via `a2ensite 000-default.conf`. And one can disable site via `a2dissite config_name`.

Above default page is usually rendered from `/var/www/html/index.html` which is also explained in default page itself. So first step usually is to change the content of the default page.

Now is the time for us to change the config file which will be inside the `/etc/apache2/sites-available`. And our default site will be using the config `000-default.conf`. It should look something like below:

```
<VirtualHost *:80>
        #ServerName www.example.com
        #ServerAdmin webmaster@localhost
        DocumentRoot /var/www/html
        <FilesMatch \.php$>
             SetHandler "proxy:unix:/var/run/php/php7.4-fpm.sock|fcgi://localhost"
        </FilesMatch>
        ErrorLog /error.log
        CustomLog /access.log combined
</VirtualHost>
```

In above config, the number 80 in the first line represents the port to listen for default site. And normally 80 is the port for TCP. While changing the 80 to any other, one should also update the `/etc/apache2/ports.conf` file, and insert the `Listen new_port` in it and restart the apache2 by doing `systemctl restart apache2`.

```
# If you just change the port or add more ports here, you will likely also
# have to change the VirtualHost statement in
# /etc/apache2/sites-enabled/000-default.conf

Listen 80
Listen 8501

<IfModule ssl_module>
	Listen 443
</IfModule>

<IfModule mod_gnutls.c>
	Listen 443
</IfModule>

# vim: syntax=apache ts=4 sw=4 sts=4 sr noet
```

In above example, I have added 8501 for listening to Streamlit app.

### Configuring Subdomain

I have made a subdomain `data` for the domain and next will be setting it up its config. We should create this config file inside `/etc/apache2/sites-available`. And I have named it as `data.mydomain.com.conf`. Here `mydomain` is just an alias to hide my domain name :P.

```
<VirtualHost *:80>
        ServerAdmin admin@mydomain.com
        ServerName data.mydomain.com
        ServerAlias data.mydomain.com
        ProxyRequests Off

        <Location />
                ProxyPreserveHost On
                ProxyPass http://mydomain.com:8502/
                ProxyPassReverse http://mydomain.com:8502/
        </Location>
     # Uncomment the line below if your site uses SSL.
     #SSLProxyEngine On
</VirtualHost>
```

Once done, we should enable this config too by doing `a2ensite data.mydomain.com.conf`. And we should check for syntax by doing `apachectl configtest` and it should return OK. Then we should do below to enable proxy:

```bash
sudo a2enmod proxy
sudo a2enmod proxy_http
sudo service apache2 restart
```

In above config, we have created a new server name and its alias too. Then we have setup the proxy to redirect it to the new port 8502 running in `mydomain.com`. For more about this step, [please follow this answer in serverfault](https://serverfault.com/a/749876).

### Running a Streamlit App
My simple way of setting up Streamlit app is like below:

![]({{site.url}}/assets/streamlit_live/app_str.png)

Inside a data folder, create a virtual environment. We can do that using python library [virtualenv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).
* `apt -y install python3.8-venv` will install the virtual environment.
* Once done, create one environment in data folder named as env, `python3 -m venv env`
* Activate it by doing `source env/bin/activate`.
* Once activated lets install streamlit and its dependencies:
    * `pip install streamlit pandas cufflinks plotly`
    * And if you are willing to install MySQL client as well, 
    ```bash
        sudo apt-get install python-dev python3-dev
        sudo apt-get install libmysqlclient-dev
        pip install pymysql
        pip install mysqlclient
    ```
* Inside a `.streamlit` folder, create a `config.toml` file where content should be something like below:

```toml
[server]
headless = true
port = 8502

[browser]
serverAddress = "data.mydomain.com"
```

* In above toml file, we have set the server address and port too. This config is used by streamlit while creating a web server. Once we do `streamlit run app.py` assuming that `app.py` is our streamlit app. It will show the urls in terminal like `data.mydomain.com:8502`. Going in that url will lead us to the streamlit app. But we could exclude that 8502 in the last and simply visit the site too because we have set up the sub domain config that way.
 

Thats all for now. Thank you so much for reading this blog.


