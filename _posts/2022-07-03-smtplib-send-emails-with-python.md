---
title:  "SMTPlib: Send Emails with Python"
date:   2022-07-03 09:29:17 +0545
categories:
    - Python
    - SMTPlib
tags:
    - Python
    - SMTPlib
header:
  teaser: assets/smtplib/smtp.png
---
## Introduction
Hello and welcome back everyone, in this blog we will be exploring how we can send emails using Python. We will be using [SMTPlib](https://docs.python.org/3/library/smtplib.html).

According to the [documentation of SMTPlib](https://docs.python.org/3/library/smtplib.html#module-smtplib), *The smtplib module defines an SMTP client session object that can be used to send mail to any internet machine with an SMTP or ESMTP listener daemon. For details of SMTP and ESMTP operation, consult RFC 821 (Simple Mail Transfer Protocol) and RFC 1869 (SMTP Service Extensions).*

Here in this blog, we will create a class that will have the ability to create a session, send text email and then email with an attachment.

## Imports
Lets import the necessary packages.


```python
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
import os
from os.path import basename
```

* `smtplib` to send emails.
* `email` and its modules to create a email with formats.
* `os` to read path

## Creating a Class
Here we will initialize a class and it will take host, port number, sender email and the password of the sender.


```python
class Mail:

    def __init__(self, host="smtp.gmail.com", port=465, sender="sender@gmail.com", 
                 password="Topsy crate."):
        self.port = port
        self.smtp_server = host
        self.sender_mail = sender
        self.password = password
```

## Creating a Service
Since we have all the information needed to login, we will create a service. We will create a service and finally login in it using email and password.

```python
     def set_service(self):
        self.service = smtplib.SMTP(self.smtp_server,self.port)
        self.service.login(self.sender_mail, self.password)
```


## Send Simple Email
To send a simple email, we already have a valid session and all we need now is the sender email, receiver and then subject. The content in below method is sent as a body.
```python
    def send(self, emails, subject, content):        
        for email in emails:
            try:
                result = self.service.sendmail(self.sender_mail, email, f"Subject: {subject}\n{content}")
                print(f"Sent `{subject}` to: {email}")
            except:
                self.set_service()
                result = self.service.sendmail(self.sender_mail, email, f"Subject: {subject}\n{content}")
                print(f"Sent `{subject}` to: {email}")
```

## Send Simple Attachment
To send an attachment in email, we will use MIMEMultipart and MIMEApplication to collect the email and attachment parts. The method takes emails, subject, content and folder as shown below in the docstring.
```python
    def send_file(self, emails, subject, content, folder):
        """
        emails: where to send
        subject: what subject to send
        content: what content to send
        folder: which folder to send
        """
        msg = MIMEMultipart()
        msg['From'] = self.sender_mail
        msg['To'] = COMMASPACE.join(emails)
        msg['Date'] = formatdate(localtime=True)
        msg['Subject'] = subject

        msg.attach(MIMEText(content))
        
        for f in os.listdir(folder):
            with open(folder+f, "rb") as fil:
                part = MIMEApplication(
                    fil.read(),
                    Name=basename(f)
                )
            
            part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
            msg.attach(part)

        try:
            self.service.sendmail(self.sender_mail, emails, msg.as_string())
        except Exception as e:
            self.set_service()
            self.service.sendmail(self.sender_mail, emails, msg.as_string())
```

## Combining all


```python
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
import os
from os.path import basename

class Mail:
    def __init__(self, host="smtp.gmail.com", port=465, sender="sender@gmail.com", 
                 password="Topsy crate"):
        self.port = port
        self.smtp_server = host
        self.sender_mail = sender
        self.password = password

    def set_service(self):
        self.service = smtplib.SMTP(self.smtp_server,self.port)
        self.service.login(self.sender_mail, self.password)

    def send(self, emails, subject, content):
        
        for email in emails:
            try:
                result = self.service.sendmail(self.sender_mail, email, f"Subject: {subject}\n{content}")
                print(f"Sent `{subject}` to: {email}")
            except:
                self.set_service()
                result = self.service.sendmail(self.sender_mail, email, f"Subject: {subject}\n{content}")
                print(f"Sent `{subject}` to: {email}")
        #service.quit()

    def send_file(self, emails, subject, content, folder):
        """
        emails:
        subject:
        content:
        folder:
        """
        msg = MIMEMultipart()
        msg['From'] = self.sender_mail
        msg['To'] = COMMASPACE.join(emails)
        msg['Date'] = formatdate(localtime=True)
        msg['Subject'] = subject

        msg.attach(MIMEText(content))
        
        for f in os.listdir(folder):
            with open(folder+f, "rb") as fil:
                part = MIMEApplication(
                    fil.read(),
                    Name=basename(f)
                )
            # After the file is closed
            part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
            msg.attach(part)

        try:
            self.service.sendmail(self.sender_mail, emails, msg.as_string())
        except Exception as e:
            self.set_service()
            self.service.sendmail(self.sender_mail, emails, msg.as_string())



```


```python
mailer = Mail(host="smtp.gmail.com", port=465, sender="sender@gmail.com", 
                 password="Topsy crate")
mailer.set_service()
mailer.send(emails_list, "Simpe Test Email", "Hey, \n This is just a test email.")
mailer.send_file(emails_comma_separated, "Attached Test Email", "Hey, \n This is just a test email.",folder)
```

In above example, folder is just a folder path that is accessible from this location. And the emails_list in `mailer.send` is the list of emails in list. And in `mailer.send_file`, emails are sent as comma separated value.

![]({{site.url}}/assets/smtplib/simple.png)
![]({{site.url}}/assets/smtplib/attached.png)

