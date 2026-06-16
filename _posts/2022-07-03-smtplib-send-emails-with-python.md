---
layout: single
title: "Send Emails with Python smtplib: Plain Text, HTML, Attachments, Gmail SMTP, and Environment Variables"
date: 2022-07-03 09:29:17 +0545
last_modified_at: 2026-06-16
categories:
  - Python
  - SMTP
  - Email
tags:
  - "Python"
  - "smtplib"
  - "SMTP"
  - "Email"
  - "Gmail SMTP"
  - "EmailMessage"
  - "Attachments"
  - "Environment variables"
description: "A beginner-friendly guide to sending emails with Python smtplib, including Gmail SMTP, secure environment variables, plain text emails, HTML emails, multiple recipients, and attachments."
excerpt: "Learn how to send emails with Python using smtplib and EmailMessage. This tutorial covers plain text email, HTML email, attachments, Gmail SMTP, environment variables, and a reusable mailer class."
header:
  teaser: assets/smtplib/smtp.png
  og_image: assets/smtplib/smtp.png
  image_description: "SMTP email sending tutorial cover image"
toc: true
toc_label: "Send Emails with Python"
toc_icon: "envelope"
toc_sticky: true
read_time: true
share: true
related: true
classes: wide
---

In this tutorial, we will learn how to **send emails with Python using `smtplib`**.

Python has a built-in module called `smtplib` that can create an SMTP client session and send emails through an SMTP server. We can use it to send:

- plain text emails
- HTML emails
- emails to multiple recipients
- emails with attachments
- emails through Gmail SMTP
- emails through other SMTP providers

The original version of this blog used the older `MIMEText`, `MIMEMultipart`, and `MIMEApplication` style. That still works, but in modern Python, the `EmailMessage` API is cleaner and easier to read.

In this updated version, we will use `smtplib` together with `EmailMessage`.

## Important Security Note

Do not hardcode your email password inside Python code.

Bad:

```python
password = "my-real-password"
```

Good:

```python
password = os.environ["SMTP_PASSWORD"]
```

If you are using Gmail, do not use your normal Google account password in scripts. Use a safer method such as:

- Google app password, if available for your account
- OAuth for production applications
- a transactional email provider
- a dedicated SMTP account

For a personal script, a Gmail app password can work when 2-Step Verification is enabled. For production apps, OAuth or an email service provider is better.

## What Is SMTP?

SMTP stands for **Simple Mail Transfer Protocol**.

It is the standard protocol used to send emails from one server to another. When we send an email from Python, our script connects to an SMTP server, logs in, and sends the message.

Common SMTP examples:

| Provider | SMTP Host | SSL Port | STARTTLS Port |
|---|---:|---:|---:|
| Gmail | `smtp.gmail.com` | `465` | `587` |
| Outlook | `smtp.office365.com` | usually STARTTLS | `587` |
| Yahoo | `smtp.mail.yahoo.com` | `465` | `587` |

In this tutorial, we will use Gmail SMTP examples, but the same idea works with other providers.

## Required Imports

```python
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
```

For guessing attachment MIME types, we can also use:

```python
import mimetypes
```

So the full import block becomes:

```python
import mimetypes
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
```

## Store Email Credentials Safely

The safest simple approach is to use environment variables.

Create a `.env` file for local development:

```text
SMTP_HOST=smtp.gmail.com
SMTP_PORT=465
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_SENDER=your_email@gmail.com
```

Add `.env` to `.gitignore`:

```text
.env
__pycache__/
*.pyc
```

Install `python-dotenv` if you want to load `.env` automatically:

```bash
pip install python-dotenv
```

Then load it:

```python
from dotenv import load_dotenv

load_dotenv()
```

For deployment, set environment variables in your server, Docker container, CI/CD platform, or cloud provider.

## Send a Plain Text Email

Here is the simplest example.

```python
import os
import smtplib
from email.message import EmailMessage

from dotenv import load_dotenv


load_dotenv()

smtp_host = os.environ["SMTP_HOST"]
smtp_port = int(os.environ.get("SMTP_PORT", "465"))
smtp_username = os.environ["SMTP_USERNAME"]
smtp_password = os.environ["SMTP_PASSWORD"]
sender = os.environ["SMTP_SENDER"]

receiver = "receiver@example.com"

message = EmailMessage()
message["From"] = sender
message["To"] = receiver
message["Subject"] = "Simple Test Email"

message.set_content(
    "Hello,\n\nThis is a test email sent from Python using smtplib."
)

with smtplib.SMTP_SSL(smtp_host, smtp_port) as smtp:
    smtp.login(smtp_username, smtp_password)
    smtp.send_message(message)
```

This sends a plain text email.

## Why Use SMTP_SSL?

Port `465` is commonly used with SMTP over SSL.

```python
with smtplib.SMTP_SSL(smtp_host, smtp_port) as smtp:
    ...
```

Another common method is STARTTLS on port `587`.

```python
with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
    smtp.starttls()
    smtp.login(username, password)
    smtp.send_message(message)
```

Both approaches are common. Use the one recommended by your email provider.

## Send Email to Multiple Recipients

The `To` header can contain multiple recipients.

```python
receivers = [
    "person1@example.com",
    "person2@example.com",
    "person3@example.com"
]

message = EmailMessage()
message["From"] = sender
message["To"] = ", ".join(receivers)
message["Subject"] = "Email to Multiple Recipients"

message.set_content(
    "Hello,\n\nThis email was sent to multiple recipients."
)

with smtplib.SMTP_SSL(smtp_host, smtp_port) as smtp:
    smtp.login(smtp_username, smtp_password)
    smtp.send_message(message)
```

## Add CC and BCC

```python
to_emails = ["person1@example.com"]
cc_emails = ["person2@example.com"]
bcc_emails = ["person3@example.com"]

message = EmailMessage()
message["From"] = sender
message["To"] = ", ".join(to_emails)
message["Cc"] = ", ".join(cc_emails)
message["Subject"] = "Email with CC and BCC"

message.set_content("Hello,\n\nThis email has CC and BCC recipients.")

all_recipients = to_emails + cc_emails + bcc_emails

with smtplib.SMTP_SSL(smtp_host, smtp_port) as smtp:
    smtp.login(smtp_username, smtp_password)
    smtp.send_message(message, to_addrs=all_recipients)
```

BCC recipients should not be added to the email header. They should only be passed to `send_message()` through `to_addrs`.

## Send an HTML Email

We can send HTML email by adding an alternative HTML body.

```python
message = EmailMessage()
message["From"] = sender
message["To"] = "receiver@example.com"
message["Subject"] = "HTML Email from Python"

message.set_content(
    "Hello,\n\nYour email client does not support HTML."
)

message.add_alternative(
    """
    <html>
      <body>
        <h2>Hello from Python</h2>
        <p>This is an <b>HTML email</b> sent using smtplib.</p>
      </body>
    </html>
    """,
    subtype="html"
)

with smtplib.SMTP_SSL(smtp_host, smtp_port) as smtp:
    smtp.login(smtp_username, smtp_password)
    smtp.send_message(message)
```

This creates both a plain text and HTML version.

## Send Email with Attachment

To attach a file, we read the file as bytes and attach it to the message.

```python
import mimetypes
from pathlib import Path


def add_attachment(message, file_path):
    file_path = Path(file_path)

    mime_type, _ = mimetypes.guess_type(file_path)

    if mime_type is None:
        mime_type = "application/octet-stream"

    main_type, sub_type = mime_type.split("/", 1)

    with open(file_path, "rb") as file:
        message.add_attachment(
            file.read(),
            maintype=main_type,
            subtype=sub_type,
            filename=file_path.name
        )
```

Use it:

```python
message = EmailMessage()
message["From"] = sender
message["To"] = "receiver@example.com"
message["Subject"] = "Email with Attachment"

message.set_content(
    "Hello,\n\nPlease find the attachment."
)

add_attachment(message, "report.pdf")

with smtplib.SMTP_SSL(smtp_host, smtp_port) as smtp:
    smtp.login(smtp_username, smtp_password)
    smtp.send_message(message)
```

## Send Multiple Attachments

```python
files = [
    "report.pdf",
    "image.png",
    "data.csv"
]

message = EmailMessage()
message["From"] = sender
message["To"] = "receiver@example.com"
message["Subject"] = "Email with Multiple Attachments"

message.set_content(
    "Hello,\n\nPlease find the attached files."
)

for file_path in files:
    add_attachment(message, file_path)

with smtplib.SMTP_SSL(smtp_host, smtp_port) as smtp:
    smtp.login(smtp_username, smtp_password)
    smtp.send_message(message)
```

## Send All Files from a Folder

The original version of this blog sent all files from a folder. Here is a cleaner version.

```python
def add_folder_attachments(message, folder_path):
    folder_path = Path(folder_path)

    for file_path in folder_path.iterdir():
        if file_path.is_file():
            add_attachment(message, file_path)
```

Use it:

```python
message = EmailMessage()
message["From"] = sender
message["To"] = "receiver@example.com"
message["Subject"] = "Folder Attachments"

message.set_content(
    "Hello,\n\nAll files from the folder are attached."
)

add_folder_attachments(message, "attachments")

with smtplib.SMTP_SSL(smtp_host, smtp_port) as smtp:
    smtp.login(smtp_username, smtp_password)
    smtp.send_message(message)
```

## Create a Reusable Mailer Class

Now let's build a reusable class.

```python
import mimetypes
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path


class Mailer:
    """Simple SMTP mailer using Python smtplib."""

    def __init__(
        self,
        host=None,
        port=None,
        username=None,
        password=None,
        sender=None,
        use_ssl=True,
    ):
        self.host = host or os.environ["SMTP_HOST"]
        self.port = int(port or os.environ.get("SMTP_PORT", "465"))
        self.username = username or os.environ["SMTP_USERNAME"]
        self.password = password or os.environ["SMTP_PASSWORD"]
        self.sender = sender or os.environ.get("SMTP_SENDER", self.username)
        self.use_ssl = use_ssl

    def _connect(self):
        if self.use_ssl:
            smtp = smtplib.SMTP_SSL(self.host, self.port)
        else:
            smtp = smtplib.SMTP(self.host, self.port)
            smtp.starttls()

        smtp.login(self.username, self.password)

        return smtp

    def _normalize_recipients(self, emails):
        if isinstance(emails, str):
            return [emails]

        return list(emails)

    def _add_attachment(self, message, file_path):
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Attachment not found: {file_path}")

        mime_type, _ = mimetypes.guess_type(file_path)

        if mime_type is None:
            mime_type = "application/octet-stream"

        main_type, sub_type = mime_type.split("/", 1)

        with open(file_path, "rb") as file:
            message.add_attachment(
                file.read(),
                maintype=main_type,
                subtype=sub_type,
                filename=file_path.name
            )

    def send_text(self, emails, subject, content):
        recipients = self._normalize_recipients(emails)

        message = EmailMessage()
        message["From"] = self.sender
        message["To"] = ", ".join(recipients)
        message["Subject"] = subject

        message.set_content(content)

        with self._connect() as smtp:
            smtp.send_message(message)

        print(f"Sent `{subject}` to {len(recipients)} recipient(s).")

    def send_html(self, emails, subject, text_content, html_content):
        recipients = self._normalize_recipients(emails)

        message = EmailMessage()
        message["From"] = self.sender
        message["To"] = ", ".join(recipients)
        message["Subject"] = subject

        message.set_content(text_content)
        message.add_alternative(html_content, subtype="html")

        with self._connect() as smtp:
            smtp.send_message(message)

        print(f"Sent HTML email `{subject}` to {len(recipients)} recipient(s).")

    def send_files(self, emails, subject, content, files):
        recipients = self._normalize_recipients(emails)

        message = EmailMessage()
        message["From"] = self.sender
        message["To"] = ", ".join(recipients)
        message["Subject"] = subject

        message.set_content(content)

        for file_path in files:
            self._add_attachment(message, file_path)

        with self._connect() as smtp:
            smtp.send_message(message)

        print(f"Sent `{subject}` with {len(files)} attachment(s).")

    def send_folder(self, emails, subject, content, folder):
        folder = Path(folder)

        files = [
            file_path
            for file_path in folder.iterdir()
            if file_path.is_file()
        ]

        self.send_files(
            emails=emails,
            subject=subject,
            content=content,
            files=files
        )
```

## Use the Mailer Class

```python
from dotenv import load_dotenv

load_dotenv()

mailer = Mailer()

mailer.send_text(
    emails=["receiver@example.com"],
    subject="Simple Test Email",
    content="Hey,\n\nThis is just a test email."
)
```

Send files:

```python
mailer.send_files(
    emails=["receiver@example.com"],
    subject="Attached Test Email",
    content="Hey,\n\nPlease find the attached file.",
    files=["report.pdf", "image.png"]
)
```

Send all files from a folder:

```python
mailer.send_folder(
    emails=["receiver@example.com"],
    subject="Folder Attachment Email",
    content="Hey,\n\nPlease find all files from the folder attached.",
    folder="attachments"
)
```

## Example Screenshots

Simple email:

![]({{site.url}}/assets/smtplib/simple.png)

Email with attachment:

![]({{site.url}}/assets/smtplib/attached.png)

## Gmail SMTP Setup

For Gmail, the common SMTP settings are:

```text
SMTP_HOST=smtp.gmail.com
SMTP_PORT=465
```

or for STARTTLS:

```text
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
```

If using port `465`, use `SMTP_SSL`.

If using port `587`, use `SMTP` with `starttls()`.

Example for SSL:

```python
with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
    smtp.login(username, password)
    smtp.send_message(message)
```

Example for STARTTLS:

```python
with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
    smtp.starttls()
    smtp.login(username, password)
    smtp.send_message(message)
```

## App Passwords for Gmail

For many Gmail accounts, you cannot simply use your normal password in a Python script.

If your account supports app passwords, you may create one from your Google Account security settings. App passwords usually require 2-Step Verification.

Use the generated app password as:

```text
SMTP_PASSWORD=your_16_digit_app_password
```

Do not store it directly in code.

For production applications, consider using OAuth or a transactional email service instead of a personal Gmail account.

## Common Errors and Fixes

### Error 1: Authentication Failed

Possible reasons:

- wrong email address
- wrong password or app password
- app password not enabled
- 2-Step Verification not enabled
- provider blocks SMTP login
- using wrong port or security mode

### Error 2: Connection Refused

Check:

- SMTP host
- SMTP port
- firewall
- internet connection
- provider settings

### Error 3: Attachment Not Found

Check the file path.

```python
Path("report.pdf").exists()
```

### Error 4: Email Goes to Spam

Possible reasons:

- suspicious sender
- missing SPF/DKIM/DMARC records
- too many emails
- poor subject line
- sending from personal Gmail in automated scripts
- attachment type is suspicious

### Error 5: Unicode Error

Use `EmailMessage` and Python strings. It handles most normal text better than manually building raw email strings.

## Avoid These Mistakes

Avoid:

- hardcoding passwords
- committing `.env` to GitHub
- sending bulk spam
- using broad `except` without logging
- attaching huge files
- using personal Gmail for production email
- putting BCC recipients in the visible header
- ignoring provider limits

## When Should You Use smtplib?

Use `smtplib` when:

- you want a simple script
- you control the SMTP server
- you are sending small numbers of emails
- you need a lightweight dependency-free solution
- you are learning how email sending works

For production systems, you may prefer:

- SendGrid
- Mailgun
- Amazon SES
- Postmark
- Resend
- Mailchimp Transactional
- provider-specific APIs

SMTP is simple, but email deliverability is a larger topic.

## Full Example File

Here is a complete script.

```python
import mimetypes
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


class Mailer:
    """Simple SMTP mailer using Python smtplib."""

    def __init__(
        self,
        host=None,
        port=None,
        username=None,
        password=None,
        sender=None,
        use_ssl=True,
    ):
        self.host = host or os.environ["SMTP_HOST"]
        self.port = int(port or os.environ.get("SMTP_PORT", "465"))
        self.username = username or os.environ["SMTP_USERNAME"]
        self.password = password or os.environ["SMTP_PASSWORD"]
        self.sender = sender or os.environ.get("SMTP_SENDER", self.username)
        self.use_ssl = use_ssl

    def _connect(self):
        if self.use_ssl:
            smtp = smtplib.SMTP_SSL(self.host, self.port)
        else:
            smtp = smtplib.SMTP(self.host, self.port)
            smtp.starttls()

        smtp.login(self.username, self.password)

        return smtp

    def _normalize_recipients(self, emails):
        if isinstance(emails, str):
            return [emails]

        return list(emails)

    def _add_attachment(self, message, file_path):
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Attachment not found: {file_path}")

        mime_type, _ = mimetypes.guess_type(file_path)

        if mime_type is None:
            mime_type = "application/octet-stream"

        main_type, sub_type = mime_type.split("/", 1)

        with open(file_path, "rb") as file:
            message.add_attachment(
                file.read(),
                maintype=main_type,
                subtype=sub_type,
                filename=file_path.name
            )

    def send_text(self, emails, subject, content):
        recipients = self._normalize_recipients(emails)

        message = EmailMessage()
        message["From"] = self.sender
        message["To"] = ", ".join(recipients)
        message["Subject"] = subject

        message.set_content(content)

        with self._connect() as smtp:
            smtp.send_message(message)

        print(f"Sent `{subject}` to {len(recipients)} recipient(s).")

    def send_files(self, emails, subject, content, files):
        recipients = self._normalize_recipients(emails)

        message = EmailMessage()
        message["From"] = self.sender
        message["To"] = ", ".join(recipients)
        message["Subject"] = subject

        message.set_content(content)

        for file_path in files:
            self._add_attachment(message, file_path)

        with self._connect() as smtp:
            smtp.send_message(message)

        print(f"Sent `{subject}` with {len(files)} attachment(s).")


if __name__ == "__main__":
    mailer = Mailer()

    mailer.send_text(
        emails=["receiver@example.com"],
        subject="Simple Test Email",
        content="Hey,\n\nThis is just a test email."
    )

    mailer.send_files(
        emails=["receiver@example.com"],
        subject="Attached Test Email",
        content="Hey,\n\nPlease find the attached file.",
        files=["report.pdf"]
    )
```

## Final Thoughts

In this blog, we learned how to **send emails with Python using `smtplib`**. We started with a simple plain text email, then sent HTML email, multiple-recipient email, and emails with attachments.

The most important improvement over the older version is security. Do not write your email password directly inside the code. Use environment variables, app passwords where appropriate, or OAuth for production.

`smtplib` is a useful Python module for learning and small automation scripts. For production email systems, also think about deliverability, authentication, logging, rate limits, and using a dedicated email provider.
