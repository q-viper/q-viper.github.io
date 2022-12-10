---
title:  "Simple Cryptography Algorithms in Python"
date:   2022-11-22 09:29:17
categories:
    - cryptography
tags:
    - cryptography
    - python
header:
  teaser: assets/python/rot13.png
---

Cryptography Algorithms have been around the world for more than centuries and there are still many inscriptions around various places in the world which we do not understand. Here in this blog, we will cover very basic cryptography algorithms in Python.
 
But if you are interested into learning how to do encryption/decryption in image as well, i have following two blogs:
* [Run Length Encoding in Python]()
* [Huffman Encoding in Python]()
 
## Introduction
This is not a complex and huge blog about Cryptography but I am trying to write some codes on python to perform Encryption/Decryption of plain text using basic algorithms.
 
 
Few terminologies on Cryptography are:
* **Plain Text**: An input text that has to be encrypted.
* **Cipher Text**: An output text generated after encryption.
* **Key**: A value to do encryption on plain text to get cipher text. This same value was used to get plain text from cipher text.
 
Encryption is done by several minor steps within it. We first start encryption by determining an algorithm, that algorithm should be acceptable by both sender and receiver side. After that if that encryption algorithm requires a key, then sender/receiver both should have that because sender has to make cipher text using that key and receiver has to use that key to get plain text. Encryption allows us to have our privacy preserved in case of the message breach.
 
## Reverse Cipher
Easy Peasy, reverse whatever is there. Reverse Cipher is the most simplest and easiest algorithm to code because our task is to reverse the plain text and we don't need any key for doing that.
 
 
```python
def reverse_cipher(msg, mode="encrypt"):  
    msg = msg
    if mode == "encrypt":
        return msg[::-1]
    elif mode == "decrypt":
        return msg[::-1]
    else:
        return "mode unknown!"
   
```
 
 
```python
en = reverse_cipher("Hello world!")
print(en)
de = reverse_cipher(en, "decrypt")
print(de)
```
 
    !dlrow olleH
    Hello world!
   
 
## Caesar Cipher
Uses shift value to shift character's alphanumeric value for encrypt/decrypt. This algorithm is a bit harder than Reverse Cipher because this requires us to convert text to numeric value and then shift that numeric value by some number. The number can be called as key or simply shift. In encryption, we decrease the numeric value of character and on decrypt we increase numeric value or vice versa.
 
 
```python
def caeser_cipher(msg, mode="encrypt", shift=2):
    text = ""
    for m in msg:
        c = ord(m)
        if mode =="encrypt":
            c = c-shift
        elif mode == "decrypt":
            c = c + shift
        else:
            print("mode unknown!")
        text+=chr(c)
    return text
        #print(c)
en = caeser_cipher("Hello world!")
print(en)
de = caeser_cipher(en, "decrypt")
print(de)
```
 
    Fcjjmumpjb
    Hello world!
   
 
## ROT13 Algorithm
Rotate every character by 13 to encrypt/decrypt.
 
 
```python
def rot13(msg, mode="encrypt"):
    main_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    trans_chars = "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm"
    text = ""
    special_chrs = "?.!><# "
    for m in msg:
        if m in special_chrs:
            text+=m
        if m.isnumeric():
            text+=m
        if mode == "encrypt":
            if m in main_chars:
                text+=trans_chars[main_chars.find(m)]
        elif mode == "decrypt":
            if m in trans_chars:
                text+=main_chars[trans_chars.find(m)]
    return text
 
en = rot13("hello world from rot13!!!")
print(en)
de = rot13(en, "decrypt")
print(de)
```
 
    uryyb jbeyq sebz ebg13!!!
    hello world from rot13!!!
   
 
 
```python
rot13('This is blog about Cryptography')
```
 
 
 
 
    'Guvf vf oybt nobhg Pelcgbtencul'
 
 
 
This ends the simple cryptography algorithms in Python and there are some popular ways of applying encryption/decryption in Python and these are using following packages:
* `pip install rsa`
* `pip install cryptography`
 

