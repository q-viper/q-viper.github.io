---
layout: single
title: "Simple Cryptography Algorithms in Python: Reverse Cipher, Caesar Cipher, and ROT13"
date: 2022-11-22 09:29:17 +0100
last_modified_at: 2026-06-16
categories:
  - Python
  - Cryptography
  - Tutorial
tags:
  - "Cryptography"
  - "Python"
  - "Encryption"
  - "Decryption"
  - "Caesar cipher"
  - "ROT13"
  - "Reverse cipher"
  - "Beginner Python"
description: "Learn simple cryptography algorithms in Python, including reverse cipher, Caesar cipher, and ROT13, with beginner-friendly explanations and code examples."
excerpt: "A beginner-friendly Python tutorial on simple cryptography algorithms such as reverse cipher, Caesar cipher, and ROT13, including encryption, decryption, and limitations."
header:
  teaser: /assets/python/rot13.png
  og_image: /assets/python/rot13.png
  image_description: "ROT13 cryptography example in Python"
toc: true
toc_label: "Simple Cryptography in Python"
toc_icon: "lock"
toc_sticky: true
read_time: true
share: true
related: true
classes: wide
---

Cryptography is the practice of protecting information by converting readable text into a form that is difficult to understand without the correct method or key. It has been used for centuries, from ancient secret messages to modern secure communication on the internet.

In this post, I will explain and implement a few **simple cryptography algorithms in Python**. These are not secure enough for real-world use, but they are great for learning how encryption and decryption work at a basic level.

We will cover:

- Reverse Cipher
- Caesar Cipher
- ROT13
- simple encryption and decryption functions
- limitations of these basic algorithms
- what to use for real cryptography in Python

If you are interested in compression or encoding methods for images, you may also like:

- [Run Length Encoding in Python]()
- [Huffman Encoding in Python]()

## Important Note

The algorithms in this blog are for learning only. Reverse Cipher, Caesar Cipher, and ROT13 are **toy ciphers**. They are easy to break and should not be used for passwords, private messages, API keys, or real security systems.

For real cryptography in Python, use trusted libraries such as:

```bash
pip install cryptography
````

or, depending on the task:

```bash
pip install pycryptodome
```

Do not design your own encryption system for production unless you are trained in cryptography and security engineering.

## Basic Cryptography Terms

Before writing code, let's define a few common terms.

### Plaintext

Plaintext is the original readable message.

Example:

```text
Hello world
```

### Ciphertext

Ciphertext is the encrypted message.

Example:

```text
Khoor zruog
```

### Encryption

Encryption is the process of converting plaintext into ciphertext.

### Decryption

Decryption is the process of converting ciphertext back into plaintext.

### Key

A key is a value used by an algorithm to encrypt or decrypt data.

For example, in a Caesar Cipher, the key can be the number of letters we shift.

### Cipher

A cipher is the algorithm used for encryption and decryption.

## Why Learn Simple Cryptography Algorithms?

Simple cryptography algorithms are not secure, but they are still useful for learning.

They help us understand:

* how text can be transformed
* how keys affect encryption
* why weak ciphers are easy to break
* how encryption and decryption are related
* why modern cryptography is much more complex

These examples are also good Python exercises because they use strings, loops, functions, conditions, and character operations.

## Reverse Cipher in Python

The **Reverse Cipher** is one of the simplest ciphers. It encrypts a message by reversing the order of characters.

For example:

```text
Hello world
```

becomes:

```text
dlrow olleH
```

The same operation is used for encryption and decryption.

### Reverse Cipher Code

```python
def reverse_cipher(message):
    """Encrypt or decrypt a message by reversing it."""
    return message[::-1]
```

### Reverse Cipher Example

```python
plain_text = "Hello world!"

encrypted_text = reverse_cipher(plain_text)
decrypted_text = reverse_cipher(encrypted_text)

print("Plain text:", plain_text)
print("Encrypted text:", encrypted_text)
print("Decrypted text:", decrypted_text)
```

Output:

```text
Plain text: Hello world!
Encrypted text: !dlrow olleH
Decrypted text: Hello world!
```

### How Reverse Cipher Works

Python slicing makes this very easy:

```python
message[::-1]
```

This means: start from the end of the string and move backwards.

### Limitation of Reverse Cipher

Reverse Cipher is very easy to break. Anyone can simply reverse the message again and read it. It does not use a secret key, so it gives almost no real security.

## Caesar Cipher in Python

The **Caesar Cipher** is a classic substitution cipher. It shifts each letter by a fixed number of positions in the alphabet.

For example, with a shift of `3`:

```text
A -> D
B -> E
C -> F
```

So:

```text
HELLO
```

becomes:

```text
KHOOR
```

The shift value is the key.

## Caesar Cipher: Simple Version

A very simple Caesar Cipher can be written using `ord()` and `chr()`.

* `ord()` converts a character to its Unicode number
* `chr()` converts a Unicode number back to a character

```python
def simple_caesar_cipher(message, mode="encrypt", shift=2):
    """A simple Caesar Cipher using Unicode shifting."""
    text = ""

    for char in message:
        code = ord(char)

        if mode == "encrypt":
            code = code + shift
        elif mode == "decrypt":
            code = code - shift
        else:
            raise ValueError("mode must be either 'encrypt' or 'decrypt'")

        text += chr(code)

    return text
```

Example:

```python
plain_text = "Hello world!"

encrypted_text = simple_caesar_cipher(plain_text, mode="encrypt", shift=2)
decrypted_text = simple_caesar_cipher(encrypted_text, mode="decrypt", shift=2)

print(encrypted_text)
print(decrypted_text)
```

Output:

```text
Jgnnq"yqtnf#
Hello world!
```

This works, but it also shifts spaces and punctuation. That is not always what we want.

## Caesar Cipher: Better Alphabet-Based Version

A better Caesar Cipher only shifts letters and leaves spaces, numbers, and punctuation unchanged.

```python
def caesar_cipher(message, shift=3, mode="encrypt"):
    """Encrypt or decrypt text using the Caesar Cipher.

    Only alphabetic characters are shifted.
    Spaces, numbers, and punctuation are kept unchanged.
    """
    result = ""

    if mode == "decrypt":
        shift = -shift
    elif mode != "encrypt":
        raise ValueError("mode must be either 'encrypt' or 'decrypt'")

    for char in message:
        if char.isupper():
            start = ord("A")
            result += chr((ord(char) - start + shift) % 26 + start)

        elif char.islower():
            start = ord("a")
            result += chr((ord(char) - start + shift) % 26 + start)

        else:
            result += char

    return result
```

### Caesar Cipher Example

```python
plain_text = "Hello world!"

encrypted_text = caesar_cipher(plain_text, shift=3, mode="encrypt")
decrypted_text = caesar_cipher(encrypted_text, shift=3, mode="decrypt")

print("Plain text:", plain_text)
print("Encrypted text:", encrypted_text)
print("Decrypted text:", decrypted_text)
```

Output:

```text
Plain text: Hello world!
Encrypted text: Khoor zruog!
Decrypted text: Hello world!
```

### Why We Use Modulo in Caesar Cipher

This part is important:

```python
(ord(char) - start + shift) % 26 + start
```

The modulo operator `% 26` makes sure the alphabet wraps around.

For example, if we shift `Z` by `3`, we want:

```text
Z -> C
```

Without modulo, the result would go outside the alphabet.

## Caesar Cipher Brute Force Attack

The Caesar Cipher is weak because there are only 25 useful shifts. That means we can try every possible key.

```python
def brute_force_caesar(cipher_text):
    """Print all possible Caesar Cipher decryptions."""
    for shift in range(1, 26):
        decrypted = caesar_cipher(cipher_text, shift=shift, mode="decrypt")
        print(f"Shift {shift}: {decrypted}")
```

Example:

```python
encrypted_text = "Khoor zruog!"
brute_force_caesar(encrypted_text)
```

Output:

```text
Shift 1: Jgnnq yqtnf!
Shift 2: Ifmmp xpsme!
Shift 3: Hello world!
...
```

This shows why Caesar Cipher should not be used for real security. It can be broken very quickly.

## ROT13 in Python

**ROT13** means "rotate by 13 places." It is a special version of Caesar Cipher where the shift is always `13`.

Because the English alphabet has 26 letters, applying ROT13 twice gives the original text back.

```text
ROT13(ROT13(text)) = text
```

This means the same function can be used for both encryption and decryption.

## ROT13 Code

```python
def rot13(message):
    """Apply ROT13 encoding or decoding to a message."""
    result = ""

    for char in message:
        if char.isupper():
            start = ord("A")
            result += chr((ord(char) - start + 13) % 26 + start)

        elif char.islower():
            start = ord("a")
            result += chr((ord(char) - start + 13) % 26 + start)

        else:
            result += char

    return result
```

### ROT13 Example

```python
plain_text = "hello world from rot13!!!"

encrypted_text = rot13(plain_text)
decrypted_text = rot13(encrypted_text)

print(encrypted_text)
print(decrypted_text)
```

Output:

```text
uryyb jbeyq sebz ebg13!!!
hello world from rot13!!!
```

Another example:

```python
rot13("This is blog about Cryptography")
```

Output:

```text
Guvf vf oybt nobhg Pelcgbtencul
```

## ROT13 Using Python's Built-in Codec

Python also has a built-in way to apply ROT13 using the `codecs` module.

```python
import codecs

text = "This is blog about Cryptography"

encoded = codecs.encode(text, "rot_13")
decoded = codecs.decode(encoded, "rot_13")

print(encoded)
print(decoded)
```

Output:

```text
Guvf vf oybt nobhg Pelcgbtencul
This is blog about Cryptography
```

This is shorter, but writing the function manually helps us understand how the algorithm works.

## Comparing Reverse Cipher, Caesar Cipher, and ROT13

| Algorithm      | Uses Key? | Easy to Code? | Easy to Break? | Main Idea                      |
| -------------- | --------: | ------------: | -------------: | ------------------------------ |
| Reverse Cipher |        No |           Yes |            Yes | Reverse the message            |
| Caesar Cipher  |       Yes |           Yes |            Yes | Shift letters by a fixed value |
| ROT13          | Fixed key |           Yes |            Yes | Shift letters by 13 positions  |

All three are useful for learning, but none of them should be used for real encryption.

## Full Example Script

Here is a small complete script that combines all three algorithms.

```python
def reverse_cipher(message):
    return message[::-1]


def caesar_cipher(message, shift=3, mode="encrypt"):
    result = ""

    if mode == "decrypt":
        shift = -shift
    elif mode != "encrypt":
        raise ValueError("mode must be either 'encrypt' or 'decrypt'")

    for char in message:
        if char.isupper():
            start = ord("A")
            result += chr((ord(char) - start + shift) % 26 + start)
        elif char.islower():
            start = ord("a")
            result += chr((ord(char) - start + shift) % 26 + start)
        else:
            result += char

    return result


def rot13(message):
    return caesar_cipher(message, shift=13, mode="encrypt")


message = "Hello world!"

reverse_encrypted = reverse_cipher(message)
reverse_decrypted = reverse_cipher(reverse_encrypted)

caesar_encrypted = caesar_cipher(message, shift=3, mode="encrypt")
caesar_decrypted = caesar_cipher(caesar_encrypted, shift=3, mode="decrypt")

rot13_encrypted = rot13(message)
rot13_decrypted = rot13(rot13_encrypted)

print("Reverse:", reverse_encrypted, "->", reverse_decrypted)
print("Caesar:", caesar_encrypted, "->", caesar_decrypted)
print("ROT13:", rot13_encrypted, "->", rot13_decrypted)
```

Output:

```text
Reverse: !dlrow olleH -> Hello world!
Caesar: Khoor zruog! -> Hello world!
ROT13: Uryyb jbeyq! -> Hello world!
```

## Real Encryption in Python

For real encryption, use a trusted library instead of writing your own cipher.

The most common Python package is:

```bash
pip install cryptography
```

For example, the `cryptography` package provides Fernet symmetric encryption.

```python
from cryptography.fernet import Fernet

# Generate a secret key
key = Fernet.generate_key()
cipher = Fernet(key)

message = b"Hello world!"

encrypted_message = cipher.encrypt(message)
decrypted_message = cipher.decrypt(encrypted_message)

print(encrypted_message)
print(decrypted_message.decode())
```

This is much safer than simple ciphers because it uses modern cryptographic methods through a well-maintained library.

Still, key management is very important. If someone gets your secret key, they can decrypt your messages.

## Common Mistakes When Learning Cryptography

Here are some common mistakes beginners make:

* thinking encoding and encryption are the same
* using Caesar Cipher for real secrets
* storing secret keys directly in code
* publishing API keys or passwords in GitHub
* using weak or reused keys
* creating custom encryption logic for production

Encoding is not encryption. For example, Base64 only changes how data is represented. It does not protect the message.

## Final Thoughts

In this post, we explored **simple cryptography algorithms in Python**. We implemented Reverse Cipher, Caesar Cipher, and ROT13. These algorithms are easy to understand and good for learning the basic idea of encryption and decryption.

The main lesson is that simple ciphers are fun and useful for practice, but they are not secure. For real projects, always use trusted libraries such as `cryptography`, and never design your own encryption system without proper security knowledge.
