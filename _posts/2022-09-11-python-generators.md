---
layout: single
title: "Python Generators: Beginner Guide to yield, Iterators, Generator Expressions, and Memory-Efficient Code"
date: 2022-09-11 01:29:17 +0545
last_modified_at: 2026-06-16
categories:
  - Python
  - Tutorial
tags:
  - "Python"
  - "Generators"
  - "yield"
  - "Iterators"
  - "Generator expressions"
  - "Memory efficient Python"
  - "Python functions"
description: "A beginner-friendly guide to Python generators, covering yield, iterators, next(), StopIteration, generator expressions, memory efficiency, file reading, yield from, and common mistakes."
excerpt: "Learn how Python generators work using the yield keyword, how they differ from normal functions and lists, and why they are useful for memory-efficient programming."
header:
  teaser: assets/python/pygen.png
  og_image: assets/python/pygen.png
  image_description: "Python generator tutorial illustration"
toc: true
toc_label: "Python Generators"
toc_icon: "code"
toc_sticky: true
read_time: true
share: true
related: true
classes: wide
---

**Python generators** are a special type of iterator. They allow us to produce values one at a time instead of creating and storing the full result in memory.

In simple words:

> A generator is a function that uses `yield` instead of `return`.

Generators are useful when we are working with large data, streaming data, files, images, API responses, or any task where loading everything into memory at once would be wasteful.

For example, in deep learning, we may need to train a model on thousands of images. Loading all images into memory at once can crash the system. A generator can load only one batch at a time, send it to the model, and then load the next batch.

I used a similar idea in a custom data generator for image classification here:

- [Corn Leaf Infection Detection: Data Preprocessing and Custom DataGenerator](https://dataqoil.com/2020/10/20/corn-leaf-infection-detection-data-preprocessing-and-custom-datagenerator/)

## What This Tutorial Covers

In this post, we will learn:

- what Python generators are
- how `yield` works
- generator functions vs normal functions
- how to use `next()`
- why `StopIteration` happens
- how to loop over generators
- how generator expressions work
- why generators are memory efficient
- practical examples of generators
- how to read large files with generators
- how to use `yield from`
- common generator mistakes

## What Is a Generator in Python?

A generator is a function that returns values one by one.

A normal function uses `return`:

```python
def normal_function():
    return 10
```

A generator function uses `yield`:

```python
def generator_function():
    yield 10
```

When a normal function is called, it runs and returns a value immediately.

When a generator function is called, it does not run immediately. Instead, it returns a generator object.

## Basic Generator Syntax

The basic structure of a generator is:

```python
def my_generator():
    yield value
```

Example:

```python
def simple_generator():
    yield 1
    yield 2
    yield 3
```

Now call it:

```python
gen = simple_generator()

print(gen)
```

Output:

```text
<generator object simple_generator at ...>
```

This means the function has created a generator object. The values are not produced yet.

## Using next() with a Generator

We can get values from a generator using `next()`.

```python
gen = simple_generator()

print(next(gen))
print(next(gen))
print(next(gen))
```

Output:

```text
1
2
3
```

Each call to `next()` continues the function from where it stopped.

## StopIteration Error

If we call `next()` again after all values are used, Python raises `StopIteration`.

```python
print(next(gen))
```

Output:

```text
StopIteration
```

This means there are no more values left in the generator.

## Generator Function vs Normal Function

Let's create two functions that do the same task: return square numbers.

The normal function returns a list.

```python
def my_pow_fun(x: int):
    return [i * i for i in range(x)]
```

The generator function yields one value at a time.

```python
def my_pow_gen(x: int):
    for i in range(x):
        yield i * i
```

Now test both:

```python
print(my_pow_fun(5))
print(my_pow_gen(5))
```

Output:

```text
[0, 1, 4, 9, 16]
<generator object my_pow_gen at ...>
```

The normal function immediately returns the full list.

The generator function returns a generator object.

## Getting Items from a Generator

Let's store the generator object in a variable.

```python
mpg = my_pow_gen(5)
```

Now get values one by one.

```python
print(next(mpg))
print(next(mpg))
print(next(mpg))
```

Output:

```text
0
1
4
```

The generator remembers where it stopped. It does not start again from the beginning unless we create a new generator object.

## Looping Over a Generator

We can also loop over a generator.

```python
for value in mpg:
    print(value)
```

Output:

```text
9
16
```

The loop starts from the next available value. Since we already consumed `0`, `1`, and `4`, the loop only prints the remaining values.

## Generators Are Exhausted After Use

A generator can be consumed only once.

```python
mpg = my_pow_gen(5)

print(list(mpg))
print(list(mpg))
```

Output:

```text
[0, 1, 4, 9, 16]
[]
```

The second list is empty because the generator has already been exhausted.

To use it again, create a new generator:

```python
mpg = my_pow_gen(5)
```

## Convert a Generator to a List

We can convert a generator into a list using `list()`.

```python
mpg = my_pow_gen(10)

values = list(mpg)

print(values)
```

Output:

```text
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

This is useful for debugging or small data.

But if the data is large, converting a generator to a list removes the memory benefit because all values are stored at once.

## Why Use Generators?

Generators are useful because they are lazy.

Lazy means values are created only when needed.

This gives several benefits:

- less memory usage
- better performance for large data streams
- cleaner code for pipelines
- ability to work with infinite sequences
- easier file processing
- useful for batch loading data

## Memory Efficiency Example

A list stores all values in memory.

```python
numbers = [i * i for i in range(1_000_000)]
```

A generator produces values one at a time.

```python
numbers = (i * i for i in range(1_000_000))
```

The generator expression does not immediately create one million values. It creates them only when we iterate over it.

## Generator Expression

A generator expression is like a list comprehension, but it uses parentheses instead of square brackets.

List comprehension:

```python
squares_list = [i * i for i in range(10)]
```

Generator expression:

```python
squares_gen = (i * i for i in range(10))
```

Now use it:

```python
for value in squares_gen:
    print(value)
```

Generator expressions are useful for short and simple generator logic.

## List Comprehension vs Generator Expression

| Feature | List Comprehension | Generator Expression |
|---|---|---|
| Syntax | `[x for x in data]` | `(x for x in data)` |
| Memory | Stores all values | Produces one value at a time |
| Reusable | Yes | No, consumed once |
| Good for | Small to medium data | Large or streaming data |

Example:

```python
# List
values = [x * 2 for x in range(5)]

# Generator
values_gen = (x * 2 for x in range(5))
```

## Reading a Large File with a Generator

Generators are very useful for reading large files.

Instead of reading the whole file:

```python
with open("large_file.txt") as file:
    lines = file.readlines()
```

we can yield one line at a time:

```python
def read_large_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            yield line.strip()
```

Use it:

```python
for line in read_large_file("large_file.txt"):
    print(line)
```

This is memory efficient because it reads one line at a time.

## Generator for Batches

A common use of generators is creating batches of data.

```python
def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]
```

Example:

```python
numbers = list(range(20))

for batch in batch_generator(numbers, batch_size=5):
    print(batch)
```

Output:

```text
[0, 1, 2, 3, 4]
[5, 6, 7, 8, 9]
[10, 11, 12, 13, 14]
[15, 16, 17, 18, 19]
```

This idea is common in machine learning, where we train models using batches instead of loading all data at once.

## Infinite Generator

Generators can also create infinite sequences.

```python
def count_forever(start=0):
    number = start

    while True:
        yield number
        number += 1
```

Use it carefully:

```python
counter = count_forever()

print(next(counter))
print(next(counter))
print(next(counter))
```

Output:

```text
0
1
2
```

Do not convert an infinite generator to a list. It will never finish.

## Using yield from

The `yield from` syntax is used to yield values from another iterable or generator.

Example:

```python
def generator_a():
    yield 1
    yield 2


def generator_b():
    yield from generator_a()
    yield 3
    yield 4
```

Use it:

```python
print(list(generator_b()))
```

Output:

```text
[1, 2, 3, 4]
```

This is useful when combining multiple generators.

## Generator Pipeline Example

Generators can be combined into pipelines.

```python
def numbers(limit):
    for i in range(limit):
        yield i


def only_even(values):
    for value in values:
        if value % 2 == 0:
            yield value


def square(values):
    for value in values:
        yield value * value
```

Use the pipeline:

```python
result = square(only_even(numbers(10)))

print(list(result))
```

Output:

```text
[0, 4, 16, 36, 64]
```

This style is useful for data processing.

## return Inside a Generator

A generator can use `return`, but it stops the generator.

```python
def gen_with_return():
    yield 1
    yield 2
    return
    yield 3
```

Use it:

```python
print(list(gen_with_return()))
```

Output:

```text
[1, 2]
```

The value after `return` is never yielded.

## When Not to Use Generators

Generators are not always the best option.

Avoid generators when:

- you need to access values multiple times
- you need random indexing
- the data is small and simple
- you need to sort all values
- you need the total length before processing
- readability becomes worse

For small data, a list is often simpler.

## Common Mistakes with Generators

Here are some common beginner mistakes:

### Mistake 1: Expecting a Generator to Print Values Directly

```python
print(my_pow_gen(5))
```

This prints the generator object, not the values.

Use:

```python
print(list(my_pow_gen(5)))
```

or:

```python
for value in my_pow_gen(5):
    print(value)
```

### Mistake 2: Reusing an Exhausted Generator

```python
gen = my_pow_gen(5)

print(list(gen))
print(list(gen))
```

The second output is empty.

Create a new generator if you need to iterate again.

### Mistake 3: Converting a Huge Generator to a List

```python
list(my_pow_gen(100_000_000))
```

This can use too much memory.

Use a loop instead.

### Mistake 4: Using Generators When Indexing Is Needed

Generators do not support direct indexing.

```python
gen = my_pow_gen(5)

# This will not work
gen[0]
```

Use a list if you need indexing.

## Practical Use Cases of Python Generators

Generators are useful for:

- reading large files
- processing logs
- streaming API responses
- loading image batches
- machine learning data pipelines
- web scraping pipelines
- processing CSV files row by row
- generating infinite sequences
- lazy data transformations
- memory-efficient loops

## Final Thoughts

In this post, we learned the basics of **Python generators**. A generator is a function that uses `yield` to produce values one at a time. This makes generators memory efficient and useful for large data processing.

The most important points are:

- `yield` pauses the function and returns one value
- `next()` gets the next value
- generators are exhausted after one full iteration
- generator expressions are useful for short generator logic
- generators are useful when we do not want to store everything in memory

Generators are simple once we understand them, and they are one of the most useful features in Python for writing clean and memory-efficient code.
