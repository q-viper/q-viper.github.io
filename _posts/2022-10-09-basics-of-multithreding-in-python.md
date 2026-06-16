---
layout: single
title: "Basics of Multithreading in Python: Threading, ThreadPoolExecutor, Locks, Queues, and Joblib"
date: 2022-10-09 01:29:17 +0545
last_modified_at: 2026-06-16
categories:
  - Python
  - Multithreading
  - Concurrency
tags:
  - "Python"
  - "Multithreading"
  - "Threading"
  - "ThreadPoolExecutor"
  - "Joblib"
  - "Parallel processing"
  - "Concurrency"
  - "I/O bound"
  - "Multiprocessing"
description: "A beginner-friendly tutorial on multithreading in Python using threading, ThreadPoolExecutor, locks, queues, and Joblib, with examples for I/O-bound tasks."
excerpt: "Learn the basics of multithreading in Python, when to use threads, how the threading module works, how to use locks and queues, and how Joblib can simplify parallel tasks."
header:
  teaser: assets/python/threading.png
  og_image: assets/python/threading.png
  image_description: "Python threading illustration for a multithreading tutorial"
toc: true
toc_label: "Multithreading in Python"
toc_icon: "code-branch"
toc_sticky: true
read_time: true
share: true
related: true
classes: wide
---

**Multithreading in Python** is useful when a program has tasks that spend time waiting. These are usually called **I/O-bound tasks**. Examples include downloading files, reading and writing files, calling APIs, waiting for a database, or listening for user requests.

If a program is waiting for one task to finish, another thread can continue doing other work. This can make the program feel faster and more responsive.

In this post, I will explain the basics of multithreading in Python using:

- the built-in `threading` module
- `threading.Thread`
- custom thread classes
- locks for shared resources
- queues for safer communication between threads
- `ThreadPoolExecutor`
- `joblib` for simple parallel execution

This is a beginner-friendly tutorial, so the examples are simple and practical.

## When Should You Use Multithreading?

Multithreading is useful when tasks are mostly waiting for something outside the CPU.

Good examples are:

1. **Downloading images from the web**

   One thread can download images while another thread processes already downloaded images.

2. **Reading and writing files**

   One thread can write data to a file while another checks the file or processes previous data.

3. **Calling APIs**

   A trading bot, weather bot, or data pipeline may need to call many APIs. Threads can help send multiple requests without waiting for each one sequentially.

4. **Monitoring background tasks**

   For example, one thread can listen for user input while another checks order status or system events.

5. **Backtesting independent date ranges**

   If different chunks of historical data are independent, they can sometimes be processed in parallel.

6. **Keeping an application responsive**

   In desktop or server applications, background threads can keep long-running tasks away from the main program flow.

## Multithreading vs Multiprocessing vs Asyncio

Before writing code, it is useful to understand the difference between three common Python concurrency approaches.

| Approach | Best For | Main Idea |
|---|---|---|
| `threading` | I/O-bound tasks | Run multiple threads in one process |
| `multiprocessing` | CPU-bound tasks | Run multiple Python processes |
| `asyncio` | Many I/O tasks | Use one thread with an event loop |
| `joblib` | Easy parallel loops | Use threads or processes with a simple API |

For pure Python CPU-heavy work, multithreading may not speed things up much because of the **Global Interpreter Lock**, also called the **GIL**. For CPU-bound tasks, multiprocessing is often better.

For I/O-bound tasks, multithreading is still useful because threads can work while other threads are waiting.

## What Is the Python GIL?

The **Global Interpreter Lock** is a mechanism in CPython that allows only one thread to execute Python bytecode at a time.

This means:

- Python threads are good for I/O-bound work.
- Python threads are not always good for CPU-heavy pure Python loops.
- CPU-heavy work may need `multiprocessing`, vectorized NumPy code, or compiled extensions.

This does not mean Python threading is useless. It only means we should use it for the right type of task.

## Using the `threading` Module

Python has a built-in module called `threading`. We do not need to install anything.

```python
import threading
import time
```

The most common way to create a thread is:

```python
thread = threading.Thread(target=function_name, args=(arg1, arg2))
thread.start()
thread.join()
```

Here:

- `target` is the function the thread will run
- `args` contains the function arguments
- `start()` begins the thread
- `join()` waits for the thread to finish

## Simple Multithreading Example

Let's create two functions:

- one prints even numbers
- one prints odd numbers

The even function sleeps for two seconds. The odd function sleeps for one second. If they run in separate threads, one function does not fully block the other.

```python
import threading
import time


def even(limit):
    number = 0

    while number < limit:
        print(f"EVEN: {number}")
        number += 2
        time.sleep(2)


def odd(limit):
    number = 1

    while number < limit:
        print(f"ODD: {number}")
        number += 2
        time.sleep(1)


thread_1 = threading.Thread(target=even, args=(10,))
thread_2 = threading.Thread(target=odd, args=(10,))

thread_1.start()
thread_2.start()

thread_1.join()
thread_2.join()
```

Example output:

```text
EVEN: 0
ODD: 1
ODD: 3
EVEN: 2
ODD: 5
ODD: 7
EVEN: 4
ODD: 9
EVEN: 6
EVEN: 8
```

The exact output order can change because thread scheduling is handled by the operating system.

## Why Output Order Can Look Strange

When multiple threads print at the same time, the output can look mixed.

For example, you may see two lines printed together. This happens because both threads are writing to the console at almost the same time.

This is normal in multithreaded programs. If output order matters, you need synchronization.

## Using `join()`

The `join()` method tells the main program to wait until a thread is finished.

```python
thread_1.join()
thread_2.join()
```

Without `join()`, the main program may continue running before the threads are done.

## File Writer and Reader Example

Now let's try a simple example where one thread writes to a file and another thread reads from it.

```python
import threading
import time


def writer(file_id):
    file_name = f"{file_id}.txt"

    with open(file_name, "w") as fp:
        fp.write("")

    number = 0

    while number < file_id:
        with open(file_name, "a") as fp:
            fp.write(f"{number}")
            print(f"Writer N: {number} | Wrote: {number}")

        number += 1
        time.sleep(1)


def reader(file_id):
    file_name = f"{file_id}.txt"

    number = 0

    while number < file_id:
        with open(file_name, "r") as fp:
            content = fp.read()
            print(f"Reader N: {number} | Read: {content}")

        number += 1
        time.sleep(2)


thread_1 = threading.Thread(target=writer, args=(10,))
thread_2 = threading.Thread(target=reader, args=(10,))

thread_1.start()
thread_2.start()

thread_1.join()
thread_2.join()
```

In this example, the writer first creates the file. Then it keeps appending numbers. The reader reads the file every two seconds.

## A Common Threading Problem

What happens if the reader starts before the writer creates the file?

```python
thread_1 = threading.Thread(target=writer, args=(15,))
thread_2 = threading.Thread(target=reader, args=(15,))

thread_2.start()
thread_1.start()

thread_1.join()
thread_2.join()
```

This can cause an error like:

```text
FileNotFoundError: [Errno 2] No such file or directory: '15.txt'
```

The reader tries to read the file before it exists.

This is one of the most important lessons in multithreading:

> When threads share resources, timing matters.

To avoid such problems, we can use synchronization tools such as locks, events, and queues.

## Using a Lock in Python Threads

A `Lock` makes sure that only one thread can access a shared resource at a time.

This is useful when multiple threads write to the same file, update the same variable, or print logs.

```python
import threading

counter = 0
lock = threading.Lock()


def increase_counter():
    global counter

    for _ in range(100000):
        with lock:
            counter += 1


thread_1 = threading.Thread(target=increase_counter)
thread_2 = threading.Thread(target=increase_counter)

thread_1.start()
thread_2.start()

thread_1.join()
thread_2.join()

print(counter)
```

The `with lock:` block makes the update safer.

Without a lock, two threads may try to update the same variable at the same time and cause incorrect results.

## Using an Event to Signal Between Threads

An `Event` can be used when one thread needs to wait until another thread gives a signal.

For example, the reader should wait until the writer creates the file.

```python
import threading
import time

file_ready = threading.Event()


def writer_with_event(file_name):
    with open(file_name, "w") as fp:
        fp.write("")

    file_ready.set()

    for number in range(5):
        with open(file_name, "a") as fp:
            fp.write(str(number))

        print(f"Writer wrote: {number}")
        time.sleep(1)


def reader_with_event(file_name):
    file_ready.wait()

    for _ in range(5):
        with open(file_name, "r") as fp:
            print("Reader read:", fp.read())

        time.sleep(1)


thread_1 = threading.Thread(target=writer_with_event, args=("demo.txt",))
thread_2 = threading.Thread(target=reader_with_event, args=("demo.txt",))

thread_2.start()
thread_1.start()

thread_1.join()
thread_2.join()
```

Here:

- `file_ready.wait()` blocks the reader
- `file_ready.set()` tells the reader that the file is ready

## Using Queue for Safer Thread Communication

A `queue.Queue` is one of the safest ways to pass data between threads.

One thread can produce data, and another thread can consume it.

```python
import queue
import threading
import time

task_queue = queue.Queue()


def producer():
    for number in range(5):
        print(f"Producing: {number}")
        task_queue.put(number)
        time.sleep(1)

    task_queue.put(None)


def consumer():
    while True:
        item = task_queue.get()

        if item is None:
            break

        print(f"Consuming: {item}")
        task_queue.task_done()


thread_1 = threading.Thread(target=producer)
thread_2 = threading.Thread(target=consumer)

thread_1.start()
thread_2.start()

thread_1.join()
thread_2.join()
```

This pattern is useful for many real applications, such as:

- downloading files in one thread and processing them in another
- reading messages from an API and processing them
- sending tasks to background workers
- building simple pipelines

## Creating a Custom Thread Class

We can also create a custom class by inheriting from `threading.Thread`.

```python
import threading
import time


class ThreadClass(threading.Thread):
    def run(self):
        while True:
            # Put thread work here
            time.sleep(2)
            break

        print("Run ended.")


thread = ThreadClass()
thread.daemon = False

thread.start()
thread.join()
```

Output:

```text
Run ended.
```

A custom thread class is useful when the thread has its own state or repeated behavior.

## Daemon Threads

A daemon thread runs in the background and does not block the program from exiting.

```python
thread.daemon = True
```

Use daemon threads carefully. If the main program exits, daemon threads are stopped. This may interrupt file writing, logging, or cleanup operations.

For important tasks, use non-daemon threads and call `join()`.

## ThreadPoolExecutor: A Cleaner Way to Use Threads

For many tasks, using `ThreadPoolExecutor` is cleaner than manually creating threads.

```python
from concurrent.futures import ThreadPoolExecutor
import time


def root_printer(number):
    root = number ** 0.5
    print(f"Printer: {root}")
    time.sleep(root)
    return root


numbers = range(10, 1, -1)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(root_printer, numbers))

print(results)
```

This is useful when you want to run the same function on many inputs.

## Sequential Execution Example

Before using parallel execution, let's run the same function sequentially.

```python
import time


def root_printer(number):
    root = number ** 0.5
    print(f"Printer: {root}")
    time.sleep(root)
    return root


start_time = time.perf_counter()

result = list(map(root_printer, range(10, 1, -1)))

end_time = time.perf_counter()

print(f"Time: {end_time - start_time}")
print(result)
```

In the original example, this took around 21 seconds.

This is slow because each task waits for the previous task to finish.

## Using Joblib for Parallel Execution

Another option for parallel execution in Python is [Joblib](https://joblib.readthedocs.io/en/latest/).

Joblib is useful when we want to run the same function many times with different inputs.

Install it with:

```bash
pip install joblib
```

Import `Parallel` and `delayed`.

```python
from joblib import Parallel, delayed
```

Now run the same function with two jobs.

```python
start_time = time.perf_counter()

result = Parallel(n_jobs=2)(
    delayed(root_printer)(number)
    for number in range(10, 1, -1)
)

end_time = time.perf_counter()

print(f"Time: {end_time - start_time}")
print(result)
```

In the original experiment, this took around 13 seconds.

## Increasing Joblib Workers

Now let's use four jobs.

```python
start_time = time.perf_counter()

result = Parallel(n_jobs=4)(
    delayed(root_printer)(number)
    for number in range(10, 1, -1)
)

end_time = time.perf_counter()

print(f"Time: {end_time - start_time}")
print(result)
```

In the original experiment, this took around 8 seconds.

More workers can make the task faster, but only up to a point. Too many workers can also add overhead.

## Joblib Backends

Joblib supports different backends.

Common backends are:

- `loky`
- `multiprocessing`
- `threading`

The default backend is usually `loky`, which uses separate processes.

For I/O-bound tasks, we can use the `threading` backend.

```python
start_time = time.perf_counter()

result = Parallel(n_jobs=4, backend="threading")(
    delayed(root_printer)(number)
    for number in range(10, 1, -1)
)

end_time = time.perf_counter()

print(f"Time: {end_time - start_time}")
print(result)
```

In the original experiment, the threading backend was faster for this sleep-based example.

This makes sense because `time.sleep()` is waiting, not doing heavy CPU work.

## When to Use Joblib

Joblib is useful when:

- you have a function
- you need to call it many times
- each call is independent
- you want simple parallel execution

Examples:

- processing many files
- resizing many images
- running many independent experiments
- feature extraction
- model evaluation
- parameter search
- independent backtesting chunks

## Threading vs Joblib

| Feature | `threading` | `ThreadPoolExecutor` | `joblib` |
|---|---|---|---|
| Built-in | Yes | Yes | No |
| Easy for many tasks | Medium | Easy | Easy |
| Good for I/O-bound work | Yes | Yes | Yes |
| Good for CPU-bound work | Limited | Limited | Better with process backend |
| Best use case | Custom thread logic | Simple thread pools | Parallel loops |

For most simple parallel loops, I prefer `ThreadPoolExecutor` or Joblib. For lower-level control, I use `threading`.

## Best Practices for Python Multithreading

Here are some useful tips:

- Use threads mainly for I/O-bound tasks.
- Use `join()` when the main program should wait.
- Use `Lock` when multiple threads modify shared data.
- Use `Queue` to pass data between threads safely.
- Avoid writing to the same file from many threads without control.
- Do not assume threads will run in the same order every time.
- Keep thread functions small and clear.
- Handle exceptions inside thread functions.
- Do not create too many threads.
- Use `ThreadPoolExecutor` for simple repeated tasks.

## Common Mistakes

Some common beginner mistakes are:

- expecting threads to speed up CPU-heavy Python code
- forgetting to call `join()`
- sharing variables without locks
- reading a file before another thread creates it
- assuming print order is deterministic
- creating too many threads
- ignoring exceptions inside threads
- confusing multithreading with multiprocessing
- using daemon threads for important work

## Final Thoughts

In this post, we learned the basics of **multithreading in Python**. We started with the built-in `threading` module, created simple threads, looked at file reading and writing problems, and then introduced locks, events, queues, `ThreadPoolExecutor`, and Joblib.

The main lesson is that multithreading is powerful when used for the right tasks. It is especially useful for I/O-bound work, such as file handling, API calls, downloads, and background monitoring. For CPU-heavy work, multiprocessing or process-based Joblib backends are usually better.

This was a beginner-level introduction, and there is much more to learn about concurrency in Python.
