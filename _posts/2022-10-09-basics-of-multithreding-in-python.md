---
title:  Basics of Multi-threading in Python
date:   2022-10-09 01:29:17 +0545
categories:
    - Python
    - multi-threading
tags:
    - python
    - joblib
    - multiprocess
header:
  teaser: assets/python/threading.png
---
Multi-threading in Python is often used when there are tasks related to I/O bound. But before going further, let's take a few examples where multi-threading could be used:

1. Downloading images from the web and doing image processing-related tasks. It takes some time to download the image and some time to process it too but these two sub-tasks can be run either sequentially or parallelly. Running sequentially has some flaws when our system has multi-threading as our resources will not be using what is available. And another is image processing could take a long time and the download will be paused when processing is going on. But we could run these two tasks in independent threads. One thread will download images from the web and another will process that downloaded image.
2. While doing machine learning tasks, we have to work with multiple data and process a lot before sending them to modeling. We could run multiple threads that our CPU supports and then perform pre-processing on them.
3. Another application I recently did was for making a back-testing bot. Users will make different alerts with rules based on technical indicators and they will try to check if this alert will fire at this time of the day. One can run this test for a long period of time and the data is not related to each other. While testing sequentially, it was quite slow and was not running at the full capacity of the system. But when I made different threads and ran clusters of different dates, it was surprisingly fast.
4. Another I built is also for a trading bot. There will be an API listening to the user's request to buy stock options based on the strike price, symbol, and expiry date. Also, we had to use [bracket order](https://www.investopedia.com/terms/b/bracketedbuyorder.asp). I had to keep a track of these orders. So I ran distinct threads in distinct order until they were filled. This way one order runs independently of another.
 
## Using `threading`
Multi-threading in Python can be done using builtin module too. `threading` is one of the standard python libraries to do thread-based multiprocessing. Please follow the [documentation](https://docs.python.org/3/library/threading.html) for modules.
 
### Simple Example
Let's create a simple example of multi-threading in Python where we will print even numbers in one and odds in another function. Even will sleep for 2 seconds after printing one and odd will sleep for 1 second. There should not be any blocking of time from even functioning while it is sleeping.
 
 
```python
import threading
import time
 
def even(x):
    n=0
    while x>n:
        print(f"EVEN: {n}")
        n+=2
        time.sleep(2)
 
def odd(x):
    n=1
    while x>n:
        print(f"ODD: {n}")
        n+=2
        time.sleep(1)
 
th1 = threading.Thread(target=even, args=[10])
th2 = threading.Thread(target=odd, args=[10])
 
th1.start()
th2.start()
 
th1.join()
th2.join()
```
 
    5
    EVEN: 0
    ODD: 17
    
    ODD: 3
    EVEN: 2
    ODD: 5
    ODD: 7
    EVEN: 4ODD: 9
    
    EVEN: 6
    EVEN: 8
    5
    
 
`threading.Thread()` creates a thread object and we start the thread by calling the `start` method of it. Then we join to wait for the thread to complete.
As we can see in the above example that both functions are running independently of each other without altering the order of another. The above process can also be achieved with Asyncio but Asyncio doesn't use multi-threading.
 
Lets try another example where one function writes data to a file and another will read that from another thread.
 
 
```python
def writer(x):
    n=0
 
    with open(f"{x}.txt", "w") as fp:
        fp.write("")
    
    while x>n:
        with open(f"{x}.txt", "a") as fp:
            fp.write(f"{n}")
            print(f"Writer N: {n} | Wrote Line: {n}")
        n+=1
        time.sleep(1)
 
def reader(x):
    n=0
    while x>n:
        with open(f"{x}.txt", "r") as fp:
            lines = fp.read()
            print(f"Reader N: {n} | Read Lines: {lines}")
        n+=1
        time.sleep(2)
 
th1 = threading.Thread(target=writer, args=[10])
th2 = threading.Thread(target=reader, args=[10])
 
th1.start()
th2.start()
 
th1.join()
th2.join()
```
 
    Writer N: 0 | Wrote Line: 0
    Reader N: 0 | Read Lines: 0
    Writer N: 1 | Wrote Line: 1
    Reader N: 1 | Read Lines: 01
    Writer N: 2 | Wrote Line: 2
    Writer N: 3 | Wrote Line: 3
    Reader N: 2 | Read Lines: 0123
    Writer N: 4 | Wrote Line: 4
    Writer N: 5 | Wrote Line: 5
    Reader N: 3 | Read Lines: 012345
    Writer N: 6 | Wrote Line: 6
    Writer N: 7 | Wrote Line: 7
    Reader N: 4 | Read Lines: 01234567
    Writer N: 8 | Wrote Line: 8
    Writer N: 9 | Wrote Line: 9
    Reader N: 5 | Read Lines: 0123456789
    Reader N: 6 | Read Lines: 0123456789
    Reader N: 7 | Read Lines: 0123456789
    Reader N: 8 | Read Lines: 0123456789
    Reader N: 9 | Read Lines: 0123456789
    
 
In the above example, the first file is created and then data is written on it. Another process is trying to read it. We changed the order of the above thread execution.
 
 
```python
 
th1 = threading.Thread(target=writer, args=[15])
th2 = threading.Thread(target=reader, args=[15])
 
th2.start()
th1.start()
 
th1.join()
th2.join()
```
 
    Exception in thread Thread-42:
    Traceback (most recent call last):
      File "C:\ProgramData\Anaconda3\lib\threading.py", line 932, in _bootstrap_inner
        self.run()
      File "C:\ProgramData\Anaconda3\lib\threading.py", line 870, in run
        self._target(*self._args, **self._kwargs)
      File "<ipython-input-17-fd5c9a147b0e>", line 17, in reader
    FileNotFoundError: [Errno 2] No such file or directory: '15.txt'
    
 
    Writer N: 0 | Wrote Line: 0
    Writer N: 1 | Wrote Line: 1
    Writer N: 2 | Wrote Line: 2
    Writer N: 3 | Wrote Line: 3
    Writer N: 4 | Wrote Line: 4
    Writer N: 5 | Wrote Line: 5
    Writer N: 6 | Wrote Line: 6
    Writer N: 7 | Wrote Line: 7
    Writer N: 8 | Wrote Line: 8
    Writer N: 9 | Wrote Line: 9
    Writer N: 10 | Wrote Line: 10
    Writer N: 11 | Wrote Line: 11
    Writer N: 12 | Wrote Line: 12
    Writer N: 13 | Wrote Line: 13
    Writer N: 14 | Wrote Line: 14
    
 
Error happens! Why? Because the file `15.txt` was supposed to be created by the write function which is started later than the reader function. The reader function tries to read the file which is not been created yet. 
 
### Using `threading.Thread` Class
We can create a multi-threading in Python by by creating a child class of `threadin.Thread`. This way we can do thread stuff as well as write our own version of operations inside it.
 
 
```python
class ThreadClass(threading.Thread):
    """
    ## ThreadClass
    """
    def run(self,*args,**kwargs):
        while True:
            # operations goes here
            time.sleep(2)
            break
        print(f"Run ended.")
 
t = ThreadClass()
t.daemon=False         
 
t.start()
t.join()
 
```
 
    Run ended.
    
 
## Using Joblib
 Another option to do multi-threading in Python is using Joblib.
[Joblib](https://joblib.readthedocs.io/en/latest/) is a Python library that provides an easy way to perform parallelization. We need to install joblib before using it. We can do so by `pip install joblib`.
 
 
```python
import time
```
 
Let's take a simple example where we pass a number to a function and it sleeps after printing its square root.
 
 
```python
def root_printer(x):
    root=x**0.5
    print(f"Printer: {root}")
    time.sleep(root)
    return root
    
 
st = time.perf_counter()
res = list(map(root_printer,range(10,1,-1)))
et = time.perf_counter()
print(f"Time: {et-st}")
print(res)
 
```
 
    Printer: 3.1622776601683795
    Printer: 3.0
    Printer: 2.8284271247461903
    Printer: 2.6457513110645907
    Printer: 2.449489742783178
    Printer: 2.23606797749979
    Printer: 2.0
    Printer: 1.7320508075688772
    Printer: 1.4142135623730951
    Time: 21.6052624999993
    [3.1622776601683795, 3.0, 2.8284271247461903, 2.6457513110645907, 2.449489742783178, 2.23606797749979, 2.0, 1.7320508075688772, 1.4142135623730951]
    
 
If we were to do sequential programming which is the above way, it would take us a long time. That is 21.16 secs. What if we ran the above code in parallel?
 
 
```python
st = time.perf_counter()
result = Parallel(n_jobs=2)(delayed(root_printer)(i) for i in range(10,1,-1))
et = time.perf_counter()
print(f"Time: {et-st}")
print(result)
```
 
    Time: 13.476816299999882
    [3.1622776601683795, 3.0, 2.8284271247461903, 2.6457513110645907, 2.449489742783178, 2.23606797749979, 2.0, 1.7320508075688772, 1.4142135623730951]
    
 
With parallel execution using Joblib and 2 jobs, it took us only 13 seconds to complete the same task. What if we increased jobs?
 
 
```python
st = time.perf_counter()
result = Parallel(n_jobs=4)(delayed(root_printer)(i) for i in range(10,1,-1))
et = time.perf_counter()
print(f"Time: {et-st}")
print(result)
```
 
    Time: 8.8883737999995
    [3.1622776601683795, 3.0, 2.8284271247461903, 2.6457513110645907, 2.449489742783178, 2.23606797749979, 2.0, 1.7320508075688772, 1.4142135623730951]
    
 
It is faster than using only 2 jobs. In the above example, delayed is sending us delayed results.
 
Using joblib, we can run processes in different back-ends. Following is the docstring of `Parallel`.
 
```
backend: str, ParallelBackendBase instance or None, default: 'loky'
    Specify the parallelization backend implementation.
    Supported backends are:
 
    - "loky" used by default, can induce some
      communication and memory overhead when exchanging input and
      output data with the worker Python processes.
    - "multiprocessing" previous process-based backend based on
      `multiprocessing.Pool`. Less robust than `loky`.
    - "threading" is a very low-overhead backend but it suffers
      from the Python Global Interpreter Lock if the called function
      relies a lot on Python objects. "threading" is mostly useful
      when the execution bottleneck is a compiled extension that
      explicitly releases the GIL (for instance a Cython loop wrapped
      in a "with nogil" block or an expensive call to a library such
      as NumPy).
    - finally, you can register backends by calling
      register_parallel_backend. This will allow you to implement
      a backend of your liking.
 
    It is not recommended to hard-code the backend name in a call to
    Parallel in a library. Instead, it is recommended to set soft hints
    (prefer) or hard constraints (require) so as to make it possible
    for library users to change the backend from the outside using the
    parallel_backend context manager.
prefer: str in {'processes', 'threads'} or None, default: None
    Soft hint to choose the default backend if no specific backend
    was selected with the parallel_backend context manager. The
    default process-based backend is 'loky' and the default
    thread-based backend is 'threading'. Ignored if the ``backend``
    parameter is specified.
```
 
Which means that we can use threading or multiprocessing using Joblib which is even better. 
 
Let's use threading as the backend.
 
 
```python
st = time.perf_counter()
result = Parallel(n_jobs=4, backend="threading")(delayed(root_printer)(i) for i in range(10,1,-1))
et = time.perf_counter()
print(f"Time: {et-st}")
print(result)
```
 
    Printer: 3.1622776601683795Printer: 3.0
    Printer: 2.8284271247461903
    Printer: 2.6457513110645907
    
    Printer: 2.449489742783178
    Printer: 2.23606797749979
    Printer: 2.0
    Printer: 1.7320508075688772
    Printer: 1.4142135623730951
    Time: 6.438640599999417
    [3.1622776601683795, 3.0, 2.8284271247461903, 2.6457513110645907, 2.449489742783178, 2.23606797749979, 2.0, 1.7320508075688772, 1.4142135623730951]
    
 
It seems to be faster than loky.
 
 
This blog was a simple example of different options of multi-threading in Python and there is more to it! [Please stay tuned for more!](https://dataqoil.com/newsletter/)

## References
* [JobLib Documentation](https://joblib.readthedocs.io/en/latest/index.html)
* [Threading](https://docs.python.org/3/library/threading.html)

