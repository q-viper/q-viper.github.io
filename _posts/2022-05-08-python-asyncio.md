---
title:  "Python Asyncio: Concurrent Programming"
date:   2022-05-08 09:29:17 +0545
categories:
    - Python
    - Concurrent Programming
    - Asyncio
    
tags:
    - Asyncio
    - alpaca api
    - Streaming
---

## Introduction
Async IO means Asynchronous I/O and it has been there since the Python 3.4. The main purpose of asyncio is to achieve **Concurrency and Multiprocessing**. In Python, we can achive async via module [`asyncio`](https://docs.python.org/3/library/asyncio.html) additionally, we can use keywords like `async` and `await` to specify async functions and wait for its execution.

### But why asyncio?
There are numerous usecases of using asyncio and one of simplest is that it is simple and have very highl level way of achieving concurrency. One simple usecases where `asyncio` comes handy is:
* Wait for a operation to complete and only then proceed further but keep other part of the execution continued. Lets suppose we are scraping a huge site and we want to process those part which has been scraped, so what we will do is write those scraped content in some file via a `async` file and then read it somewhere else.



## Corutines and MultiTasking
A async function in Python itself is a coroutine. We can see it in action by running below code.


```python
import time,asyncio

async def lets_wait(wt):
    print(f"Time: {time.strftime('%X')}")
    print(f"Waiting for {wt}.")
    await asyncio.sleep(wt)
    print(f"Waited for {wt}.")
    print(f"Time: {time.strftime('%X')}")
    print(f"Again Waiting for {wt}")
    await asyncio.sleep(wt)
    print(f"Waited for {wt}.")
    print(f"Time: {time.strftime('%X')}")

lets_wait(1)
```




    <coroutine object lets_wait at 0x7ff2b38b17a0>



In above code, We have defined a `lets_wait` `async` function which simply prints the number we passed into and waits for that number of second before going further. But inorder to get the output, we have to run it using `asyncio.run(coroutine,*)`. In our case, `asyncio.run(lets_wait(1))`.

```bash
Time: 19:16:15
Waiting for 1.
Waited for 1.
Time: 19:16:16
Again Waiting for 1
Waited for 1.
Time: 19:16:17
```

### Multi Tasking
In `asyncio`, we can create multiple tasks and run them concurrently. A task is then awaited. Lets see an example:


```python
import time,asyncio

async def lets_wait(wt):
    print(f"Time: {time.strftime('%X')}")
    print(f"Waiting for {wt}.")
    await asyncio.sleep(wt)
    print(f"Waited for {wt}.")
    print(f"Time: {time.strftime('%X')}")
    print(f"Again Waiting for {wt}")
    await asyncio.sleep(wt)
    print(f"Waited for {wt}.")
    print(f"Time: {time.strftime('%X')}")

async def main():
    task1 = asyncio.create_task(lets_wait(5))
    task2 = asyncio.create_task(lets_wait(10))
    task3 = asyncio.create_task(lets_wait(4))
    
    print(f"Await start Time: {time.strftime('%X')}")
    await task1
    await task2
    await task3
    print(f"Time: {time.strftime('%X')}")
    
asyncio.run(main())
```

```bash
Await start Time: 19:20:06
Time: 19:20:06
Waiting for 5.
Time: 19:20:06
Waiting for 10.
Time: 19:20:06
Waiting for 4.
Waited for 4.
Time: 19:20:10
Again Waiting for 4
Waited for 5.
Time: 19:20:11
Again Waiting for 5
Waited for 4.
Time: 19:20:14
Waited for 10.
Time: 19:20:16
Again Waiting for 10
Waited for 5.
Time: 19:20:16
Waited for 10.
Time: 19:20:26
Time: 19:20:26
```

In above code, we have 3 tasks each with 5, 10 and 4 seconds as wait time. First the task with second 5 is run and then 10 and 4 then they wait for respective time. But the third task has only 4 seconds to wait and it shows print statement befor task with 5 seconds. Then it again waits for 4 seconds and only then task with 5 seconds prints its wait statement.

### Concurrent Tasks
We can run task concurrently and wait for them and it is much neat than the task above. We can do so using `asyncio.gather(*fxns, return_exceptions=False)`. If any function is awaitable and a coroutine then they are scheduled as a task. Any function is awaitable if it can be used as `await function`.


```python
import time,asyncio

async def lets_wait(wt):
    t1=time.time()
    print(f"Time: {time.strftime('%X')}")
    print(f"Waiting for {wt}.")
    await asyncio.sleep(wt)
    print(f"Waited for {wt}.")
    print(f"Time: {time.strftime('%X')}")
    print(f"Again Waiting for {wt}")
    await asyncio.sleep(wt)
    print(f"Waited for {wt}.")
    print(f"Time: {time.strftime('%X')}")
    wtt=time.time()-t1
    print(f"Completed task-{wt} in {wtt}.")
    return {wt:wtt}

async def main():
    task1 = asyncio.create_task(lets_wait(5))
    task2 = asyncio.create_task(lets_wait(10))
    task3 = asyncio.create_task(lets_wait(4))
    
    print(f"Await start Time: {time.strftime('%X')}")
    await task1
    await task2
    await task3
    print(f"Time: {time.strftime('%X')}")
    
async def main2():
    ret = await asyncio.gather(lets_wait(2),lets_wait(1),lets_wait(4),lets_wait(3))
    print(f"Returned from main2: {ret}")
asyncio.run(main2())
```

We have created another function `main2` which will pass 4 different tasks to run concurrently. Lets see the output below.

```bash
Time: 19:35:29
Waiting for 2.
Time: 19:35:29
Waiting for 1.
Time: 19:35:29
Waiting for 4.
Time: 19:35:29
Waiting for 3.
Waited for 1.
Time: 19:35:30
Again Waiting for 1
Waited for 2.
Time: 19:35:31
Again Waiting for 2
Waited for 1.
Time: 19:35:31
Completed task-1 in 2.0059587955474854.
Waited for 3.
Time: 19:35:32
Again Waiting for 3
Waited for 4.
Time: 19:35:33
Again Waiting for 4
Waited for 2.
Time: 19:35:33
Completed task-2 in 4.033399820327759.
Waited for 3.
Time: 19:35:35
Completed task-3 in 6.003902196884155.
Waited for 4.
Time: 19:35:37
Completed task-4 in 8.007280111312866.
Returned from main2: [{2: 4.033399820327759}, {1: 2.0059587955474854}, {4: 8.007280111312866}, {3: 6.003902196884155}]
```

Since there are 2 waits inside our `lets_wait` function, we can see that each task has completed in time more than twice of its `wt` value. The seconds after decimal is because of the print statement and calculations which can be ignored here. At last, the `asyncio.gather` returned the return of each task in a list format. Which is pretty useful.

### Timeouts
Timeouts are useful when we wait for more than usual and we want to forcefully stop the task and show an error. Lets do that.


```python
async def main3():
    try:
        await asyncio.wait_for(lets_wait(2),timeout=4)
    except asyncio.TimeoutError:
        print("Timeout")    
        
asyncio.run(main3())
```

In above code, we are waiting for 4 seconds and the function `lets_wait` is given 2 which means it will take more than 2 seconds to complete.

```bash
Time: 19:46:28
Waiting for 2.
Waited for 2.
Time: 19:46:30
Again Waiting for 2
Timeout
```

What happens is, it waits for 4 seconds and if the async is still running, time out error is shown. And our sleep is done twice for 2 minutes each and there are also print and subtraction which will take some time too. And thus the task did not complete in 4 seconds and we got an error.

There are some other useful usecases and functions provided by asyncio and its worth reading them [here](https://docs.python.org/3/library/asyncio-task.html) but now I will move into streaming..


```python

```

## Streaming Data With Asyncio
Data streaming is today's most wanted feature in any data apps as we want to see things happening in life time. One simple usecase can be seen in the Stock market and which has led me here. In stock market, transactions happen in any moment and we will have our own platform to show the data in realtime. In that case, we will wait for eternity and call the api or data source and update our data if changes has been made in source. Asyncio has a good documentation about [streaming](https://docs.python.org/3/library/asyncio-stream.html). But I am not going to use Asyncio like that. We will use `asyncio` to stream data from [Alpaca Trade API](https://github.com/alpacahq/alpaca-trade-api-python) and [you can read our last week's blog for that](https://q-viper.github.io/2022/05/01/python-for-stock-market-analysis-alpaca-api/).

