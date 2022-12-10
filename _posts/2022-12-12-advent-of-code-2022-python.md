---
title:  "Advent of Code 2022 with Python"
date:   2022-12-12 09:29:17
categories:
    - Python
    - Challenge
tags:
    - Advent of Code
    - DSA
header:
  teaser: assets/advent_of_code/stars.png
---
The advent of Code is a yearly festival for programmers like me where we try to solve different stories to gain stars. I love these challenge because its fun and takes us slowly from beginner level to harder level. I am really weak in competitive programming and DSA stuff but still, I like to try Advent of Code. Last year I was only able to complete up to day 16 and I had to take help from some sources like Reddit too. This year I forgot the start date and it's already December 3. I will not be in the rank but this is really a fun and great challenge. Let's start from Day 1.

The input data will be in two parts, the first part will be the test given on the site and the second part is the personalized input. There will be a special line `Split from here` which separates these parts. 

After I copy the data from the challenge page to the respective text file of a challenge i.e. day1.txt for the day 1 challenge, I write a solution for the test first and if it matches the answer then I run it on my input. Mostly it works.

As usual, my solutions are in [repository](https://github.com/q-viper/Adevent-Of-Code.git) as well.

## Preparation and Pushing Solution

### Data Files
No need to run this again.




```python
for i in range(2, 26):
    with open(f"data/day{i}.txt","w") as fp:
        fp.writelines("Split From Here")
    
```

### Pushing Solution


```python
! git add .
! git commit -m"added day 2,3,4 solution"
! git push origin master
```

    warning: LF will be replaced by CRLF in 2022/2022.ipynb.
    The file will have its original line endings in your working directory
    warning: LF will be replaced by CRLF in 2022/data/day2.txt.
    The file will have its original line endings in your working directory
    warning: LF will be replaced by CRLF in 2022/data/day3.txt.
    The file will have its original line endings in your working directory
    warning: LF will be replaced by CRLF in 2022/data/day4.txt.
    The file will have its original line endings in your working directory
    warning: LF will be replaced by CRLF in 2022/data/day1.txt.
    The file will have its original line endings in your working directory
    

    [master f672c05] added day 2,3,4 solution
     5 files changed, 4231 insertions(+), 72 deletions(-)
     rename 2022/data/{1.txt => day1.txt} (100%)
     rewrite 2022/data/day2.txt (100%)
     rewrite 2022/data/day3.txt (100%)
     rewrite 2022/data/day4.txt (100%)
    

    To https://github.com/q-viper/Adevent-Of-Code.git
       3bd91b4..f672c05  master -> master
    

## Reader Function

This is the same as the previous year's function. I copy the input into a file inside the `data` directory and it will be read from here.



```python
import numpy as np
```


```python

def get_data(day=1):
    """
    Returns test and real data in list format.
    Raw data should be maintained as:
        [test data]
        Split From Here
        [actual data]
    """
    file_name = f"data/day{day}.txt"
    
    with open(file_name) as fp:
        data = fp.read().strip().split("Split From Here")
        data = [d.strip().split("\n") for d in data]
        return data
# get_data()
```

## Day 1
Problem [link](https://adventofcode.com/2022/day/1).


```python
data=get_data()[1]
data[0]
```




    '18313'



### Part 1

>Find the Elf carrying the most Calories. How many total Calories is that Elf carrying?


```python
data[0]
```




    '18313'




```python
ndata = np.array(data)
indx = np.where(ndata=='')[0]
calories = np.split(ndata, indx)
calories = np.array([np.delete(nd, np.where(nd == '')[0]).astype(int).sum() for nd in calories])
calories.max()
```




    71924




```python

```

### Part 2


```python
ncalories = calories.copy() 
ncalories.sort()
ncalories[-3:].sum()
```




    210406



## Day 2
Problem [link](https://adventofcode.com/2022/day/2).

### Part 1


```python
data = get_data(2)
data[0]
```




    ['A Y', 'B X', 'C Z']




```python
# Response X for Rock, Y for Paper, and Z for Scissors
# Elf A for Rock, B for Paper, and C for Scissors
rvalue = {'X':1, 'Y':2, 'Z':3}
rps1 = "ABC"
rps2 = 'XYZ'
win = [f"{r1} {r2}" for r1,r2 in zip(rps1, rps2[1:]+rps2[:1])]
loss = [f"{r1} {r2}" for r1,r2 in zip(rps1, rps2[-1:] +rps2[:-1])]

score = 0
for game in data[1]:
    score+=rvalue[game[-1]]
    if game in win:
        score+=6
    elif game in loss:
        score+= 0
    else:
        score+=3
    # print(score)
score   
```




    10816




```python

```

### Part 2


```python
score = 0
avalue = {i:j+1 for j,i in enumerate(rps1)}
for game in data[1]: 
    # print(game)
    if game[-1] == 'Z':
        v = [k for k in win if game[0] in k][0][-1]
        score+=6
    elif game[-1] =='X':
        v = [k for k in loss if game[0] in k][0][-1]
        score+= 0
    else:
        v = rps2[rps1.index(game[0])]
        score+=3
    score+=rvalue[v]
print(score)
```

    11657
    


```python

```

## Day 3

Problem [link](https://adventofcode.com/2022/day/2).

### Part 1




```python
data = get_data(3)
```

Let's use string and set to solve this problem. We need to find what is the common letter in both halves and Python's `set.intersection` does it.


```python
import string
items = string.ascii_lowercase+string.ascii_uppercase

score=0
for ruck in data[1]:
    hi = len(ruck)//2
    h1 = ruck[:hi]
    h2 = ruck[hi:]
    rep = list(set(h1).intersection(set(h2)))
    score+=sum([items.index(v)+1 for v in rep ])
print(score)
```

    8240
    

### Part 2

There are 3 groups and we need to find the common letter among these three. Again using NumPy to splitting array and set intersection to find the common.


```python
darr = np.array(data[1])
darr = np.split(darr, np.arange(3, len(darr), 3))

score = 0
for arr in darr:
    cv = set(arr[0]).intersection(set(arr[1])).intersection(set(arr[2]))
    score+=sum([items.index(v)+1 for v in cv ])
print(score)
    
```

    2587
    


```python

```

## Day 4

### Part 1


```python
data = get_data(4)
```


```python
data[0]
```




    ['2-4,6-8', '2-3,4-5', '5-7,7-9', '2-8,3-7', '6-6,4-6', '2-6,4-8']



Again, using the set seems to be the best idea. If the difference between two sets results in empty sets then the first set really is a subset of the second set.


```python
score = 0
for grp in data[1]:
    g1 = list(map(int, grp.split(',')[0].split("-")))
    g2 = list(map(int,grp.split(',')[1].split("-")))
    
    g1 = set(range(g1[0], g1[1]+1))
    g2 = set(range(g2[0], g2[1]+1))
    
    if g1-g2 == set() or g2-g1==set():
        score+=1
    
print(score)    
```

    602
    

### Part 2

Now let's find if the intersection is empty or not. If it not then some overlapping has occurred.


```python
score = 0
for grp in data[1]:
    g1 = list(map(int, grp.split(',')[0].split("-")))
    g2 = list(map(int,grp.split(',')[1].split("-")))
    
    g1 = set(range(g1[0], g1[1]+1))
    g2 = set(range(g2[0], g2[1]+1))
    
    if g1.intersection(g2) != set():
        score+=1
    
print(score)    
```

    891
    


```python

```
