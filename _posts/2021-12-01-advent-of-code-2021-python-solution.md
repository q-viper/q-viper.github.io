---
title:  "Advent of Code 2021: Python Solution"
date:   2021-12-01 10:29:17 +0545
last_modified_at: 2021-12-05 12:29:17 +0545
categories:
    - adventofcode
    - challenges
tags:
    - algorithms
    - python
    - adventofcode 2021
    - programming challenges
header:
  teaser: assets/advent_of_code/stars.png
toc: true
---
<!-- {toc:} -->

# Advent of Code 2021: Python Solutions
I am not good at solving problems fast but I try to do them with best I could. And here in this blog post, I try to show my solutions of all days in one post. For the Jupyter Notebook, please refer to this [repository](https://github.com/q-viper/Adevent-Of-Code.git). The code will be updated once solved but I am preparing headers, code blocks and links before day starts. Also I am not trying to get into any rank.

## Before Solving Day 1
Last year I had hard time reading input data and thus, this year I am using some helper functions. First of them is `get_data()`.

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
get_data()
```

I created a folder `data` in working directory and inside there will be a text files with names as `day[current_day].txt`. The file will contain the test input in first half and real input in second half. The separator will be a string `"Split From Here"`. 

Additionally, to make things faster, I created text files for all days with below code:

```python
for i in range(2, 26):
    with open(f"data/day{i}.txt","w") as fp:
        fp.writelines("Split From Here")
```

## Day 1
[Here is the problem link.](https://adventofcode.com/2021/day/1)

### Get Data

```python
data,data1 = get_data() 
data = list(map(int, data))
data1 = list(map(int, data1))

data
```

### Part 1
```python
pd = None
res = []
for d in data:
    if pd is None:
        res.append(None)
    else:
        if pd>d:
            res.append("0")
        else:
            res.append("1")
    pd=d
res.count("1")
```
The result will be printed out and it is 7 for test data. Changing variable data1 from data in above code will give the answer.

### Part 2
```python
w = []
wsum = []
i = 0
ps = None
for j in range(3, len(data1)+1):
    wsum.append(sum(data1[i:j]))
    i+=1

pd = None
res1 = []
for d in wsum:
    if pd is None:
        res1.append(None)
    else:
        if pd>=d:
            res1.append("0")
        else:
            res1.append("1")
    pd=d
res1.count("1")
```

## Day 2
[Here is the problem link.](https://adventofcode.com/2021/day/2)

### Part 1
```python
data,data1 = get_data(day=2)

hs = 0
vs = 0

for d in data:
    k,v = d.split()
    
    if k =="forward":
        hs+=int(v)
    elif k == "down":
        vs+=int(v)
    elif k=="up":
        vs-=int(v)
    
print(hs*vs)
```

The test output will be 150 and for real output, should change the variable to data1.


### Part 2 
```python
hs = 0
vs = 0
aim = 0

for d in data1:
    k,v = d.split()
    v = int(v)
    
    if k =="forward":
        hs+=v
        vs+=aim*v
    elif k == "down":
        aim+=v
#         vs+=v
    elif k=="up":
        aim-=v
#         vs-=v
    
print(hs*vs)
```

## Day 3
[Here is the problem link.](https://adventofcode.com/2021/day/3)

### Part 1
```python
from collections import Counter
import numpy as np

data,data1 = get_data(day=3)

def part1(inp):
    cs = len(inp[0])
    dt = [int(d) for dt in inp for d in dt]
    dt = np.array(dt).reshape(-1, cs)

    print(dt[0])

    
    counts = [sorted(dict(Counter(dt[:, i])).items(), key=lambda item: item[1]) for i in range(len(dt[0]))]
    counts = np.array(counts).reshape(-1,2)


    minidx = np.arange(0, len(counts), 2)
    maxidx = np.arange(1, len(counts), 2)

    minv = int("".join(list(map(str, counts[minidx, 0]))), 2)
    maxv = int("".join(list(map(str, counts[maxidx, 0]))), 2)
    print(minv, maxv)
    print(minv*maxv)
part1(data)
```

The result of test input is:

```
[0 0 1 0 0]
9 22
198
```
All hail to the NumPy.
For real input, should change the variable name data to data1 and call `part1`.

### Part 2
```python
def o2(dt):
    ndt = dt.copy()
    #print(len(ndt[0]))
    
    curr_c = 0
    while curr_c<len(dt[0]):
        print(f"Current Col: {curr_c}")
        counts = [sorted(dict(Counter(ndt[:, curr_c])).items(), key=lambda item: item[1])]
        counts = np.array(counts).reshape(-1,2)
        if len(counts)>1:
            if counts[0, 1]==counts[1, 1]:
                ndt = ndt[ndt[:,curr_c]==1]
            else:
                ndt = ndt[ndt[:,curr_c]==counts[1][0]]
        else:
            ndt = ndt[ndt[:,curr_c]==counts[1][0]]
        print(f"Current Col: {curr_c} Rows: {len(ndt)}")
#         print(counts)
#         print(ndt[ndt[:,curr_c]==counts[1][0]])
            
        
        
        curr_c+=1
    res = int("".join(list(map(str, ndt[0]))), 2)
    print(res)
    return res

def co2(dt):
    ndt = dt.copy()
    #print(len(ndt[0]))
    
    curr_c = 0
    while curr_c<len(dt[0]):
        
        counts = [sorted(dict(Counter(ndt[:, curr_c])).items(), key=lambda item: item[1])]
        counts = np.array(counts).reshape(-1,2)
        if len(counts)>1:
            if counts[0, 1]== counts[1, 1]:
                ndt = ndt[ndt[:,curr_c]==0]
            else:
                ndt = ndt[ndt[:,curr_c]==counts[0][0]]
        else:
            ndt = ndt[ndt[:,curr_c]==counts[0][0]]
            
#         print(ndt)
        print(f"Current Col: {curr_c} Rows: {len(ndt)}")
        curr_c+=1
    return int("".join(list(map(str, ndt[0]))), 2)

def part2(inp):
    cs = len(inp[0])
    dt = [int(d) for dt in inp for d in dt]
    dt = np.array(dt).reshape(-1, cs)
    
    print("O2")
    o2v = o2(dt)
    co2v = co2(dt)
    print(o2v, co2v)
    print(o2v*co2v)
part2(data)
    
```

Code seems little bit messy and we could refactor it but I am too busy to do so :(. The test output will be:

```
O2
Current Col: 0
Current Col: 0 Rows: 7
Current Col: 1
Current Col: 1 Rows: 4
Current Col: 2
Current Col: 2 Rows: 3
Current Col: 3
Current Col: 3 Rows: 2
Current Col: 4
Current Col: 4 Rows: 1
23
Current Col: 0 Rows: 5
Current Col: 1 Rows: 2
Current Col: 2 Rows: 1
Current Col: 3 Rows: 1
Current Col: 4 Rows: 1
23 10
230 
```
## Day 4
[Here is the problem link.](https://adventofcode.com/2021/day/4)

### Part 1
```python
import numpy as np
data,data1 = get_data(day=4)

def get_blocks(dt):
    block = []
    num = [int(i) for i in dt[0].split(",")]
    row = []
    tdata=[]
    blocks = 0
    for d in dt[2:]:
        if d == "":
            tdata.append(block)
            block=[]
            blocks+=1

        else:
            block.append([int(i) for i in d.strip().split(" ") if i!=""])
    tdata.append(block)
    block=[]
    blocks+=1
    tdata = np.array(tdata).reshape(blocks,-1, 5)
    return tdata, num

def get_first_matched(tdata, num):
    results = np.zeros_like(tdata).astype(np.bool)
    matched = False
    
    for n in num:
        for i,block in enumerate(tdata):
            results[i] += block==n
            # search across row
            if (results[i]==[ True,  True,  True,  True,  True]).all(axis=1).any():
                print(f"Row Matched Block:{i}")
                matched=True
                break

            # search across cols
            if (results[i].T==[ True,  True,  True,  True,  True]).all(axis=1).any():
                print(f"Col Matched Block: {i}")
                matched=True
                break
        if matched:
            print(f"\nResult Block: {tdata[i]}")
            s = (tdata[i]*~results[i]).sum()
            print(f"Sum: {s}")
            print(f"Last number: {n}")
            print(f"Answer: {n*s}\n")
            break



d1,n1 = get_blocks(data1)
get_first_matched(tdata=d1, num=n1)

# d1, n1
```


### Part 2
```python
def get_last_matched(tdata, num):
    results = np.zeros_like(tdata).astype(np.bool)
    matched = False
    mblocks=[]
    all_blocks = list(range(0, len(results)))
    
    for n in num:
        for i,block in enumerate(tdata):
            results[i] += block==n
            # search across row
            if (results[i]==[ True,  True,  True,  True,  True]).all(axis=1).any():
                print(f"Row Matched Block:{i}")
                if i not in mblocks:
                    mblocks.append(i)
                if len(mblocks) == len(all_blocks):
                    matched=True

            # search across cols
            if (results[i].T==[ True,  True,  True,  True,  True]).all(axis=1).any():
                print(f"Col Matched Block: {i}")
                if i not in mblocks:
                    mblocks.append(i)
                if len(mblocks) == len(all_blocks):
                    matched=True

        if matched:
            i = mblocks[i]

            print(f"\nResult Block: {tdata[i]}")
            s = (tdata[i]*~results[i]).sum()
            print(f"Sum: {s}")
            print(f"Last number: {n}")
            print(f"Answer: {n*s}")
            break
get_last_matched(tdata=d1, num=n1)
```
Again, NumPy came to the aid.

## Day 5
    [Here is the problem link.](https://adventofcode.com/2021/day/5)
    
### Part 1
```python
import numpy as np    
data,data1 = get_data(day=5)

# get(x1, y1, x2, y2)
coordinates = []
for d in data:
    x1, y1, x2, y2 = list(map(int, d.replace(" -> ", ",").split(",")))
    coordinates.append((x1, y1, x2, y2))
    
coordinates = np.array(coordinates)   
mxx,mxy = coordinates[[0, 2]].max(), coordinates[[1, 3]].max()

board = np.zeros((mxx*2, mxy*2))

# check only horizontal or vertical line
m1 = coordinates[:, 0]==coordinates[:, 2]
m2 = coordinates[:, 1]==coordinates[:, 3]
m = m1 | m2

masked = coordinates[m]  
for co in masked:    
    for x in range(min(co[0], co[2]), max(co[0], co[2])+1):
        for y in range(min(co[1], co[3]), max(co[1], co[3])+1):
            board[x, y] += 1
print((board.flatten()>1).sum())    
```

The output will be 5 for above code where test data was used. Should use `data1` for real output.
    
### Part 2
```python

# diagonal line
m1 = coordinates[:, 0]!=coordinates[:, 2]
m2 = coordinates[:, 1]!=coordinates[:, 3]
m=m1*m2
masked = coordinates[m]

for co in masked:
    # add or sub to x1?
    dx = int(co[2]>co[0]) or -1
    dy = int(co[3]>co[1]) or -1
    
    for dp in range(abs(co[2]-co[0])+1):
        x = co[0]+dx*dp
        y = co[1]+dy*dp
        board[x,y]+=1
        
print((board.flatten()>1).sum())    
```

The output of above code will be 12 for test data.
    

## Day 6
[Here is the problem link.](https://adventofcode.com/2021/day/6)

Today's challenge was easy but tricky one. I found 4th day's challenge to be more tough than today's but still I had hard time finding the right way to loop through all days.

### Part 1
```python
data,data1 = get_data(day=6)
data = [int(d) for d in data[0].split(",")]
data1 = [int(d) for d in data1[0].split(",")]

days = {}
total_days = 81
curr_data = data1.copy()
for day in range(1, total_days):    
    temp_data = []
    new_fish = []
    for d in curr_data:
        if d == 0:
            new_fish.append(8)
            d=6
        else:
            d-=1
        temp_data.append(d)
    temp_data.extend(new_fish)
    curr_data = temp_data
print(f"Total Fish: {len(curr_data)}\n")
```
Answer was:
```
Total Fish: 388419
```

It took around 2 seconds to run above code while `total_days` was 81 but when `total_day` was 257, it seemed like loop will be going on forever. And total fish keeps increasing by time. I even tried to make things faster by using NumPy but it did not work out. Thus, I used dictionaries to store lifespans of fish. Since it could be one of the 0 to 8, why bother keeping lifes?

### Part 2

```python
from collections import Counter

lifes = dict(Counter(data1))

days = 256
for day in range(1, days+1):
    lifes = {l: (0 if lifes.get(l+1) is None else lifes.get(l+1)) for l in range(-1, 8)}
    # make all 8s -1 because we create new fish with 8 after it reaches 0
    lifes[8] = lifes[-1]
    # add new lifes to that are exhausted
    lifes[6] += lifes[-1]
    # reset exhausted lifes
    lifes[-1] = 0 
    
print(sum(lifes.values()))
```




## Day 7
[Here is the problem link.](https://adventofcode.com/2021/day/7)

I found today's solution to be easier than previous day's.

### Part 1

```python
data,data1 = get_data(day=7)

data = [int(d) for d in data[0].split(",")]
data1 = [int(d) for d in data1[0].split(",")]

l = len(data1)
f = []

for v in range(l):
    f.append((sum([abs(d-v) for d in data1])))
print(min(f))        
```

#### One Liner Solution

```python
min([sum([abs(d-v) for d in data1]) for v in range(len(data1))])
```

### Part 2

```python
l = len(data1)
f = []

for v in range(l):
    diff = [abs(d-v) for d in data1]
    diffs = sum([sum(list(range(dif+1))) for dif in diff])
    f.append(diffs)
print(min(f))        
```

#### One Liner Solution

```python
min([sum([sum(list(range(abs(d-v)+1))) for d in data1]) for v in range(len(data1))])
```


## Day 8
[Here is the problem link.](https://adventofcode.com/2021/day/8)

First part was very easy but for the second part, I took little help from [here](https://github.com/MasterMedo/aoc/blob/master/2021/day/8.py).

### Solution
```python
def permutation(li: list):
    all_ps = set()
    psl = np.prod(np.linspace(1,len(li), len(li)).astype(int))
    
    
    while len(all_ps)!=psl:
        curr_ps = np.random.choice(range(len(li)), len(li), replace=False)
        curr_ps = "".join([li[i] for i in curr_ps])
        all_ps.add(curr_ps)
    return all_ps

all_ps = permutation("abcdefg")

d = {
    "abcefg": 0,
    "cf": 1,
    "acdeg": 2,
    "acdfg": 3,
    "bcdf": 4,
    "abdfg": 5,
    "abdefg": 6,
    "acf": 7,
    "abcdefg": 8,
    "abcdfg": 9,
}

cnts = {2:1, 4:4, 3:7, 7:8}

sol1 = 0
sol2 = 0

for row in data1:
    signals, output = row.split("|")
    signals = [s.strip() for s in signals.strip().split(" ")]
    output = [s.strip() for s in output.strip().split(" ")]    
    
    for o in output:
        l = len(o)
        if ls.get(l):
            
            sol1+=1
        
    for pr in all_ps:
        to = str.maketrans("abcdefg", pr)
        ts = ["".join(sorted(sig.translate(to))) for sig in signals]
        top = ["".join(sorted(op.translate(to))) for op in output]

        if all(code in d for code in ts):
            sol2 += int("".join(str(d[code]) for code in top))            
            break
sol1, sol2  
```

Later I knew there is actually a python generator `permutation` inside `itertools`.


## Day 9
[Here is the problem link.](https://adventofcode.com/2021/day/9)

First part was not much harder to crack but it still took plenty of time. But second part was tricky.

### Part 1
```python
import numpy as np
data,data1 = get_data(day=9)

dl = len(data1[0])
dt = np.array([int(d) for dt in data1 for d in dt])
dt = dt.reshape(-1, dl)

nums = []
pos = []
dc = len(dt[0])
dr = len(dt)
for r in range(len(dt)):
    for c in range(len(dt[0])):
        if r==0:
            if c==0:
                if dt[r,c]<dt[r+1, c] and dt[r,c]<dt[r, c+1]:
                    nums.append(dt[r,c])
                    pos.append((r,c))
            elif c==dc-1:
                if dt[r,c]<dt[r+1, c] and dt[r,c]<dt[r, c-1]:
                    nums.append(dt[r,c])
                    pos.append((r,c))
            else:
                if dt[r,c]<dt[r+1, c] and dt[r,c]<dt[r, c+1] and dt[r,c]<dt[r, c-1]:
                    nums.append(dt[r,c])
                    pos.append((r,c))
        elif r==dr-1:
            if c==0:
                if dt[r,c]<dt[r-1, c] and dt[r,c]<dt[r, c+1]:
                    nums.append(dt[r,c])
                    pos.append((r,c))
            elif c==dc-1:
                if dt[r,c]<dt[r-1, c] and dt[r,c]<dt[r, c-1]:
                    nums.append(dt[r,c])
                    pos.append((r,c))
            else:
                if dt[r,c]<dt[r-1, c] and dt[r,c]<dt[r, c+1] and dt[r,c]<dt[r, c-1]:
                    nums.append(dt[r,c])
                    pos.append((r,c))
        else:
            if c==0:
                if dt[r,c]<dt[r-1, c] and dt[r,c]<dt[r, c+1] and dt[r,c]<dt[r+1, c]:
                    nums.append(dt[r,c])
                    pos.append((r,c))
            elif c==dc-1:
                if dt[r,c]<dt[r-1, c] and dt[r,c]<dt[r, c-1] and dt[r,c]<dt[r+1, c]:
                    nums.append(dt[r,c])
                    pos.append((r,c))
            else:
                if dt[r,c]<dt[r-1, c] and dt[r,c]<dt[r, c+1] and dt[r,c]<dt[r, c-1] and dt[r,c]<dt[r+1, c]:
                    nums.append(dt[r,c])
                    pos.append((r,c))
                
                
nums
```

### Part 2
I thought I had to use some sort of Searching algorithm like DFS or BFS but I found a solution on StackOverflow using NumPy.

```python
from scipy import ndimage

label, num_label = ndimage.label(dt < 9)
size = np.bincount(label.ravel())

top3 = sorted(size[1:], reverse=True)[:3]
print(np.prod(top3))
``` 


## Day 10
[Here is the problem link.](https://adventofcode.com/2021/day/10)

I forgot the time of the challenge but still managed to make it done. Stack came to the aid.

### Part 1

```python
data,data1=get_data(day=10)


table = {
    ")": 3,
    "]": 57,
    "}": 1197,
    ">": 25137}

pair = {"(":")","{":"}", "[":"]", "<":">"}


corruptions = []
rem = []
for i,r in enumerate(data):
    stack = []
    is_corr=False
    for c in r:
        if c in pair:
            stack.append(pair[c])
        elif stack.pop() != c:
            print(f"Corrupted {c} at row {i}")
            corruptions.append(c)
            is_corr=True
            break
    if is_corr==False and len(stack)>0:
        rem.append(stack)
            
            
corr = dict(Counter(corruptions))
sum([table[k]*v for k,v in corr.items()])
```

### Part 2

```python
mult = {")": 1,
"]": 2,
"}": 3,
">": 4}
all_total=[]
for row in rem:
    s = 0
    for i,c in enumerate(row):
        s+=5**i*mult[c]
    all_total.append(s)
at = sorted(all_total)
at[len(at)//2]
```


## Day 11
[Here is the problem link.](https://adventofcode.com/2021/day/11)

Today's challenge was harder than previous day's.

### Solution

```python
adj = [(i,j) for i in range(-1, 2) for j in range(-1,2) if i!=0 or j!=0]
window = {(i,j):darr[i][j] for i in range(10) for j in range(10)}

flashes = 0
i=0
previous = set()

while len(previous)<len(window):
    previous = set()
    window = {k:v+1 for k, v in window.items()}
    while True:
        if sum(v>9 for k,v in window.items() if k not in previous)==0:
            break
            
        for k,v in window.items():
            if k not in previous and v>9:
                previous.add(k)
                for ad in [(k[0]+i,k[1]+j) for i,j in adj if (k[0]+i,k[1]+j) in window]:
                    window[ad]+=1
    
    flashes+=len(previous)
    window.update({k:0 for k in previous})
    i+=1
    if i==100:
        print(f"Part 1: {flashes}")
    
print(f"Part 2: {i}")
```


## Day 12
[Here is the problem link.](https://adventofcode.com/2021/day/12)

Due to work I was unable to solve it on time and thus I had to take help from [here](https://www.reddit.com/r/adventofcode/comments/rehj2r/comment/ho8zlrj/?utm_source=share&utm_medium=web2x&context=3).

### Solution

```python
data,data1 = get_data(day=12)

class Solver:
    def __init__(self, data):
        self.paths = {}
        self.data = data
        self.visited = set()
        
        self.prepare_paths()
        print(self.solve(part="1"))
        print(self.solve(part="2"))
        
    def prepare_paths(self):
        for d in self.data:
            l,r = d.split("-")
            if self.paths.get(l):
                self.paths[l].append(r)
            else:
                self.paths[l] = [r]
            if self.paths.get(r):
                self.paths[r].append(l)
            else:
                self.paths[r]=[l]
    
    def solve(self, curr_cave="start", part="1"):
        if (curr_cave=="end"):
            return 1
        if curr_cave.islower():
            self.visited.add(curr_cave)
        
        ways_count = sum([self.solve(cave, part) for cave in self.paths[curr_cave] if cave not in self.visited])
        ways_count += 0 if part!="2" else sum([self.solve(cave, cave) for cave in self.paths[curr_cave] if cave in self.visited and cave != "start"])
        
        if (curr_cave != part): self.visited.discard(curr_cave)
        return ways_count
        
s = Solver(data1)
```

Search either BFS or DFS comes into the aid.


## Day 13
[Here is the problem link.](https://adventofcode.com/2021/day/13)

Today's challenge was fun to do and it was not that hard as well.

### Solution

```python
import numpy as np
data1, data = get_data(day=13)

dots = [list(map(int, f.split(","))) for f in data[:data.index("")]]
folds = data[data.index("")+1:]
folds = [f.split("along ")[1].split("=") for f in folds]
folds = [(f[0], int(f[1])) for f in folds]

dots = np.array(dots)
window = np.zeros(dots.max(axis=0)+3)

for c in dots:
    window[c[0], c[1]] = 1
window = window.T


tw = window.copy()
# print(tw)
for f in folds:
    print(f)
    axis,value=f
    cr,cc = tw.shape
    print(tw)
    if axis=="y":
        #fold y axis
        chunk = tw[value+1:-2]
        # print(value-crs,chunk.shape, chunk)
        crs,ccs = chunk.shape
        tw[np.abs(value-crs):value] += chunk[::-1]
        tw = tw[:value]
        tw = np.append(tw, np.zeros((2, tw.shape[1])), axis=0)
        #break
        
    else:
        # fold x axis
        chunk = tw[:, value+1:-2]
        crs,ccs = chunk.shape
        print(value-crs,chunk.shape, chunk)
        tw[:, abs(value-ccs):value] += chunk[:,::-1]
        tw = tw[:, :value]
        tw = np.append(tw, np.zeros((tw.shape[0], 2)), axis=1)
    print(f"Dots: {np.sum(tw>0)}")

print(np.array2string(tw>0, separator='',
    formatter = {'bool':lambda x: ' █'[x]}))
```

My answer was:

```python
[[ ██  █  █ ███  ███  ███   ██  █  █ ████   ]
 [█  █ █  █ █  █ █  █ █  █ █  █ █  █    █   ]
 [█  █ ████ █  █ █  █ █  █ █  █ █  █   █    ]
 [████ █  █ ███  ███  ███  ████ █  █  █     ]
 [█  █ █  █ █    █ █  █    █  █ █  █ █      ]
 [█  █ █  █ █    █  █ █    █  █  ██  ████   ]
 [                                          ]
 [                                          ]]
```

## Day 14
[Here is the problem link.](https://adventofcode.com/2021/day/14)

They know we could fall into a trap. And again I fell. I went full looping mode and got the result of part 1 but the part 2 could take days.

### Part 1
```python
from collections import Counter
data,data1 = get_data(day=14)

wdata=data1.copy()
polymer = wdata[0]
rule = {v[0]:v[1] for v in [d.split(" -> ") for d in wdata[2:]]}

curr_polymer = polymer
i=0
while i< 10:
    tpoly = curr_polymer
#     print(i, tpoly)
    ind = 0
    added = 0
    for k, c in enumerate(tpoly):
        k+=1
        ch = curr_polymer[k-1:k+1]
        
        mc = rule.get(ch)
        if mc:
            tpoly = [c for c in tpoly]
            tpoly.insert(k+added, mc)
            tpoly = "".join(tpoly)
            added+=1
    curr_polymer = tpoly
    i+=1

res = dict(Counter(curr_polymer))
res = sorted(res.items(), key=lambda x: x[1], reverse=True)
res[0][1]-res[-1][1]
```

### Part 2
Taken hint from [here](https://www.reddit.com/r/adventofcode/comments/rfzq6f/comment/hoib78w/?utm_source=share&utm_medium=web2x&context=3).

```python
tmp_poly = Counter(a+b for a,b in zip(polymer, polymer[1:]))
print(tmp_poly)
chars = Counter(polymer)

for _ in range(40):
    tmp = Counter()
    for (c1,c2),value in tmp_poly.items():
        mc = rule[c1+c2]
        tmp[c1+mc] += value
        tmp[mc+c2] += value
        chars[mc] += value
    tmp_poly=tmp
max(chars.values()) - min(chars.values())
```

## Day 15
[Here is the problem link.](https://adventofcode.com/2021/day/15)

Once I failed DSA in my bachelor's degree and I never really understood Graphs and Path Finding but each year Advent of Code makes me try it once. Instead I used something easier than Dijkastra from scratch. Skimage have a way to find Minimum Cost Path

### Solution
```python
import numpy as np
from skimage import graph

data,data1 = get_data(15)

data = np.array([int(i) for dt in data for i in dt ]).reshape(-1, len(data[0]))
data
data1 = np.array([int(i) for dt in data1 for i in dt ]).reshape(-1, len(data1[0]))

window = data1.copy()

rs,cs = window.shape

cost = graph.MCP(window, fully_connected=False)
cost.find_costs(starts = [(0,0)])

journey = [window[pos] for pos in  cost.traceback((rs-1,cs-1))[1:]]
print(f"Part1: {sum(journey)}")

# 5times bigger
new_window = window.copy()
nrow = np.hstack([new_window, new_window+1, new_window+2, new_window+3, new_window+4])
new_window = np.vstack([nrow,nrow+1,nrow+2,nrow+3,nrow+4])
rs,cs = new_window.shape

new_window%=9
new_window[new_window==0]=9

cost = graph.MCP(new_window, fully_connected=False)
cost.find_costs(starts = [(0,0)])

journey = [new_window[pos] for pos in  cost.traceback((rs-1,cs-1))[1:]]
print(f"Part2: {sum(journey)}")
```


## Day 16
[Here is the problem link.](https://adventofcode.com/2021/day/16)

I was too busy to solve this challenge (but I tried for around 30min) and I did not even want to skip a day so, I had to look over other people's code. 
The following code is taken from [here](https://github.com/SiddhantAttavar/Competitive-Programming/tree/main/Other%20Contests/Advent%20Of%20Code%202021/Day16). All credit goes to the author of this repository. 

### Part 1

```python
data,data1=get_data(day=16)

data = '''38006F45291200'''.splitlines()
data=data1[0].splitlines()

s = bin(int(data[0], 16))[2:]
n = len(s)
if n % 4 != 0:
    s = '0' * (4 - n % 4) + s
n = len(s)
res = 0
c = 0

while c < n and '1' in s[c:]:
    v = int(s[c: c + 3], 2)
    res += v
    c += 3
    t = int(s[c: c + 3], 2)
    c += 3

    if t == 4:
        num = ''
        while s[c] == '1':
            num += s[c + 1: c + 5]
            c += 5
        num += s[c + 1: c + 5]
        c += 5
        num = int(num, 2)
    else:
        l = int(s[c], 2)
        c += 1
        if l == 0:
            num = int(s[c: c + 15], 2)
            c += 15
        else:
            num = int(s[c: c + 11], 2)
            c += 11

print(res)
```

### Part 2

```python
from functools import reduce

funcDict = {
    0: sum,
    1: lambda a: reduce(lambda x, y: x * y, a),
    2: min,
    3: max,
    5: lambda a: int(a[0] > a[1]),
    6: lambda a: int(a[0] < a[1]),
    7: lambda a: int(a[0] == a[1])
}

def evaluate(u):
    if packets[u][1] == 4:
        return packets[u][2]

    res = []
    for v in graph[u]:
        res.append(evaluate(v))
    return funcDict[packets[u][1]](res)

s = bin(int(data[0], 16))[2:]
for i in data[0]:
    if i != '0':
        break
    s = '0' * 4 + s
n = len(s)
if n % 4 != 0:
    s = '0' * (4 - n % 4) + s
n = len(s)
c = 0
packets = []

while c < n and '1' in s[c:]:
    v = int(s[c: c + 3], 2)
    c += 3
    t = int(s[c: c + 3], 2)
    c += 3
    
    if t == 4:
        num = ''
        while s[c] == '1':
            num += s[c + 1: c + 5]
            c += 5
        num += s[c + 1: c + 5]
        c += 5
        num = int(num, 2)

        packets.append([v, t, num, c])
    else:
        l = int(s[c], 2)
        c += 1
        if l == 0:
            num = int(s[c: c + 15], 2)
            c += 15
        else:
            num = int(s[c: c + 11], 2)
            c += 11
    
        packets.append([v, t, l, num, c])

stack = []
graph = [[] for _ in range(len(packets))]

for i, u in enumerate(packets):
    if len(stack) > 0:
        p = stack[-1]
        graph[p].append(i)
        packets[p][3] -= 1
        if packets[p][3] == 0:
            stack.pop()

    while len(stack) > 0:
        p = stack[-1]
        if packets[p][2] == 0 and packets[p][3] <= u[-1] - packets[p][-1]:
            stack.pop()
        else:
            break

    if u[1] != 4:
        stack.append(i)

print(evaluate(0))
```

## Day 17
[Here is the problem link.](https://adventofcode.com/2021/day/17)

### Part 1
```python

```

### Part 2
```python

```


## Day 18
[Here is the problem link.](https://adventofcode.com/2021/day/18)

### Part 1
```python

```

### Part 2
```python

```


## Day 19
[Here is the problem link.](https://adventofcode.com/2021/day/19)

### Part 1
```python

```

### Part 2
```python

```


## Day 20
[Here is the problem link.](https://adventofcode.com/2021/day/20)

### Part 1
```python

```

### Part 2
```python

```


## Day 21
[Here is the problem link.](https://adventofcode.com/2021/day/21)

### Part 1
```python

```

### Part 2
```python

```


## Day 22
[Here is the problem link.](https://adventofcode.com/2021/day/22)

### Part 1
```python

```

### Part 2
```python

```


## Day 23
[Here is the problem link.](https://adventofcode.com/2021/day/23)

### Part 1
```python

```

### Part 2
```python

```


## Day 24
[Here is the problem link.](https://adventofcode.com/2021/day/24)

### Part 1
```python

```

### Part 2
```python

```


## Day 25
[Here is the problem link.](https://adventofcode.com/2021/day/25)

### Part 1
```python

```

### Part 2
```python

```