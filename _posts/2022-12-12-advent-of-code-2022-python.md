````markdown
---
layout: single
title: "Advent of Code 2022 Python Solutions: Days 1–4"
date: 2022-12-12 09:29:17 +0100
last_modified_at: 2026-06-16
categories:
  - Python
  - Challenge
tags:
  - "Advent of Code"
  - "Advent of Code 2022"
  - "Python"
  - "Python solutions"
  - "DSA"
  - "Programming challenge"
description: "My Advent of Code 2022 Python solutions for Days 1 to 4, with input handling, clean explanations, and beginner-friendly code."
excerpt: "A beginner-friendly walkthrough of my Advent of Code 2022 Python solutions for Days 1 to 4, including data preparation, input parsing, and clean Python code."
header:
  teaser: /assets/advent_of_code/stars.png
  og_image: /assets/advent_of_code/stars.png
  image_description: "Advent of Code stars used as a teaser image for Python solutions"
toc: true
toc_label: "Advent of Code 2022 Python Guide"
toc_icon: "code"
toc_sticky: true
read_time: true
share: true
related: true
classes: wide
---

Advent of Code is a yearly programming challenge where we solve small story-based problems and collect stars. I enjoy it because the problems usually start simple and slowly become harder. It is also a nice way to practice Python, problem solving, data structures, and algorithms.

I am not very strong in competitive programming, but I still like trying Advent of Code because it helps me improve step by step. In 2021, I reached Day 16 and sometimes checked hints from places like Reddit when I got stuck. For Advent of Code 2022, I started a little late, but I still wanted to solve the problems and document my approach.

This post contains my **Advent of Code 2022 Python solutions for Days 1 to 4**. I explain how I prepare input files, how I read the data, and how I solve each part.

My full code is also available in my GitHub repository: [Advent of Code solutions](https://github.com/q-viper/Adevent-Of-Code.git).

## How I Organize Advent of Code Inputs

For each day, I keep one text file inside the `data` directory. The file contains two parts:

1. the sample test input from the Advent of Code problem page
2. my personal puzzle input

I separate both parts using this line:

```text
Split From Here
````

For example, the file for Day 1 is saved as:

```text
data/day1.txt
```

The same pattern is used for the other days:

```text
data/day2.txt
data/day3.txt
data/day4.txt
```

This makes it easy to first test the solution on the sample input. If the sample answer is correct, I run the same code on my actual puzzle input.

## Creating Empty Data Files

I used the small script below to create empty input files for the remaining days.

```python
for i in range(2, 26):
    with open(f"data/day{i}.txt", "w") as fp:
        fp.writelines("Split From Here")
```

This only needs to be done once. After that, I copy the sample input and actual input into the correct file.

## Reader Function for Advent of Code Inputs

This is the helper function I use to read the input for each day.

```python
def get_data(day=1):
    """Return sample and real Advent of Code input as lists.

    The input file should be written as:

        [sample input]
        Split From Here
        [actual input]

    Parameters
    ----------
    day : int
        Advent of Code day number.

    Returns
    -------
    list[list[str]]
        A list with two elements:
        - data[0] is the sample input
        - data[1] is the real puzzle input
    """
    file_name = f"data/day{day}.txt"

    with open(file_name) as fp:
        data = fp.read().strip().split("Split From Here")
        data = [d.strip().split("\n") for d in data]
        return data
```

I use `data[0]` for the sample input and `data[1]` for the real input.

## Day 1: Calorie Counting

Problem link: [Advent of Code 2022 Day 1](https://adventofcode.com/2022/day/1)

Day 1 is about finding how many calories each Elf is carrying. Each Elf has a group of numbers, and empty lines separate the groups.

### Day 1 Part 1

**Question:** Find the Elf carrying the most Calories. How many total Calories is that Elf carrying?

```python
import numpy as np

data = get_data(day=1)[1]

ndata = np.array(data)
split_indexes = np.where(ndata == "")[0]

calorie_groups = np.split(ndata, split_indexes)

calories = np.array([
    np.delete(group, np.where(group == "")[0]).astype(int).sum()
    for group in calorie_groups
])

answer_part_1 = calories.max()
print(answer_part_1)
```

Answer:

```text
71924
```

The idea is simple:

* split the input wherever there is an empty line
* convert each group to integers
* calculate the sum for each Elf
* find the maximum sum

### Day 1 Part 2

**Question:** Find the total Calories carried by the top three Elves.

```python
sorted_calories = calories.copy()
sorted_calories.sort()

answer_part_2 = sorted_calories[-3:].sum()
print(answer_part_2)
```

Answer:

```text
210406
```

Here, I sort all calorie sums and then add the last three values.

## Day 2: Rock Paper Scissors

Problem link: [Advent of Code 2022 Day 2](https://adventofcode.com/2022/day/2)

Day 2 is based on Rock Paper Scissors. The first column is the opponent's move, and the second column has a different meaning depending on the part.

For Part 1:

* `A` means Rock
* `B` means Paper
* `C` means Scissors
* `X` means Rock
* `Y` means Paper
* `Z` means Scissors

### Day 2 Part 1

```python
data = get_data(day=2)

# Response scores:
# X = Rock, Y = Paper, Z = Scissors
response_value = {
    "X": 1,
    "Y": 2,
    "Z": 3,
}

elf_moves = "ABC"
my_moves = "XYZ"

winning_games = [
    f"{elf} {me}"
    for elf, me in zip(elf_moves, my_moves[1:] + my_moves[:1])
]

losing_games = [
    f"{elf} {me}"
    for elf, me in zip(elf_moves, my_moves[-1:] + my_moves[:-1])
]

score = 0

for game in data[1]:
    score += response_value[game[-1]]

    if game in winning_games:
        score += 6
    elif game in losing_games:
        score += 0
    else:
        score += 3

print(score)
```

Answer:

```text
10816
```

In this part, I first add the score for my selected shape. Then I add:

* `6` points for a win
* `3` points for a draw
* `0` points for a loss

### Day 2 Part 2

In Part 2, the second column does not directly represent my move. Instead:

* `X` means I need to lose
* `Y` means I need to draw
* `Z` means I need to win

```python
score = 0

for game in data[1]:
    opponent_move = game[0]
    expected_result = game[-1]

    if expected_result == "Z":
        my_move = [k for k in winning_games if opponent_move in k][0][-1]
        score += 6

    elif expected_result == "X":
        my_move = [k for k in losing_games if opponent_move in k][0][-1]
        score += 0

    else:
        my_move = my_moves[elf_moves.index(opponent_move)]
        score += 3

    score += response_value[my_move]

print(score)
```

Answer:

```text
11657
```

This time, I first decide which move I should play to get the required result. Then I calculate the score in the same way as Part 1.

## Day 3: Rucksack Reorganization

Problem link: [Advent of Code 2022 Day 3](https://adventofcode.com/2022/day/3)

Day 3 is about finding common item types in rucksacks. Lowercase and uppercase letters have different priorities.

The priority list is:

```python
import string

items = string.ascii_lowercase + string.ascii_uppercase
```

So:

* `a` has priority `1`
* `z` has priority `26`
* `A` has priority `27`
* `Z` has priority `52`

### Day 3 Part 1

Each rucksack has two compartments. We need to find the item type that appears in both compartments.

```python
data = get_data(day=3)

score = 0

for rucksack in data[1]:
    half_index = len(rucksack) // 2

    first_half = rucksack[:half_index]
    second_half = rucksack[half_index:]

    repeated_items = set(first_half).intersection(set(second_half))

    score += sum(items.index(item) + 1 for item in repeated_items)

print(score)
```

Answer:

```text
8240
```

I used Python sets here because `set.intersection()` makes it easy to find common letters.

### Day 3 Part 2

In Part 2, the rucksacks are divided into groups of three. We need to find the common item type in each group.

```python
rucksacks = np.array(data[1])
groups = np.split(rucksacks, np.arange(3, len(rucksacks), 3))

score = 0

for group in groups:
    common_items = (
        set(group[0])
        .intersection(set(group[1]))
        .intersection(set(group[2]))
    )

    score += sum(items.index(item) + 1 for item in common_items)

print(score)
```

Answer:

```text
2587
```

Again, sets make the solution short and readable.

## Day 4: Camp Cleanup

Problem link: [Advent of Code 2022 Day 4](https://adventofcode.com/2022/day/4)

Day 4 is about pairs of section ranges. We need to check whether one range fully contains another range, and later whether the two ranges overlap at all.

The input looks like this:

```text
2-4,6-8
2-3,4-5
5-7,7-9
2-8,3-7
6-6,4-6
2-6,4-8
```

### Day 4 Part 1

**Question:** In how many assignment pairs does one range fully contain the other?

```python
data = get_data(day=4)

score = 0

for group in data[1]:
    first_range, second_range = group.split(",")

    first_start, first_end = map(int, first_range.split("-"))
    second_start, second_end = map(int, second_range.split("-"))

    first_sections = set(range(first_start, first_end + 1))
    second_sections = set(range(second_start, second_end + 1))

    if first_sections.issubset(second_sections) or second_sections.issubset(first_sections):
        score += 1

print(score)
```

Answer:

```text
602
```

Here, I convert both ranges into sets. Then I check if one set is a subset of the other.

### Day 4 Part 2

**Question:** In how many assignment pairs do the ranges overlap at all?

```python
score = 0

for group in data[1]:
    first_range, second_range = group.split(",")

    first_start, first_end = map(int, first_range.split("-"))
    second_start, second_end = map(int, second_range.split("-"))

    first_sections = set(range(first_start, first_end + 1))
    second_sections = set(range(second_start, second_end + 1))

    if first_sections.intersection(second_sections):
        score += 1

print(score)
```

Answer:

```text
891
```

For this part, I only need to check whether the intersection between both sets is empty or not.

## What I Learned

These first four days are a good warm-up for Advent of Code 2022. The main ideas I used were:

* reading and splitting text input
* using lists and NumPy arrays
* sorting values
* working with dictionaries
* using Python sets for intersections and subset checks

The most useful part for me was using sets. They made Day 3 and Day 4 much easier to write and understand.

## Final Notes

This post is part of my Advent of Code 2022 Python solutions series. I use these challenges to practice Python and improve my problem-solving skills. Even if I do not finish every day, documenting the process helps me understand the solutions better.

```
::: ​​
```
