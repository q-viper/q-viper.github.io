---
title:  "Introduction to Probability for Data Science: Getting Started"
date:   2022-04-24 09:29:17 +0545
categories:
    - Data Analysis
    - Probability
    - Statistics
tags:
    - data analysis
    - probability
    - statistics
# header:
#   teaser: assets/timeseries_analysis/modeling/output_50_0.png
---

## Introduction

Hello there welcome to the new blog series about Probability in the Data Science field. Here in this blog, we will start from basic concepts needed in using Probability in some datasets. This blog is going to be very short and basic yet informative.

Probability is all about measurement of some event's occurrence. We are not far from probability in our daily life. In fact, we have been using Probability from the very beginning of our civilization. Some of best example of probability being unknowingly used can be:
* A farmer assuming that if this amount of pesticide will harm to plant itself or only the pests.
* A chef assuming that if s/he fried chips for 5 minutes then it will be tasty.
* A stock broker can assume that tomorrow's price will rise.
* A student believing that studying for 5 hours a day will pass the exam.
* A driver hitting brakes.

The list goes on. If we sit down and grab a coffee to think more about this, then there is a high probability that we will not realize there was some probability of thinking about this topic too. Okay now I am making recursive events.

## Terminologies Mostly Used
### Experiment
An experiment is done in some kind of simulated or predefined environment and upon which some outcome is expected and will be calculated. For example a media company will perform some campaign to launch their product based on the outcome. An experiment is called a random experiment when the result of the experiment have various possible outcomes. For example stock price rise/fall.

### Event
An event is a outcome of an experiment and it can be as simple as the customer buying product or not buying. In coin toss, we can either get head or tail and these two cases are events.

#### Exhaustive Cases
It is the total number of possible outcomes from an experiment conducted in random manner. For a rolling dice, the possible cases are any of (1, 2, 3, 4, 5, 6), but for one dice and a coin toss, the possible outcome will be the (1H, 2H, 3H, 4H, 5H, 6H, 1T, 2T, 3T, 4T, 5T, 6T). In above dice and coin toss example, the case to get head and dice face of even are (2H, 4H, 6H). Upon this, we will calculate the possibility.

#### Mutually Exclusive Cases
What is the possibility of getting head and tail at the same time? None. While occurring one event, if another event's occurrence is affected then those events or cases are mutually exclusive. For example rolling of dice where none of faces can occur at the same time.

#### Sample Space
It is the set of all the possible outcome from a random experiment. In a coin toss, it is {$\phi$, H, T, HT}.

#### Independent Cases
Two events are called independent events if occurring of one has nothing to do with occurring of another. For example, while tossing coin for number of times, the outcome of one toss does not affect outcome of another.

## Probability
If there are N equally likely, mutually exclusive events and m is the number of expected outcome of an even A, then the probability of occurrence of A can be given by:

$$
P(A) = \frac{\text{Expected number of outcome of an event A (m)}}{\text{Total number of outcomes (N)}}
$$


### Laws/Rules of Probability
#### Additive Law
IF A, B and C are three mutually exclusive events then the probability of occurrence of either of them is given by,

P(A or B or C) = P(A) + P(B) + P(C)

For example in a dice throw, the probability of occurring either 1, 2 or 3 can be written as, 1/6+1/6+1/6 = 1/2.

If A, B and C are not mutually exclusive events then we can calculate the probability of occurrence of either of them is,

P(A or B or C) = P(A) + P(B) + P(C) - P(A and B) - P(A and C) - P(B and C) + P(A and B and C)


Just like set theories, or is Union and and is intersection.

#### Multiplicative Law
If A, B and C are mutually exclusive events then occurrence of A, B and C can be written as,
P(A and B and C) = P(A) * P(B) * P(C)
In above example, the occurrence of 1 after 2 and after that 3 can be written as 1/6*1/6*1/6 = 1/136.

If A and B are two dependent events then P(A and B) = P(B)P(A|B) = P(A)P(B|A). Where P(B|A) means that probability of occurrence of event B given that A has already occurred

### Conditional Probability and Bayes' Theorem
The conditional probability of an event B given the A can be written as, P(A|B) = P(A and B) / P(B) when P(B) > 0. Similarly, P(B|A) = P(A and B) / P(A) when P(A)>0.

Naive Bayes' Theorem is one of most popular algorithm in ML that depends on conditional probability. Which can be calculated as below:

P(A/B) = P(A) P(B/A) / P(B)



## Probability Distribution

Probability distribution is all about explaining the occurrence of any event within given range. This distribution depends on the facts of data like mean, spread, skweness etc.

There are different types of probability distributions in statistics and most common are:
* Binomial Distribution
* Poisson Distribution
* Normal Distribution

It requires separate blog for the explanation about above distribution. In Data Science, the occurrence of events might not follow any distribution but we always hope them to follow normal curve and we even force it to behave like Normal Distribution.



Thank you so much for reading this blog and in the next part, we will explore some of distributions with plots and some coding too.


```python

```
