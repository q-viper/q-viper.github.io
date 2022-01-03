---
title: Writing Popular Machine Learning Optimizers From Scratch on Python
date: 2020-06-05T14:32:48+05:45
header:
  teaser: assets/wp-content/uploads/2020/06/opt.png
categories:
  - Artificial Intelligence
  - Machine Learning
  - Programming
tags:
  - gradient descent
  - machine learning
  - optimization
---
**Contents**
* TOC
{:toc}


# 1. Writing popular Machine Learning Optimizers from scratch on Python
I might stop to write new blogs in this site so please visit [dataqoil.com](https://dataqoil.com) for more cool stuffs.

This blog will include some mathematical and theoritical representation along with Python codes from scratch. Most of the codes and formulas are taken from different resources and I have given links to them also.

This post is related to below posts(these posts depends on this post):
* [Writing a Feed forward Neural Network from Scratch on Python]({{site.url}}/2020/05/30/writing-a-deep-neural-network-from-scratch-on-python/)
* [Convolutional Neural Networks from Scratch on Python]({{site.url}}/2020/06/05/convolutional-neural-networks-from-scratch-on-python/)

## Before begining,
If you want to check the files first, then please follow this link [ML From Basics](https://github.com/q-viper/ML-from-Basics).


# 2. Contains
([Optimizers Code were referenced from here.](https://www.github.com/ShivamShrirao/dnn_from_scratch))
* Gradient Descent
* Momentum
* RMS Prop 
* ADAM
* ADAGrad
* ADAMAX
* ADADelta



## 2.1 Initialize our class

```python
  class Optimizer:
    def __init__(self, layers, name=None, learning_rate = 0.01, mr=0.001):
        self.name = name
        self.learning_rate = learning_rate
        self.mr = mr
        keys = ["sgd", "iterative", "momentum", "rmsprop", "adagrad", "adam", "adamax", "adadelta"]
        values = [self.sgd, self.iterative, self.momentum, self.rmsprop, self.adagrad, self.adam, self.adamax, self.adadelta]
        self.opt_dict = {keys[i]:values[i] for i in range(len(keys))}
        if name != None and name in keys:
            self.opt_dict[name](layers=layers, training=False)
```

We are using the reference of every optimizer method. The object of this class will be made while compiling the model and at the same time, the reference of the optimizer is taken from the `opt_dict`. The boolean `training` sets all the terms(to 0) that are required for weight optimization.

Let me take some notation form the book Tensorflow for Dummies by Matthew Scarpino. Most of the formulas and concepts are taken from this book.
* The	set	of	trainable	variables	is	denoted	θ.	The	values	of	the	variables	at Step	t	is	denoted	θt. 
* The	loss,	which	is	a	mathematical	relationship	containing	the	model’s variables,	is	denoted	J(θ).	The	gradient	of	the	loss	is	∇J(θ). 
* The	learning	rate,	denoted	η,	is	a	value	that	affects	how	much	θj	changes from	step	to	step.

Before diving into algorithm and comparing it with code, let us understand that, I have done addition with all delta terms because I have already taken `minus` of delta terms.

## Gradient Descent
Weight update term for all units is:-
$$
\begin{equation}
\triangle w_{ji} = \alpha \delta_j x_{ji}
\end{equation}
\begin{equation}
\ when\ momentum\ term\ is\ applied\,
\end{equation}
\begin{equation}
\triangle w_{ji}(n) = \beta \delta_j x_{ji} + \triangle w_{ji}(n-1) 
\end{equation}
\begin{equation}
\ \beta\ is\ momentum\ rate
\end{equation}
\begin{equation}
\delta_j\ formula\ varies\ with\ the\ unit\ being\ output\ or\ internal. 
\end{equation}
\begin{equation}
w_{ji} = w_{ji} -  \triangle w_{ji}\\
\end{equation}
$$

OR more simple representation:-
$$
\begin{equation}
\theta_t = \theta_t - \alpha \triangle J(\theta)
\end{equation}
$$

This	shows	how	the	model’s	variables	change	with	each	training	operation. As	training	continues,	∇J(θ)	should	approach	zero,	which	means	that	each new	set	of	variables	should	be	approximately	equal	to	the	previous	set.	At this	point,	optimization	has	completed	because	the	optimizer	has	converged to a minimum. But Gradient Descent always suffers from convergence because the loss might never reach global minimum and oscillate between values.

## 2.3 Momentum Optimizer
This optimizer tries to eliminate the previous problem of Oscilation between values by introducing momentum term.
\begin{equation}
v_t = \beta v_{t-1} - \alpha \triangle J(\theta)\\
\theta = \theta + v_t
\end{equation}

```python
    def momentum(self, layers, learning_rate=0.1, beta1=0.9, weight_decay=0.0005, training=True):
        learning_rate = self.learning_rate
        for l in layers:
            if l.parameters !=0:
                if training:
                    l.weights_momentum = beta1 * l.weights_momentum + learning_rate * 
                                         l.delta_weights-weight_decay * learning_rate*l.weights
                    l.weights+=l.weights_momentum
                    l.biases_momentum = beta1 * l.biases_momentum + learning_rate * 
                                        l.delta_biases-weight_decay *learning_rate*l.biases
                    l.biases+=l.biases_momentum
                else:
                    l.weights_momentum = 0
                    l.biases_momentum = 0
```

The term `weight_decay` and `beta1` is not present on original Momentum Algorithm but it helps to slowly converge the loss towards global minima.

## 2.4 Adagrad
* The	learning	rate	changes	from	variable	to	variable	and	from	step	to	step. The	learning	rate	at	the	tth	step	for	the	ith	variable	is	denoted		. 
* Adagrad	methods	compute	subgradients	instead	of	gradients.	A subgradient	is	a	generalization	of	a	gradient	that	applies	to nondifferentiable	functions.	This	means	AdaGrad	methods	can	optimize both	differentiable	and	nondifferentiable	functions.

The learning rate will be:-
\begin{equation}
\alpha_{t,i} = \frac{\alpha}{\sqrt G_{t,ii}}
\end{equation}

In	this	equation,	Gt,	ii	is	the	ith	element	of	the	diagonal	of	the	matrix	formed by	taking	the	outer	product	of	the	subgradient	of	the	loss	with	itself.	After computing	the	learning	rates,	the	optimizer	updates	the	variables:
\begin{equation}
\theta_{t,i} = \theta_{t-1,i} - \alpha g_t
\end{equation}

Shorcoming of this optimizer is that, the learning rate eventually becomes 0 and training stops.

```python
    def adagrad(self, layers, learning_rate=0.01, beta1=0.9, epsilon=1e-8, training=True):
        learning_rate=self.learning_rate
        for l in layers:
            if l.parameters != 0:
                if training:
                    l.weights_adagrad += l.delta_weights ** 2
                    l.weights += learning_rate * (l.delta_weights/np.sqrt(l.weights_adagrad+epsilon))
                    l.biases_adagrad += l.delta_biases ** 2
                    l.biases += learning_rate * (l.delta_biases/np.sqrt(l.biases_adagrad+epsilon))
                else:
                    l.weights_adagrad = 0
                    l.biases_adagrad = 0
```

## 2.5 RMS Prop

* This algorithm is devised by Geoffrey Hinton. 
* This algorithm uses different learning rate for different parameters by using moving average of squared gradient. It utilizes the magnitude of recent gradient to normalize gradient.
* Divides learning rate by the average of exponential decay of squared grdients.

\begin{equation}
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{(1-\beta) g^2_{t-1} + \beta g_t + \epsilon}} * g_t
\end{equation}

Where g_t is a delta term for parameter 𝜃.

```python
    def rmsprop(self, layers, learning_rate=0.001, beta1=0.9, epsilon=1e-8, training=True):
        learning_rate=self.learning_rate
        for l in layers:
            if l.parameters !=0:
                if training:
                    l.weights_rms = beta1*l.weights_rms + (1-beta1)*(l.delta_weights ** 2)
                    l.weights += learning_rate * (l.delta_weights/np.sqrt(l.weights_rms + epsilon))
                    l.biases_rms = beta1*l.biases_rms + (1-beta1)*(l.delta_biases ** 2)
                    l.biases += learning_rate * (l.delta_biases/np.sqrt(l.biases_rms + epsilon))
                else:
                    l.weights_rms = 0
                    l.biases_rms = 0
```

## 2.6 Adam Optimizer
The	Adam	(Adaptive	Moment	Estimation)	algorithm	closely	resembles	the Adagrad	algorithm	in	many	respects.	It	also	resembles	the	Momentum algorithm	because	it	takes	two	factors	into	account:
* The	first	moment	vector:	Scales	the	gradient	by	`1-beta1`
* The	second	moment	vector:	Scales	the	square	of	the	gradient	by	`1-beta2`

These	moment	vectors	are	denoted	`mt` and	`vt`,	respectively.	
The	following equations	show	how	their	values	change	from	step	to	step:
$$
\begin{equation}
m_t = \beta_1 m_{t-1} + (1-\beta_1) \triangle J(\theta)\\
v_t = \beta_2 v_{t-1} + (1-\beta_2)[\triangle J(\theta)]^2\\
m^{\prime}_t = \frac{m_t}{1-\beta^t_1}\\
v^{\prime}_t = \frac{v_t}{1-\beta^t_2}\\
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{v^{\prime}_t} + \epsilon} * m^{\prime}_t
\end{equation}
$$
Where 𝜖 is a small value which is used to prevent from divide by 0.

```python
    def adam(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0, training=True):
        for l in layers:
            if l.parameters != 0:
                if training:
                    l.t += 1
                    if l.t == 1:
                        l.pdelta_biases = 0
                        l.pdelta_weights = 0
                    l.weights_adam1 = beta1 * l.weights_adam1 + (1-beta1)*l.delta_weights
                    l.weights_adam2 = beta2 * l.weights_adam2 + (1-beta2)*(l.delta_weights**2)
                    mcap = l.weights_adam1/(1-beta1**l.t)
                    vcap = l.weights_adam2/(1-beta2**l.t)
                    l.delta_weights = mcap/(np.sqrt(vcap) + epsilon)
                    l.weights += l.pdelta_weights * self.mr + learning_rate * l.delta_weights
                    l.pdelta_weights = l.delta_weights * 0
                    l.biases_adam1 = beta1 * l.biases_adam1 + (1-beta1)*l.delta_biases
                    l.biases_adam2 = beta2 * l.biases_adam2 + (1-beta2)*(l.delta_biases**2)
                    mcap = l.biases_adam1/(1-beta1**l.t)
                    vcap = l.biases_adam2/(1-beta2**l.t)
                    l.delta_biases = mcap/(np.sqrt(vcap) +epsilon)
                    l.biases += l.pdelta_biases * self.mr + learning_rate * l.delta_biases
                    l.pdelta_biases = l.delta_biases * 0                    
                else:
                    l.t = 0
                    l.weights_adam1 = 0
                    l.weights_adam2 = 0
                    l.biases_adam1 = 0
                    l.biases_adam2 = 0
```

## 2.7 Adamax
Please refer to the [Sebastian Ruder's site](https://ruder.io) for more explanation.

This is slight variation of Adam optimizer.
$$
\begin{equation}
u_t = \beta_2^{\infty}+ (1-\beta_2^\infty) * abs(g_t)^\infty\\
\ = max(\beta_2 * v_{t-1}, abs(g_t))\\
\text now, \\ 
\theta_{t+1} = \theta_t - \frac{\alpha}{u_t} * m^{\prime}_t
\end{equation}
$$

```python
    def adamax(self, layers, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8, training=True):
        for l in layers:
            if l.parameters != 0:
                if training:
                    l.weights_m = beta1*l.weights_m + (1-beta1)*l.delta_weights
                    l.weights_v = np.maximum(beta2*l.weights_v, abs(l.delta_weights))
                    l.weights += (self.learning_rate/(1-beta1))*(l.weights_m/(l.weights_v+epsilon))                    
                    l.biases_m = beta1*l.biases_m + (1-beta1)*l.delta_biases
                    l.biases_v = np.maximum(beta2*l.biases_v, abs(l.delta_biases))
                    l.biases += (self.learning_rate/(1-beta1))*(l.biases_m/(l.biases_v+epsilon))                    
                else:
                    l.weights_m = 0
                    l.biases_m = 0
                    l.weights_v = 0
                    l.biases_v = 0
```

## 2.8 Adadelta
* We don't use learning rate here. But the ratio of the running average of the previous time steps to the current gradient is used.
* This algorithm tries to reduce learning rate monotonically. This is extended version of Adagrad.
$$
\begin{equation}
\theta_{t+1} = \theta_t + \triangle \theta_t\\
\triangle \theta = - \frac{RMS[\triangle \theta_t-1]}{RMS[g_t]}.g_t
\end{equation}
$$
Where `gt` is the gradient term.

```python
    def adadelta(self, layers, learning_rate=0.01, beta1=0.9, epsilon=1e-8, training=True):
        for l in layers:
            if l.parameters != 0:
                if training:
                    l.weights_v = beta1*l.weights_v + (1-beta1)*(l.delta_weights ** 2)
                    l.delta_weights = np.sqrt((l.weights_m+epsilon)/(l.weights_v+epsilon))*l.delta_weights
                    l.weights_m = beta1*l.weights_m + (1-beta1)*(l.delta_weights)
                    l.weights += l.delta_weights                    
                    l.biases_v = beta1*l.biases_v + (1-beta1)*(l.delta_biases ** 2)
                    l.delta_biases = np.sqrt((l.biases_m+epsilon)/(l.biases_v+epsilon))*l.delta_biases
                    l.biases_m = beta1*l.biases_m+ (1-beta1)*(l.delta_biases)
                    l.biases += l.delta_biases                    
                else:
                    l.weights_m = 0
                    l.biases_m = 0
                    l.weights_v = 0
                    l.biases_v = 0
```

# My other blogs:-
* [Writing Popular Machine Learning Optimizers from Scratch on Python]({{site.url}}/2020/06/05/writing-popular-machine-learning-optimizers-from-scratch-on-python/)
* [Writing Image Processing Class From Scratch on Python]({{site.url}}/2020/05/30/image-processing-class-from-scratch-on-python/)
* [Writing a Deep Neural Network from Scratch on Python]({{site.url}}/2020/05/30/writing-a-deep-neural-network-from-scratch-on-python/)
* [Convolutional Neural Networks from Scratch on Python]({{site.url}}/2020/06/05/convolutional-neural-networks-from-scratch-on-python/)

# References
* Tensorflow for Dummies by Matthew Scarpino
* [Optimizers code were referenced from here](https://www.github.com/ShivamShrirao/dnn_from_scratch)
* [An Overview of Gradient Descent Optimization Algorithms](https://ruder.io/optimizing-gradient-descent/index.html)

<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Link-to-repo">Link to repo<a class="anchor-link" href="#Link-to-repo">¶</a></h1><ul>
<li><a href="#">Please follow this link for the github repo</a></li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="References">References<a class="anchor-link" href="#References">¶</a></h1><ul>
<li>Tensorflow for Dummies by Matthew Scarpino</li>
<li><a href="https://www.github.com/ShivamShrirao/dnn_from_scratch">Optimizers code were referenced from here</a></li>
<li><a href="https://ruder.io/optimizing-gradient-descent/index.html">An Overview of Gradient Descent Optimization Algorithms</a></li>
</ul>

</div>
</div>
</div>
    
<!-- /wp:html -->