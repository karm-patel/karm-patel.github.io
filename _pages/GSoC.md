---
layout: archive
title: "GSoC"
permalink: /GSoC/
author_profile: true
---

# Google Summer of Code 2022: Final Report
Getting accepted as a GSoC contributor and working with Dr. Kevin Murphy was a dream!

In this post, I will share my GSoC'22 journey with the TensorFlow organization on the project "JAX ML Textbook" mentored by [Dr. Kevin Murphy](https://www.cs.ubc.ca/~murphyk/). The project contains the tasks related to the upcoming textbook [Probabilistic Machine Learning: Advanced Topics](https://probml.github.io/pml-book/book2.html) (book2) and the existing textbook [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html) (book1) by Dr. Murphy. I contributed in mainly two repos - 1) [probml/pyprobml](https://github.com/probml/pyprobml): A public repo where the code is published. 2) [probml/bookv2](https://github.com/probml/bookv2): A private repo for the changes in the latex source code of the textbook.

## Highlights
- I worked on this project for 16 weeks and on an average 38 hours a week.
- Created 75 PRs in public `probml/pyprobml` repo, 25 PRs in [probml/probml-utils](https://github.com/probml/probml-utils), and 35 PRs in private `probml/bookv2` repo
- I learned many concepts of probabilistic machine learning
- I learned about the [JAX](https://github.com/google/jax) library for high-performance machine learning
- I learned some GitHub tricks such as managing workflows 

## Main contributions
I had broadly 3 main tasks in my GSoC project

| No. | Task | Time allocated (in %) |
| -- | -- | -- |
|1 | JAX implementation of algorithms | 30% |
|2 | Improving figures quality by matplotlib tricks | 40% |
|3 | Refactoring the codebase | 30% |
 
  
### 1. JAX implementation of algorithms:
In this task, I have converted existing python code to the JAX framework or created a new educational demo to create figures in the textbook. I have mainly worked on examples of Markov Chain Monte Carlo (MCMC) algorithms, however, I have created a few notebooks for other topics also. Major work is as following:

1) **Automatic Differentiation Variational Inference (ADVI) from scratch in JAX [[Notebook]( https://github.com/probml/pyprobml/blob/master/notebooks/book2/07/advi_beta_binom.ipynb)]:** This was my first demo which I have created for the textbook, in which I have implemented ADVI algorithm from scratch on a beta-binomial model by referring the original ADVI [paper](https://arxiv.org/abs/1603.00788). Where I started to explore JAX tricks such as using lax.scan() instead of for loop to speed up program (by 50x in this demo!).


<details>
<summary>Figure</summary>
<!-- ||
|--|
| ![image](https://user-images.githubusercontent.com/59387624/189528975-1cd09f04-dc01-4a24-8b8b-f0fa0ab908e8.png) | -->
<img alt="Description" src="https://user-images.githubusercontent.com/59387624/189528975-1cd09f04-dc01-4a24-8b8b-f0fa0ab908e8.png">

</details>

2) **Posterior of a beta-Bernoulli model using MCMC [[PR](https://github.com/probml/probml-notebooks/pull/94)] [[Notebook](https://github.com/probml/pyprobml/blob/master/notebooks/book2/07/hmc_beta_binom.ipynb)]:** Approximated posterior of beta-binomial model using blackjax sampling library and No U-turn Sampling algorithm (NUTS). In this demo first time I introduced with blackjax library which I used a lot in upcoming demos.

<!-- ||
|--|
| ![image](https://user-images.githubusercontent.com/59387624/189529931-a19375ea-e05b-4b60-b7a8-46fd43c3c7d8.png) | -->

<details>
<summary>Figure</summary>
<!-- ||
|--|
| ![image](https://user-images.githubusercontent.com/59387624/189528975-1cd09f04-dc01-4a24-8b8b-f0fa0ab908e8.png) | -->
<img alt="Description" src="https://user-images.githubusercontent.com/59387624/189529931-a19375ea-e05b-4b60-b7a8-46fd43c3c7d8.png">

</details>

3) **Laplace approximation for beta-binomial model [[PR](https://github.com/probml/probml-notebooks/pull/93)] [[Notebook](https://github.com/probml/pyprobml/blob/master/notebooks/book1/04/laplace_approx_beta_binom_jax.ipynb)]:** Approximated the posterior of beta-bernoulli model for coin toss problem using Laplace approximation method from scratch.

<!-- ||
|--|
| ![image](https://user-images.githubusercontent.com/59387624/189529104-aeb71750-2fdd-476c-b53d-317b185f027f.png) | -->

<details>
<summary>Figure</summary>
<!-- ||
|--|
| ![image](https://user-images.githubusercontent.com/59387624/189528975-1cd09f04-dc01-4a24-8b8b-f0fa0ab908e8.png) | -->
<img alt="Description" src="https://user-images.githubusercontent.com/59387624/189529104-aeb71750-2fdd-476c-b53d-317b185f027f.png">

</details>

4) **Markov chain convergence on uniform distribution [[Notebook](https://github.com/probml/pyprobml/blob/master/notebooks/book2/12/random_walk_integers.ipynb)] [[PR](https://github.com/probml/pyprobml/pull/902)]:** Recreated the figure which illustartes markov chain convergence.  

<!-- ||
|--|
| ![image](https://user-images.githubusercontent.com/59387624/189530086-657f56dd-d6f3-470d-9fcd-0d924237d0aa.png) | -->

<details>
<summary>Figure</summary>
<!-- ||
|--|
| ![image](https://user-images.githubusercontent.com/59387624/189528975-1cd09f04-dc01-4a24-8b8b-f0fa0ab908e8.png) | -->
<img alt="Description" src="https://user-images.githubusercontent.com/59387624/189530086-657f56dd-d6f3-470d-9fcd-0d924237d0aa.png">

</details>

5) **MCMC's diagnostic: R-hat [[PR](https://github.com/probml/pyprobml/pull/913)] [[Notebook](https://github.com/probml/pyprobml/blob/master/notebooks/book2/12/rhat_slow_mixing_chains.ipynb)]:** R-hat (potential scale reduction) is MCMC diagnostic to quantify convergence of MCMC samples. I have reproduced the figure given in this [paper](https://arxiv.org/abs/1903.08008) to illustrate the difference between split R-hat and non split R-hat.

<!-- ||
|--|
| ![image](https://user-images.githubusercontent.com/59387624/189530695-ea67ed20-000b-4ba0-a0c9-1f3570029d4e.png) | -->

<details>
<summary>Figure</summary>
<!-- ||
|--|
| ![image](https://user-images.githubusercontent.com/59387624/189528975-1cd09f04-dc01-4a24-8b8b-f0fa0ab908e8.png) | -->
<img alt="Description" src="https://user-images.githubusercontent.com/59387624/189530695-ea67ed20-000b-4ba0-a0c9-1f3570029d4e.png">

</details>

6) **MCMC's diagnostic: Trace plots and rank plots [[PR](https://github.com/probml/pyprobml/pull/908)] [[Notebook](https://github.com/probml/pyprobml/blob/master/notebooks/book2/12/mcmc_traceplots_unigauss.ipynb)]:** Again trace plots and rank plots are MCMC diagnostics which is used to judge convergence. I have reproduced the numpyro demo into JAX using blackjax library. Here I have used the arviz library (famous plotting library for PML algorithms) with blackjax first time.    

<!-- | | | |
| -- | -- | -- |
| Diffuse prior | ![image](https://user-images.githubusercontent.com/59387624/189530159-ff7ddd04-e91f-43c6-ba3c-4efe8abb476e.png)| ![image](https://user-images.githubusercontent.com/59387624/189530177-52af3260-f929-4be1-90ab-09eac9599def.png)|
| Sensible prior | ![image](https://user-images.githubusercontent.com/59387624/189530276-777afd64-f6dd-491c-8fac-651209ba6bd0.png) | ![image](https://user-images.githubusercontent.com/59387624/189530256-c18296d6-afd3-4b67-931e-8a57f59e0db2.png) | -->


<details>
<summary>Figure</summary>
<!-- ||
|--|
| ![image](https://user-images.githubusercontent.com/59387624/189528975-1cd09f04-dc01-4a24-8b8b-f0fa0ab908e8.png) | -->
<table>
<tr>
<th> </th> <th> </th> 
</tr>

<tr>
<td> Diffuse prior </td>
<td> <img alt="Description" src="https://user-images.githubusercontent.com/59387624/189530159-ff7ddd04-e91f-43c6-ba3c-4efe8abb476e.png"></td>

<td> <img alt="Description" src="https://user-images.githubusercontent.com/59387624/189530177-52af3260-f929-4be1-90ab-09eac9599def.png"></td>
</tr>

<tr>
<td> Sensible prior </td>
<td> <img alt="Description" src="https://user-images.githubusercontent.com/59387624/189530276-777afd64-f6dd-491c-8fac-651209ba6bd0.png"></td>

<td> <img alt="Description" src="https://user-images.githubusercontent.com/59387624/189530256-c18296d6-afd3-4b67-931e-8a57f59e0db2.png"></td>
</tr>

</table>

</details>

7) **Non centered parameterization in hierarchical bayesian model [[PR](https://github.com/probml/pyprobml/pull/918)] [[Notebook](https://github.com/probml/pyprobml/blob/master/notebooks/book2/12/neals_funnel.ipynb)]:** Recreated [example](https://num.pyro.ai/en/stable/examples/funnel.html) of pymc in JAX which shows the problem of using bayesian hierarchical model without using reparameterization trick.

<!-- ||
|--|
| ![image](https://user-images.githubusercontent.com/59387624/189530056-ca0deed7-6f06-44d5-9499-6b30a43f2741.png)| -->

<details>
<summary>Figure</summary>
<!-- ||
|--|
| ![image](https://user-images.githubusercontent.com/59387624/189528975-1cd09f04-dc01-4a24-8b8b-f0fa0ab908e8.png) | -->
<img alt="Description" src="https://user-images.githubusercontent.com/59387624/189530056-ca0deed7-6f06-44d5-9499-6b30a43f2741.png">

</details>


8) **Bayesian neural networks (BNN) using SGD & SGLD [[PR](https://github.com/probml/pyprobml/pull/987)] [[Notebook](https://github.com/probml/pyprobml/blob/master/notebooks/book2/19/bnn_mnist_sgld_blackjax.ipynb)]:** This demo compare prediction's uncertainty between bayesian algorithm (SGLD) and non-bayesian algorithm (SGD). In this demo first I explored the flax library to create multi-layer perceptron (MLP) neural network model. Then I used SGLD algorithm using the blackjax library for sampling the weights. 

<!-- ||
|--|
| ![image](https://user-images.githubusercontent.com/59387624/189530467-a77605aa-0583-4dc9-8467-5533d9651cee.png) | -->

<details>
<summary>Figure</summary>
<!-- ||
|--|
| ![image](https://user-images.githubusercontent.com/59387624/189528975-1cd09f04-dc01-4a24-8b8b-f0fa0ab908e8.png) | -->
<img alt="Description" src="https://user-images.githubusercontent.com/59387624/189530467-a77605aa-0583-4dc9-8467-5533d9651cee.png">

</details>


9) **Change of variable in Hamiltonian Monte Carlo (HMC) [[PR]()] [[Notebook]()]:** In this demo I have illustrated the need for a change of variable while defining the log joint density function of bayesian hierarchical models. This is done implicitly in high-level inference libraries such as pymc, pyro, numpyro, etc but we need to do it manually while using a low level library such as blackjax. I have also added this demo in blackjax library's examples **[[PR](https://github.com/blackjax-devs/blackjax/pull/254)]**.

<!-- ||
|--|
| ![image](https://user-images.githubusercontent.com/59387624/189529995-0d0d647a-8f6b-40ca-a325-a00664190d81.png) | -->


<details>
<summary>Figure</summary>
<!-- ||
|--|
| ![image](https://user-images.githubusercontent.com/59387624/189528975-1cd09f04-dc01-4a24-8b8b-f0fa0ab908e8.png) | -->
<img alt="Description" src="https://user-images.githubusercontent.com/59387624/189529995-0d0d647a-8f6b-40ca-a325-a00664190d81.png">

</details>

### 2. Improving figures quality by matplotlib tricks:
In the first draft of the book, most of the figures were made according matplotlib (A plotting library) defaults which were not fitting with textbook settings, it had some issues such as fonts in figures were small compared to caption; font style did not match with text; some labels were missing such as x-label, y-label, legend, etc. I synchronized with Zeel (GSoC contributor) and improved almost all needed figures by the `latexification` of the figures. One example of improvement as shown in the following.   

| Before PR | After PR |
| :-: | :-: |
|![image](https://user-images.githubusercontent.com/59387624/160594173-dbbf0e05-b27d-4a64-b4d1-45191ec1b2b6.png)|![image](https://user-images.githubusercontent.com/59387624/160594021-60162b05-0cd0-42d9-b22a-6dabace6dda0.png)

I can classify this task into 3 sub-taks:

| No. | Sub-task | Description/PRs |
| -- | -- | -- |
| 1. | Latexification of figures | Examples PR: [#723](https://github.com/probml/pyprobml/pull/723) [#726](https://github.com/probml/pyprobml/pull/726) [#891](https://github.com/probml/pyprobml/pull/891) [#1101](https://github.com/probml/pyprobml/pull/1101) [#1038](https://github.com/probml/pyprobml/pull/1038)|
| 2. | Editing latex source code to add updated figure | All the edits done in private `bookv2/` repo |
| 3. | Reviewing code of Non-GSoC contributors who were helping us to complete this task | I reviewed 50+ PRs in [pyprobml-review/pyprobml-review](https://github.com/pyprobml-review/pyprobml-review) repo. |

### 3. Refactoring pyprobml repo for better management: 
There are 425+ notebooks in the pyprobml repo which contains code that uses almost all ML libraries of Python. To manage this large codebase we need mechanisms to manage it well. In my GSoC, I synchronized with Zeel in contributing refactoring tasks including but not limited to organizing structure of repo, converting .py to .ipynb notebooks, redirection of figures' code from the textbook, Creating workflows on PR, Creating dashboards of notebooks, generating well-organized readme files, dead url checking in textbook, converting pdfs of figures to cmyk format. resolving comments on the book by MIT press, etc. I have enlisted some parts of this tasks in the following table: 

| Task | PR |
| -- | -- |
| Resolving notebook errors raised due to library update or other reasons| [#765](https://github.com/probml/pyprobml/pull/756) [#774](https://github.com/probml/pyprobml/pull/774) [#935](https://github.com/probml/pyprobml/pull/935) [#936](https://github.com/probml/pyprobml/pull/936) [#960](https://github.com/probml/pyprobml/pull/960) |
| Detected dead URLs in book and created a dummy notebook for figures which has more than one notebook. This dummy notebook contains links to original notebooks. | [#781](https://github.com/probml/pyprobml/pull/781) |
| Added auto-generated reademe.md files for book1 | [#815](https://github.com/probml/pyprobml/pull/815) [#816](https://github.com/probml/pyprobml/pull/816) |
| Moved tutorials notebooks to corresponding chapter programmatically | [#841](https://github.com/probml/pyprobml/pull/841) |
| Moved 49 duplicated notebooks having the same names from notebooks/misc to deprecated/| [#858]( https://github.com/probml/pyprobml/pull/858) |
| Added mapping of figure name to figure height, which overcomes the problem of seeing in latex source code of book while setting figure height in code. | [#869](https://github.com/probml/pyprobml/pull/896) [#911](https://github.com/probml/pyprobml/pull/911) |
| Added [notebooks.md](https://github.com/probml/pyprobml/blob/auto_notebooks_md/notebooks.md) which is an index of all notebooks, reader will be redirected to the notebook's entry in this table after clicking on the hyperlink given in figure's caption. | [#932](https://github.com/probml/pyprobml/pull/932) |
| Renamed notebook names from camel case to snake case | [#942](https://github.com/probml/pyprobml/pull/942) |
| Removed suffix like `_pymc`, `_blackjax`, etc from book2 notebooks | [#940](https://github.com/probml/pyprobml/pull/940) |
| Updated book1 redirection links in the firestore database to introduce new redirection to notebooks.md | [#1100]( https://github.com/probml/pyprobml/pull/1100) |

## Conclusion
GSoC'22 was high learning experience for me, In this period I improved a lot in both technical and non-technical skills. Special thanks to mentor Dr. Murphy and Prof. Nipun Batra for their consistent support and this amazing opportunity. I would also like to thank Zeel for always helping me whenever I get stuck.