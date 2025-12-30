---
title: Flow Matching
date: 2025-12-16 22:24:00 +0800
img_path: /assets/img/
categories: [Technology]
tags: [machine-learning]     # TAG names should always be lowercase
pin: false
math: true
mermaid: false
---

最近在学习vision-Language-Action model，发现目前的Action Expert使用到了Flow Matching的方法，因此想了解一下Flow Matching的原理。Flow Matching是最近几年比较流行的生成模型范式。本文介绍Flow Matching的基本原理。

## Flow Matching

Flow Matching解决的问题是寻找一个连续的变换，将一个连续的分布X变换为另一个连续的分布Y。

### 应用场景

在深入到Flow Matching之前，先说明一下为什么要将“一个连续的分布X变换为另一个连续的分布Y”。比如，在文生图的场景中，我们希望将用户的语言输入转换为图片，那么这个问题就可以建模为“将语言分布转换为图片分布”。又比如在VLA中，Action expert的作用就是将VLM的输出分布转换为具体action分布（比如，电机角度等）。所以，这种分布之间的转换是很常见的生成问题的建模方式。

### 基本原理

Flow Matching 的核心思想是：构建一条从源分布 $p_0 = p$（通常是高斯噪声）到目标分布 $p_1 = q$（数据分布）的**概率路径** $(p_t)_{0 \leq t \leq 1}$，然后学习一个**速度场**来描述样本沿这条路径的运动。

#### 速度场与 ODE

想象每个样本点是一个粒子，速度场 $u_t: \mathbb{R}^d \to \mathbb{R}^d$ 告诉我们在时刻 $t$、位置 $x$ 的粒子应该往哪个方向、以多快的速度移动。这个速度场决定了一个**流** $\psi_t$，满足：

$$
\frac{d}{dt} \psi_t(x) = u_t(\psi_t(x)), \quad \psi_0(x) = x
$$

直观地说，如果我们从源分布采样一个点 $X_0 \sim p_0$，然后让它沿着速度场"流动"，那么在时刻 $t$ 它会到达 $X_t = \psi_t(X_0)$。如果速度场设计得当，那么 $X_1 = \psi_1(X_0)$ 就服从目标分布 $q$。

因此，**生成新样本的过程**就是：(1) 从高斯噪声采样 $X_0 \sim \mathcal{N}(0, I)$；(2) 求解上述 ODE 到 $t=1$，得到 $X_1$。

#### 概率路径的设计

一个常用的概率路径是**线性插值路径**（也叫 conditional optimal-transport path）：

$$
X_t = (1-t) X_0 + t X_1, \quad X_0 \sim \mathcal{N}(0, I), \; X_1 \sim q
$$

这个设计非常直观：在 $t=0$ 时是纯噪声，在 $t=1$ 时是数据样本，中间是两者的线性混合。

#### 训练目标

Flow Matching 的训练目标是让神经网络 $u_t^\theta$ 去拟合能够生成概率路径 $p_t$ 的真实速度场 $u_t$：

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, X_t} \left\| u_t^\theta(X_t) - u_t(X_t) \right\|^2
$$

但问题是，真实的边缘速度场 $u_t$ 涉及两个高维分布之间的联合变换，很难直接计算。

#### Conditional Flow Matching

巧妙的解决方案是：**条件化到单个数据点**。对于线性插值路径，给定目标样本 $X_1 = x_1$，条件速度场有一个简洁的闭式解：

$$
u_t(x | x_1) = \frac{x_1 - x}{1 - t}
$$

这就是说，条件速度场就是"指向目标点 $x_1$ 的方向"。

基于此，我们可以定义 **Conditional Flow Matching (CFM) Loss**：

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, X_0, X_1} \left\| u_t^\theta(X_t) - (X_1 - X_0) \right\|^2
$$

其中 $t \sim \mathcal{U}[0,1]$，$X_0 \sim \mathcal{N}(0, I)$，$X_1 \sim q$，$X_t = (1-t)X_0 + tX_1$。

一个关键的理论结果是：**CFM Loss 和原始 FM Loss 的梯度相同**，即 $\nabla_\theta \mathcal{L}_{\text{FM}} = \nabla_\theta \mathcal{L}_{\text{CFM}}$。这意味着我们可以用简单的 CFM Loss 来训练，却能得到正确的边缘速度场。

## 一些问题

**Q1:** 除了线性插值路径还有什么别的路径吗？不同路径是否对神经网络的学习有影响？

实际上，线性插值路径是一个最简单的路径，属于目前常用的一类路径gaussian paths:

$$
p_{t|1}(x|x_1) = \mathcal{N}(x; \alpha_t x_1, \sigma_t^2 I)
$$

其中，$\alpha_t$和$\sigma_t$是可调的参数。

不同的$\alpha_t$和$\sigma_t$的选择会被称之为“Scheduler”。比如，Variance Preserving Path和Variance Exploding Path的Scheduler分别是：

$$
\text{VP:}\quad \alpha_t \equiv 1, \sigma_0 \gg 1, \sigma_0 = 0
$$

$$
\text{VE:}\quad \alpha_t = e^{-\frac{1}{2}\beta_t}, \sigma_t = \sqrt{1 - e^{-\beta_t}}, \beta_0 \gg 1, \beta_1 = 0
$$

不同的Scheduler也会对不同的任务有不同的影响 [2]。

**Q2:** 是否存在别的建模“生成问题”的方式？

Diffusion Models, autoregressive models, GAN, VAE等都是建模“生成问题”的方式。

[Wancheng Lin](https://walden-lin.github.io/home/)给我推荐了[Generative Models and Physics by Yizhuang You](https://www.youtube.com/watch?v=rKbVUmSYsZ8&t=1s)，感觉比较神奇，有空再学习。

## Reference

[1] [Flow Matching Guide and Code](https://ai.meta.com/research/publications/flow-matching-guide-and-code/), Meta AI.

[2] [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/pdf/2011.13456), ICLR 2021.