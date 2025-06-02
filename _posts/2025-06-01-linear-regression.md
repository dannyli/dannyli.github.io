---
title: '线性回归'
date: 2025-06-01
permalink: /posts/2025/06/linear-regression/
tags:
  - optimizer
  - machine learning
author: Yang Li
---


线性回归问题（Linear Regression）
------

给定输入数据 $\mathbf{X}_{\text{raw}} \in \mathbb{R}^{m \times n}$，输出数据 $\mathbf{y} \in \mathbb{R}^m$，其中 $m$ 为数据长度，$n$ 为输入特征维度。 


把原始输入数据扩展为 $\mathbf{X} =  [ \boldsymbol{1}, \mathbf{X}_{\text{raw}} ] \in \mathbb{R}^{m \times (n + 1)}$，其中 $\boldsymbol{1} \in \mathbb{R}^m$ 并且其所有元素为 $1$。 

记 $x_j^{(i)} = \mathbf{X}(i,j)$，$\mathbf{x}^{(i)} = [x_0^{(i)}, x_1^{(i)}, x_2^{(i)}, \cdots, x_n^{(i)}]^{\top}$ 为第 $i$ 个训练样本的输入特征向量（我们定义 $x_0^{(i)} = 1$），$y^{(i)} = \mathbf{y}(i)$ 为第 $i$ 个训练样本的标签值，$\hat{y}^{(i)}$ 为对应的模型预测值，$\epsilon^{(i)} = \hat{y}^{(i)} - y^{(i)}$ 为模型预测误差（error）或残差（residual）。

考虑以下线性模型

$$
\hat{y}^{(i)} = h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}) = \theta_0 x_0^{(i)} + \theta_1 x_1^{(i)} + \cdots + \theta_n x_n^{(i)} = \sum_{j=0}^{n} \theta_j x_j^{(i)} = \boldsymbol{\theta}^{\top} \mathbf{x}^{(i)} 
$$

其中 $\boldsymbol{\theta} = [\theta_0, \theta_1, \cdots, \theta_n]^{\top} \in \mathbb{R}^{n + 1}$ 为该线性模型的参数。

我们希望找到一组参数 $\boldsymbol{\theta}$，使得以下均方误差（MSE）形式的损失函数最小

$$
\min_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}, \mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \cdots, \mathbf{x}^{(n)} )
$$

$$
 \mathcal{L} = \frac{1}{m} \sum_{i=1}^{m}  \mathcal{L}_i(\hat{y}^{(i)}, y^{(i)}) 
= \frac{1}{m} \sum_{i=1}^{m}  \frac{1}{2} \left( \epsilon^{(i)} \right)^2  
= \frac{1}{2 m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
$$

该损失函数可以更简化地表示为

$$
 \mathcal{L} = \frac{1}{2 m} \sum_{i=1}^{m} \left( \boldsymbol{\theta}^{\top} \mathbf{x}^{(i)} - y^{(i)} \right)^2
=  \frac{1}{2 m} (\mathbf{X} \boldsymbol{\theta} - \mathbf{y})^{\top}   (\mathbf{X} \boldsymbol{\theta} - \mathbf{y}) = \frac{1}{2 m} \Vert \mathbf{X} \boldsymbol{\theta} - \mathbf{y} \Vert^2
$$

梯度下降法（Gradient Descent）
------

梯度下降法迭代公式为：

$$
{\theta}_j \leftarrow {\theta}_j - \eta \frac{\partial \mathcal{L}}{\partial {\theta}_j} 
$$

其中 $\eta$ 为学习率。

对于**批量梯度下降法（Batch Gradient Descent）**，使用所有的数据点来计算梯度：

$$
\frac{\partial \mathcal{L}}{\partial {\theta}_j} = \frac{1}{m}  
\sum_{i=1}^{m} (\boldsymbol{\theta}^{\top} \mathbf{x}^{(i)} -  y^{(i)}) \cdot x_j^{(i)}
= \frac{1}{m}  (\mathbf{X} \boldsymbol{\theta} - \mathbf{y})^{\top} \mathbf{x}_j
= \frac{1}{m}  \mathbf{x}_j^{\top} (\mathbf{X} \boldsymbol{\theta} - \mathbf{y})
$$

其中 $ \mathbf{x}_j =  [\mathbf{x}_j^{(1)}, \mathbf{x}_j^{(2)}, \cdots \mathbf{x}_j^{(m)}]^{\top}$.

对于所有参数，迭代公式可以写为

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})
$$

$$
\nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})
= \left[\frac{\partial \mathcal{L}}{\partial {\theta}_0}, \frac{\partial \mathcal{L}}{\partial {\theta}_1}, \cdots, \frac{\partial \mathcal{L}}{\partial {\theta}_n}  \right]^{\top}
= \frac{1}{m}  \mathbf{X}^{\top} (\mathbf{X} \boldsymbol{\theta} - \mathbf{y}) 
$$

批量梯度下降法在数据量大的情况下计算量高。为了减少计算量，可以使用**随机梯度下降法（Stochastic Gradient Descent）**，每一步只使用一个随机选取样本点 $i = 1, 2, \cdots, m$ 来计算梯度：

$$
\frac{\partial \mathcal{L}}{\partial {\theta}_j} 
=  (\boldsymbol{\theta}^{\top} \mathbf{x}^{(i)} -  y^{(i)}) \cdot x_j^{(i)}
$$

$$
\nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})
= \left[\frac{\partial \mathcal{L}}{\partial {\theta}_0}, \frac{\partial \mathcal{L}}{\partial {\theta}_1}, \cdots, \frac{\partial \mathcal{L}}{\partial {\theta}_n}  \right]^{\top}
=  (\boldsymbol{\theta}^{\top} \mathbf{x}^{(i)} -  y^{(i)})  \mathbf{x}^{(i)}
$$

随机梯度下降法收敛速度和稳定性较差。改进收敛特性可以使用**小批量梯度下降法（Mini-Batch Gradient Descent）**：每一步选取一个大小为 $b$ 的批量 {% raw %}$\mathcal{B} \subset \{ 1, 2, \cdots, m \}${% endraw %} 样本数据来计算梯度：

$$
\frac{\partial \mathcal{L}}{\partial {\theta}_j} = \frac{1}{b}  
\sum_{i \in \mathcal{B}} (\boldsymbol{\theta}^{\top} \mathbf{x}^{(i)} -  y^{(i)}) \cdot x_j^{(i)}
$$

可以看到，如果 $b = 1$，小批量梯度下降法就是随机梯度下降法；如果 $b = m$，小批量梯度下降法就是批量梯度下降法。

最小二乘法（Least Squares）
------

批量梯度下降法的迭代公式可以写成以下形式：

$$
\boldsymbol{\theta}_{k + 1} = \boldsymbol{\theta}_{k} - \eta  \frac{1}{m}  \mathbf{X}^{\top} (\mathbf{X} \boldsymbol{\theta} - \mathbf{y}) 
$$

这里的下标 $k$ 表示迭代次数。以上公式可以写成：

$$
\frac{\boldsymbol{\theta}_{k + 1} - \boldsymbol{\theta}_{k}}{\frac{\eta}{m}} = \mathbf{X}^{\top} (\mathbf{X} \boldsymbol{\theta} - \mathbf{y}) 
$$

可以看出，如果迭代收敛，会有 

$$\boldsymbol{\theta}_{k + 1} = \boldsymbol{\theta}_{k}$$

则

$$
\mathbf{X}^{\top} (\mathbf{X} \boldsymbol{\theta} - \mathbf{y}) = 0
$$

其解为**正规方程（normal equation）**

$$
 \boldsymbol{\theta} = (\mathbf{X}^{\top} \mathbf{X})^{-1} \mathbf{X}^{\top}  \mathbf{y}  
$$

对应的方法就是**（普通）最小二乘法（Ordinary Least Squares, OLS）**。

## 参考资料 ##
- Stanford CS229: Machine Learning - Linear Regression and Gradient Descent | Lecture 2 (Autumn 2018)
[https://www.youtube.com/watch?v=het9HFqo1TQ](https://www.youtube.com/watch?v=het9HFqo1TQ)
- 维基百科 - 梯度下降法 [https://zh.wikipedia.org/zh-hans/梯度下降法](https://zh.wikipedia.org/zh-hans/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95)
- Wikipedia - Gradient Descent [https://en.wikipedia.org/wiki/Gradient_descent](https://en.wikipedia.org/wiki/Gradient_descent)
- Wikipedia - Ordinary Least Squares [https://en.wikipedia.org/wiki/Ordinary_least_squares](https://en.wikipedia.org/wiki/Ordinary_least_squares)