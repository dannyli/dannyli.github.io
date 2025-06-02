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


把原始输入数据扩展为 $\mathbf{X} =  [ \boldsymbol{1}, \mathbf{X}_{\text{raw}} ] \in \mathbb{R}^{m \times (n + 1)}$，其中 $\boldsymbol{1} \in \mathbb{R}^m$ 并且其所有元素为 $1$。$\mathbf{X}$ 也叫做**设计矩阵（Design Matrix）**。

记 $x_j^{(i)} = \mathbf{X}(i,j)$，$\mathbf{x}^{(i)} = [x_0^{(i)}, x_1^{(i)}, x_2^{(i)}, \cdots, x_n^{(i)}]^{\top}$ 为第 $i$ 个训练样本的输入特征向量（我们定义 $x_0^{(i)} = 1$），$y^{(i)} = \mathbf{y}(i)$ 为第 $i$ 个训练样本的标签值，$\hat{y}^{(i)}$ 为对应的模型预测值，$\epsilon^{(i)} = \hat{y}^{(i)} - y^{(i)}$ 为模型预测误差（error）或残差（residual）。

考虑以下线性模型

$$
\hat{y}^{(i)} = h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}) = \theta_0 x_0^{(i)} + \theta_1 x_1^{(i)} + \cdots + \theta_n x_n^{(i)} = \sum_{j=0}^{n} \theta_j x_j^{(i)} = \boldsymbol{\theta}^{\top} \mathbf{x}^{(i)} 
$$

其中 $\boldsymbol{\theta} = [\theta_0, \theta_1, \cdots, \theta_n]^{\top} \in \mathbb{R}^{n + 1}$ 为该线性模型的参数，也称为**回归系数（Regression Coefficients）**。

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

注意到原最优化问题是一个无约束的二次规划的问题（unconstrained linear programming），通过求驻点可以得到

$$
 \frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}} = \frac{1}{2 m} \mathbf{X}^{\top} (\mathbf{X} \boldsymbol{\theta} - \mathbf{y}) = 0
$$

其解为**正规方程（Normal Equation）**：

$$
 \boldsymbol{\theta} = (\mathbf{X}^{\top} \mathbf{X})^{-1} \mathbf{X}^{\top}  \mathbf{y}  
$$

为对应的方法就是**（普通）最小二乘法（Ordinary Least Squares, OLS）**。

梯度下降法（Gradient Descent）
------

可以看到最小二乘法需要计算大矩阵的逆，计算量很大。

梯度下降法通过迭代计算避免求逆运算。梯度下降法迭代公式为：

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

最大似然估计（MLE）推导最小二乘法
------

考虑线性回归模型：

$$
y^{(i)} = \boldsymbol{\theta}^\top \mathbf{x}^{(i)} - \epsilon^{(i)}, \quad \epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)
$$

由于误差服从高斯分布，观测输出的条件概率为：

$$
p(y^{(i)} \mid \mathbf{x}^{(i)}; \boldsymbol{\theta}) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{( \boldsymbol{\theta}^\top \mathbf{x}^{(i)} - y^{(i)})^2}{2\sigma^2} \right)
$$

则整个数据集的联合似然函数为：

$$
L(\boldsymbol{\theta}) = \prod_{i=1}^m p(y^{(i)} \mid \mathbf{x}^{(i)}; \boldsymbol{\theta})
= \left( \frac{1}{\sqrt{2\pi\sigma^2}} \right)^m \exp\left( -\frac{1}{2\sigma^2} \sum_{i=1}^m ( \boldsymbol{\theta}^\top \mathbf{x}^{(i)} - y^{(i)})^2 \right)
$$

对似然函数取对数，得到对数似然（log-likelihood）：

$$
\log L(\boldsymbol{\theta}) = -\frac{m}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^m (\boldsymbol{\theta}^\top \mathbf{x}^{(i)} - y^{(i)})^2
$$

由于第一项与 $\boldsymbol{\theta}$ 无关，最大化对数似然等价于最小化平方误差和：

$$
\boldsymbol{\theta}_{\text{MLE}} = \arg\min_{\boldsymbol{\theta}} \sum_{i=1}^m (\boldsymbol{\theta}^\top \mathbf{x}^{(i)} - y^{(i)})^2
$$

因此，最小二乘法等价于在高斯噪声假设下的最大似然估计。


最小二乘法的另一种推导
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

则得到

$$
\mathbf{X}^{\top} (\mathbf{X} \boldsymbol{\theta} - \mathbf{y}) = 0
$$

其解为正规方程

$$
 \boldsymbol{\theta} = (\mathbf{X}^{\top} \mathbf{X})^{-1} \mathbf{X}^{\top}  \mathbf{y}  
$$


加权最小二乘法与加权线性回归
------
线性回归问题中我们对于预测误差$\epsilon^{(i)}$有以下的假设：
- 零均值假设（Unbiasedness）：$\mathbb{E}[\epsilon^{(i)}] = 0$
- 同方差性（Homoscedasticity）：$\text{Var}(\epsilon^{(i)}) = \sigma^2$
- 误差之间独立（Independence）：$\text{Cov}(\epsilon^{(i)},\epsilon^{(j)}) = 0$ for $ i \neq j$
- 正态分布（Gaussian）：$\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$
- 
**加权线性回归（Weighted Linear Squares）**中没有同方差性假设，而是采用异方差性假设（heteroscedasticity）。这时，损失函数变为：

$$
 \mathcal{L} = \frac{1}{m} \sum_{i=1}^{m}  \frac{1}{2} w^{(i)} \left( \epsilon^{(i)} \right)^2  
= \frac{1}{2 m} \sum_{i=1}^{m} w^{(i)} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
$$

或者使用加权范数表示：

$$
 \mathcal{L} = \frac{1}{2 m} \Vert \mathbf{X} \boldsymbol{\theta} - \mathbf{y} \Vert^2_{\mathbf{W}}
$$

对应的迭代公式形式和线性回归相同，区别是梯度的表达式：

$$
\frac{\partial \mathcal{L}}{\partial {\theta}_j} = \frac{1}{m}  
\sum_{i=1}^{m}  w^{(i)} \cdot (\boldsymbol{\theta}^{\top} \mathbf{x}^{(i)} -  y^{(i)}) \cdot x_j^{(i)} \\
= \frac{1}{m}  (\mathbf{X} \boldsymbol{\theta} - \mathbf{y})^{\top} \mathbf{W} \mathbf{x}_j \\
= \frac{1}{m}  \mathbf{x}_j^{\top} \mathbf{W} (\mathbf{X} \boldsymbol{\theta} - \mathbf{y})
$$

其中 $ \mathbf{W} = \mathbf{W}^{\top} = \text{diag}(w^{(1)}, w^{(2)}, \cdots, w^{(m)})$ 为固定值的权重矩阵或协方差矩阵。

以及

$$
\nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})
= \left[\frac{\partial \mathcal{L}}{\partial {\theta}_0}, \frac{\partial \mathcal{L}}{\partial {\theta}_1}, \cdots, \frac{\partial \mathcal{L}}{\partial {\theta}_n}  \right]^{\top}
= \frac{1}{m}  \mathbf{X}^{\top} \mathbf{W} (\mathbf{X} \boldsymbol{\theta} - \mathbf{y}) 
$$

可推出其正规方程为

$$
 \boldsymbol{\theta} = (\mathbf{X}^{\top} \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^{\top} \mathbf{W} \mathbf{y}  
$$

为对应的方法为**加权最小二乘法（Weighted Least Squares, WLS）**。

局部加权最小二乘法与局部加权线性回归
------

局部加权线性回归（Locally Weighted Linear Regression, LWLR）是加权线性回归的一个变形，它的权重并非常数，而与预测点有关。最常用的权重表达式为**高斯核**：

$$ w^{(i)}(\mathbf{x}) = \exp{\left( -\frac{\Vert \mathbf{x} - \mathbf{x}^{(i)} \Vert^2}{ 2 \tau^2}\right)}$$

超参数带宽 $\tau$ 太小会出现过拟合，太大会退化为线性回归问题。需交叉验证调优获得最佳的 $\tau$ 值。

岭回归（Ridge Regression）
------

当特征矩阵 $\mathbf{X}$ 的列高度相关（线性相关或近似相关）时，$\mathbf{X}^{\top}\mathbf{X}$ 接近奇异或不可逆，OLS 的解会不稳定，受微小扰动影响巨大。拟合结果会出现过大的参数估计值、震荡和过拟合现象。

在岭回归中，损失函数添加了 $L_2$ 正则项

$$
 \mathcal{L} = \frac{1}{2 m} \Vert \mathbf{X} \boldsymbol{\theta} - \mathbf{y} \Vert^2 + \frac{\lambda}{2 m} \Vert \boldsymbol{\theta} \Vert^2
$$

其中 $\lambda$ 称为正则化强度。其正规方程为

$$
 \boldsymbol{\theta} = (\mathbf{X}^{\top}  \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^{\top} \mathbf{y}  
$$

此外，在高维场景下，OLS 可能拟合训练集很好，但在测试集上性能差，OLS 模型会学到了训练数据中的噪声。岭回归的正则项相当于对参数范数加惩罚，防止权重过大导致模型波动，起到“收缩”作用（Shrinkage），使模型更平滑、更稳健。

对应的梯度下降公式为

$$
\nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})
= \frac{1}{m}  (\mathbf{X}^{\top} (\mathbf{X} \boldsymbol{\theta} - \mathbf{y}) + \lambda \boldsymbol{\theta})
$$

Tikhonov 正则回归
------

Tikhonov 正则回归是岭回归的广义形式。对损失函数中正则项进行修改：

$$
 \mathcal{L} = \frac{1}{2 m} \Vert \mathbf{X} \boldsymbol{\theta} - \mathbf{y} \Vert^2 + \frac{\lambda}{2 m} \Vert \mathbf{R} \boldsymbol{\theta} \Vert^2
$$

其中 $\mathbf{R}$ 为正则矩阵。Tikhonov 正则回归对应的正规方程为

$$
 \boldsymbol{\theta} = (\mathbf{X}^{\top}  \mathbf{X} + \lambda \mathbf{R}^{\top} \mathbf{R})^{-1} \mathbf{X}^{\top} \mathbf{y}  
$$

对应的梯度下降公式为

$$
\nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})
= \frac{1}{m}  (\mathbf{X}^{\top} (\mathbf{X} \boldsymbol{\theta} - \mathbf{y}) + \lambda \mathbf{R}^{\top} \mathbf{R} \boldsymbol{\theta})
$$



## 参考资料 ##
- Stanford CS229: Machine Learning - Linear Regression and Gradient Descent | Lecture 2 (Autumn 2018)
[https://www.youtube.com/watch?v=het9HFqo1TQ](https://www.youtube.com/watch?v=het9HFqo1TQ)
- 维基百科 - 梯度下降法 [https://zh.wikipedia.org/zh-hans/梯度下降法](https://zh.wikipedia.org/zh-hans/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95)
- Wikipedia - Gradient Descent [https://en.wikipedia.org/wiki/Gradient_descent](https://en.wikipedia.org/wiki/Gradient_descent)
- Wikipedia - Ordinary Least Squares [https://en.wikipedia.org/wiki/Ordinary_least_squares](https://en.wikipedia.org/wiki/Ordinary_least_squares)