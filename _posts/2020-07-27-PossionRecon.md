---
layout: post
title: 泊松表面重建算法
description: "Possion Reconstruction算法论文阅读（补充了涉及到的数学知识）"
modified: 2020-7-27
tags: [Geometric, Papers, Visualization]
math: true
image:
  feature: abstract-5.jpg
---
<details open><!-- 可选open -->
<summary>Contents</summary>
<div markdown="1">
<!-- TOC -->

- [泊松重建 Possion Reconstruction](#%E6%B3%8A%E6%9D%BE%E9%87%8D%E5%BB%BA-possion-reconstruction)
    - [三维指示函数拟合](#%E4%B8%89%E7%BB%B4%E6%8C%87%E7%A4%BA%E5%87%BD%E6%95%B0%E6%8B%9F%E5%90%88)
        - [隐函数拟合](#%E9%9A%90%E5%87%BD%E6%95%B0%E6%8B%9F%E5%90%88)
        - [求解方法](#%E6%B1%82%E8%A7%A3%E6%96%B9%E6%B3%95)
    - [等值面提取](#%E7%AD%89%E5%80%BC%E9%9D%A2%E6%8F%90%E5%8F%96)
- [【附：相关数学知识】](#%E9%99%84%E7%9B%B8%E5%85%B3%E6%95%B0%E5%AD%A6%E7%9F%A5%E8%AF%86)
    - [泊松方程定义与求解（图像场景）](#%E6%B3%8A%E6%9D%BE%E6%96%B9%E7%A8%8B%E5%AE%9A%E4%B9%89%E4%B8%8E%E6%B1%82%E8%A7%A3%E5%9B%BE%E5%83%8F%E5%9C%BA%E6%99%AF)
    - [矢量分析：梯度、散度、旋度](#%E7%9F%A2%E9%87%8F%E5%88%86%E6%9E%90%E6%A2%AF%E5%BA%A6%E6%95%A3%E5%BA%A6%E6%97%8B%E5%BA%A6)
    - [向量的点乘（内积）、叉乘（外积）](#%E5%90%91%E9%87%8F%E7%9A%84%E7%82%B9%E4%B9%98%E5%86%85%E7%A7%AF%E5%8F%89%E4%B9%98%E5%A4%96%E7%A7%AF)

<!-- /TOC -->
</div>
</details>

## 泊松重建 (Possion Reconstruction)

[论文](http://hhoppe.com/poissonrecon.pdf)	[代码](https://github.com/mkazhdan/PoissonRecon)

> Reconstructing 3D surfaces from point samples

### 三维指示函数拟合

输入：有法向量的点集$$S$$（有向点云）

样本点 $$s\in{S}$$ 位于未知模型 $$M$$ 的表面 $$∂M$$ 附in近，且必须包含两个属性： $$s.p$$（坐标）和 $$s.\overrightarrow{N}$$（朝内的法向）。 

输出：指示函数（indicator function）$$\chi$$，用于确定表里

<!--more-->

#### 隐函数拟合

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727091834595.png" alt="image-20200727091834595" style="zoom:80%;" />

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727092147103.png" alt="image-20200727092147103" style="zoom:80%;" />

> Key insight: The gradient of the indicator function is a vector field that is zero almost everywhere (since the indicator function is constant almost everywhere), except at points near the surface, where it is equal to the inward surface normal. 
>
> Thus, the oriented point samples can be viewed as samples of the gradient of the model’s indicator function.

#### 求解方法

1. 证明：*平滑指示函数的梯度* 等于 *平滑表面法向场得到的向量场* 

   <img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727111811022.png" style="zoom:50%;" />

   其中$$\overrightarrow N_{\partial_{M}}(p)$$为$$p\in∂M$$处的法向量。即$$\nabla\widetilde\chi=\overrightarrow{V}$$

2. 用离散求和近似的方式去计算曲面积分，算出向量场$$\overrightarrow{V}$$：

   <img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727111653923.png" alt="image-20200727112002410" style="zoom:50%;" />

   其中用于平滑的滤波器$$\overrightarrow{F}$$要满足两个条件：1.足够窄（避免数据过拟合）2. 足够宽（$$\mathcal P_S$$处的积分能较好地被the value at s.p scaled by the patch area估计）。可选：方差约等于采样分辨率的高斯滤波器。

3. 向量场是不可积的，只能近似。因此利用最小二乘估计，引入散度算子组成泊松方程。求$$\chi$$即解泊松方程：

   $$\Delta\widetilde\chi\equiv\nabla^2\widetilde\chi\equiv\nabla\cdot{\nabla\widetilde\chi}=\nabla\cdot{\overrightarrow{V}}$$

   即$$\chi$$的拉普拉斯算子(梯度的散度)等于向量场的散度。

### 等值面提取

2.1 计算等值面值（isovalue）

evaluating $$\widetilde\chi$$ at the sample positions and use the average of the values for isosurface extraction:

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727112136417.png" alt="image-20200727112136417" style="zoom:50%;" />

2.2 通过定义在八叉树上的[Marching Cubes](https://e0hyl.github.io/BLOG-OF-E0/MarchingCube/)方法提取等值面

## 【附：相关数学知识】

### 泊松方程定义与求解（图像场景）

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727093446421.png" alt="image-20200727093446421" style="zoom:80%;" />

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727094105592.png" alt="image-20200727094105592" style="zoom:80%;" />

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727094800681.png" alt="image-20200727094800681" style="zoom:50%;center;" />

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727094908015.png" alt="image-20200727094908015" style="zoom:80%;" />

[泊松图像融合](https://zhuanlan.zhihu.com/p/68349210)

### [矢量分析：梯度、散度、旋度](https://zhuanlan.zhihu.com/p/22654688)

向量场**A**，数量场u：

▽称为汉密尔顿算子； ▽·▽=△，△称为[拉普拉斯算子](https://blog.csdn.net/qq_30815237/article/details/86543091)。

【梯度】▽u	【散度】**▽**·**A**（点乘结果为数）

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727090328746.png" alt="image-20200727090328746" style="zoom:80%;" />

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727090354194.png" alt="image-20200727090354194" style="zoom:80%;" />

本意是一个向量（矢量），表示某一函数在该点处的方向导数沿着该方向取得最大值，即函数在该点处沿着该方向（此梯度的方向）变化最快，变化率最大（为该梯度的模）

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727090409484.png" alt="image-20200727090409484" style="zoom:80%;" />

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727090434092.png" alt="image-20200727090434092" style="zoom:80%;" />

### [向量的点乘（内积）、叉乘（外积）](https://www.cnblogs.com/gxcdream/p/7597865.html)

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727090716101.png" alt="image-20200727090716101" style="zoom:75%;" />

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727090839325.png" alt="image-20200727090839325" style="zoom:80%;" />


