---
layout: post
title: 泊松表面重建算法
description: "Possion Reconstruction算法论文阅读（补充了涉及到的数学知识）"
modified: 2020-7-27
tags: [Geometric, Paper, Visualization]
math: true
image:
  feature: abstract-4.jpg
---
<details open><!-- 可选open -->
<summary>Contents</summary>
<div markdown="1">
<!-- TOC -->

   - [泊松重建 (Possion Reconstruction)](#01-泊松重建(PossionReconstruction))
      - [三维指示函数拟合](#011-三维指示函数拟合)
         - [隐函数拟合](#0111-隐函数拟合)
         - [求解方法](#0112-求解方法)
      - [等值面提取](#012-等值面提取)
               - [【附：相关数学知识】](#01201-【附：相关数学知识】)
                  - [泊松方程定义与求解（图像场景）](#012011-泊松方程定义与求解（图像场景）)
                  - [[矢量分析：梯度、散度、旋度](https://zhuanlan.zhihu.com/p/22654688)](#012012-[矢量分析：梯度、散度、旋度](https://zhuanlan.zhihu.com/p/22654688))
                  - [[向量的点乘（内积）、叉乘（外积）](https://www.cnblogs.com/gxcdream/p/7597865.html)](#012013-[向量的点乘（内积）、叉乘（外积）](https://www.cnblogs.com/gxcdream/p/7597865.html))

<!-- /TOC -->
</div>
</details>

<a id="toc_anchor" name="#01-泊松重建(PossionReconstruction)"></a>

## 泊松重建 (Possion Reconstruction)

[论文](http://hhoppe.com/poissonrecon.pdf)	[代码](https://github.com/mkazhdan/PoissonRecon)

> Reconstructing 3D surfaces from point samples

<a id="toc_anchor" name="#011-三维指示函数拟合"></a>

### 三维指示函数拟合

输入：有法向量的点集$$S$$（有向点云）

样本点 $$s\in{S}$$ 位于未知模型 $$M$$ 的表面 $$∂M$$ 附in近，且必须包含两个属性： $$s.p$$（坐标）和 $$s.\overrightarrow{N}$$（朝内的法向）。 

输出：指示函数（indicator function）$$\chi$$，用于确定表里

<!--more-->

<a id="toc_anchor" name="#0111-隐函数拟合"></a>

#### 隐函数拟合

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727091834595.png" alt="image-20200727091834595" style="zoom:80%;" />

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727092147103.png" alt="image-20200727092147103" style="zoom:80%;" />

> Key insight: The gradient of the indicator function is a vector field that is zero almost everywhere (since the indicator function is constant almost everywhere), except at points near the surface, where it is equal to the inward surface normal. 
>
> Thus, the oriented point samples can be viewed as samples of the gradient of the model’s indicator function.

<a id="toc_anchor" name="#0112-求解方法"></a>

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

<a id="toc_anchor" name="#012-等值面提取"></a>

### 等值面提取

2.1 计算等值面值（isovalue）

evaluating $$\widetilde\chi$$ at the sample positions and use the average of the values for isosurface extraction:

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727112136417.png" alt="image-20200727112136417" style="zoom:50%;" />

2.2 通过定义在八叉树上的Marching Cubes方法提取等值面

<a id="toc_anchor" name="#01201-【附：相关数学知识】"></a>

##### 【附：相关数学知识】

<a id="toc_anchor" name="#012011-泊松方程定义与求解（图像场景）"></a>

###### 泊松方程定义与求解（图像场景）

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727093446421.png" alt="image-20200727093446421" style="zoom:80%;" />

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727094105592.png" alt="image-20200727094105592" style="zoom:80%;" />

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727094800681.png" alt="image-20200727094800681" style="zoom:50%;center;" />

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727094908015.png" alt="image-20200727094908015" style="zoom:80%;" />

[泊松图像融合](https://zhuanlan.zhihu.com/p/68349210)

<a id="toc_anchor" name="#012012-[矢量分析：梯度、散度、旋度](https://zhuanlan.zhihu.com/p/22654688)"></a>

###### [矢量分析：梯度、散度、旋度](https://zhuanlan.zhihu.com/p/22654688)

向量场**A**，数量场u：

▽称为汉密尔顿算子； ▽·▽=△，△称为[拉普拉斯算子](https://blog.csdn.net/qq_30815237/article/details/86543091)。

【梯度】▽u	【散度】**▽**·**A**（点乘结果为数）

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727090328746.png" alt="image-20200727090328746" style="zoom:80%;" />

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727090354194.png" alt="image-20200727090354194" style="zoom:80%;" />

本意是一个向量（矢量），表示某一函数在该点处的方向导数沿着该方向取得最大值，即函数在该点处沿着该方向（此梯度的方向）变化最快，变化率最大（为该梯度的模）

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727090409484.png" alt="image-20200727090409484" style="zoom:80%;" />

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727090434092.png" alt="image-20200727090434092" style="zoom:80%;" />

<a id="toc_anchor" name="#012013-[向量的点乘（内积）、叉乘（外积）](https://www.cnblogs.com/gxcdream/p/7597865.html)"></a>

###### [向量的点乘（内积）、叉乘（外积）](https://www.cnblogs.com/gxcdream/p/7597865.html)

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727090716101.png" alt="image-20200727090716101" style="zoom:75%;" />

<img src="{{ site.url }}/images/2020-07-27-PossionRecon/image-20200727090839325.png" alt="image-20200727090839325" style="zoom:80%;" />


