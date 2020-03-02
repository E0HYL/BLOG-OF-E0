---
layout: post
title: More Than Digital Image Processing
description: "A bin for ImageFingerprint collections."
modified: 2018-12-1
tags: Camera Fingerprint
image:
  feature: abstract-3.jpg
---

### The general structure and sequence of stages in the camera pipeline

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S1742287617303146-gr1.jpg">

<!--more-->

*以下引自论文 [Blind source camera identification](https://ieeexplore.ieee.org/document/1418853)*
<div markdown="1" class="largebq">
> After light enters the camera through the lens, a set of filters are employed, the most important being an anti-aliasing filter. <br>
    The CCD detector is the main component of a digital camera. The detector measures the intensity of light at each pixel location on the detectors surface. In the ideal case, a separate CCD would be used for each of the three color (RGB) channels, but then the manufacturing cost would be quite high. A common approach is to use only a single CCD detector at every pixel, but partition it's surface with different spectral filters. Such filters are called Color Filter Arrays or CFA.<br>
    Looking at the RGB values in the CFA pattern it is evident that the missing RGB values need to be interpolated for each pixel. There are a number of different interpolation algorithms which could be used and different manufacturers use different interpolation techniques.<br> 
    After color decomposition is done by CFA, a detector is used to obtain a digital representation of light intensity in each color band. Next a number of operations are done by the camera, which include color interpolation as explained before, gamma correction, color processing, white point correction, and last but not least compression. 
</div>
    Although the operations and stages explained in this section are standard stages in a digital camera pipeline, the exact processing detail in each stage varies from one manufacturer to the other, and even in different camera models manufactured by the same company.
    
- CFA & Color Interpolation 
  - [为什么数码相机可以拍出彩色照片？](http://www.ruanyifeng.com/blog/2012/12/bayer_filter.html)
  - [图像bayer格式介绍以及bayer插值原理CFA](https://blog.csdn.net/u011776903/article/details/78437809)
  
### Camera Identification
  
