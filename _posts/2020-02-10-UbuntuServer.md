---
layout: post
title: Ubuntu-GPU-Server常用操作指南
description: "Commom operations for Ubuntu server."
modified: 2020-6-22
tags: Skills
image:
  feature: abstract-4.jpg
---

<details open><!-- 可选open -->
<summary>Contents</summary>
<div markdown="1">
<!-- TOC -->

- [SSH神器：MobaXterm](#1-SSH神器：MobaXterm)
    - [基本场景0: [SSH](https://zhuanlan.zhihu.com/p/56341917)](#11-基本场景0:[SSH](https://zhuanlan.zhihu.com/p/56341917))
    - [场景1: 通过跳板机的远程登录](#12-场景1:通过跳板机的远程登录)
        - [需求介绍](#121-需求介绍)
        - [配置[教程](https://blog.csdn.net/xuyuqingfeng953/article/details/96180642)](#122-配置[教程](https://blog.csdn.net/xuyuqingfeng953/article/details/96180642))
    - [场景2: 本地浏览器访问远程端口](#13-场景2:本地浏览器访问远程端口)
        - [需求介绍](#131-需求介绍)
        - [配置[教程](http://m.blog.sina.com.cn/s/blog_e11ac5850102xurp.html)](#132-配置[教程](http://m.blog.sina.com.cn/s/blog_e11ac5850102xurp.html))
- [更新CUDA版本](#2-更新CUDA版本)
- [VSCode: 编辑远程文件](#3-VSCode:编辑远程文件)
- [Anaconda3安装OpenCV](#4-Anaconda3安装OpenCV)
- [Linux常用命令](#5-Linux常用命令)
    - [执行`.sh`文件](#51-执行`.sh`文件)
    - [tar: tape archive](#52-tar:tapearchive)
        - [主要选项](#521-主要选项)
        - [辅助选项](#522-辅助选项)
        - [压缩选项 (not support compress directly)](#523-压缩选项(notsupportcompressdirectly))
        - [附：（其它）压缩 / 解压](#524-附：（其它）压缩/解压)
    - [统计目录下 文件/目录 个数](#53-统计目录下文件/目录个数)
    - [实时监测命令的运行结果](#54-实时监测命令的运行结果)
    - [kill进程](#55-kill进程)
    - [修改环境变量](#56-修改环境变量)
    - [查看系统版本](#57-查看系统版本)
    - [系统管理](#58-系统管理)
        - [sudo: superuser do](#581-sudo:superuserdo)
        - [su: switch user](#582-su:switchuser)
        - [用户与用户组](#583-用户与用户组)
        - [ps: process status](#584-ps:processstatus)
        - [nohup: no hang up](#585-nohup:nohangup)
        - [screen](#586-screen)
        - [reboot](#587-reboot)
    - [文档编辑](#59-文档编辑)
        - [grep: global regular expression print](#591-grep:globalregularexpressionprint)
        - [wc: word count](#592-wc:wordcount)
        - [[Vim](https://www.openvim.com/)编辑器](#593-[Vim](https://www.openvim.com/)编辑器)
    - [文件管理](#510-文件管理)
        - [chown: change owner](#5101-chown:changeowner)
        - [chgrp: change group](#5102-chgrp:changegroup)
        - [chmod: change mode](#5103-chmod:changemode)
        - [cat: concatenate](#5104-cat:concatenate)
        - [more / less](#5105-more/less)
        - [ln: link](#5106-ln:link)
        - [cp: copy](#5107-cp:copy)
        - [rm: remove](#5108-rm:remove)
        - [mv: move](#5109-mv:move)
        - [scp: secure copy](#51010-scp:securecopy)
        - [wget & curl](#51011-wget&curl)
        - [locate](#51012-locate)
        - [whereis](#51013-whereis)
        - [which](#51014-which)
        - [split](#51015-split)
    - [磁盘管理](#511-磁盘管理)
        - [cd: change directory](#5111-cd:changedirectory)
        - [pwd: print work directory](#5112-pwd:printworkdirectory)
        - [ls: list](#5113-ls:list)
        - [df: disk free](#5114-df:diskfree)
        - [du: disk usage](#5115-du:diskusage)
        - [mkdir](#5116-mkdir)
    - [网络通讯](#512-网络通讯)
        - [netstat](#5121-netstat)
        - [tcpdump](#5122-tcpdump)
        - [ifconfig](#5123-ifconfig)
    - [常用符号](#513-常用符号)
    - [正则表达式](#514-正则表达式)
        - [限定符](#5141-限定符)
        - [定位符](#5142-定位符)
        - [非打印字符](#5143-非打印字符)
        - [其它特殊字符](#5144-其它特殊字符)
        - [字符簇](#5145-字符簇)

<!-- /TOC -->
</div>
</details>

<!--more-->

<a id="toc_anchor" name="#1-SSH神器：MobaXterm"></a>

# 1. SSH神器：MobaXterm

> Free X server for Windows with tabbed SSH terminal, telnet, RDP, VNC and X11-forwarding 
>
> MobaXterm是Windows下的一个远程连接客户端，功能十分强大，它不仅仅只支持ssh连接，它支持许多的远程连接方式，包括SSH，X11，RDP，VNC，FTP，MOSH

<a id="toc_anchor" name="#11-基本场景0:[SSH](https://zhuanlan.zhihu.com/p/56341917)"></a>

## 1.1. 基本场景0: [SSH](https://zhuanlan.zhihu.com/p/56341917)

> 图形化的SSH隧道管理 
>
> SSH登陆后左边会自动列出sftp文件传输窗口，可以随手进行文件上传和下载 
>
> 免费的绿色免安装版本就可以满足日常工作的需求  

<a id="toc_anchor" name="#12-场景1:通过跳板机的远程登录"></a>

## 1.2. 场景1: 通过跳板机的远程登录

<a id="toc_anchor" name="#121-需求介绍"></a>

### 1.2.1. 需求介绍

客户端A使用校园网IP（10.开头）；

校园网为路由器分配IP，路由器又为实验室的电脑分配IP。服务器B和服务器C在实验室内网中（IP为192.168.1.x）。

其中服务器B做过端口转发，可以通过校园网访问。但客户端访问服务器C则需要通过B来做跳板机。

<a id="toc_anchor" name="#122-配置[教程](https://blog.csdn.net/xuyuqingfeng953/article/details/96180642)"></a>

### 1.2.2. 配置[教程](https://blog.csdn.net/xuyuqingfeng953/article/details/96180642)

```
Session：SSH：Network Setting：Connect through SSH gateway
```

<a id="toc_anchor" name="#13-场景2:本地浏览器访问远程端口"></a>

## 1.3. 场景2: 本地浏览器访问远程端口

<a id="toc_anchor" name="#131-需求介绍"></a>

### 1.3.1. 需求介绍
远端服务器无浏览器界面，但希望使用jupyter notebook、tensorboard等。

<a id="toc_anchor" name="#132-配置[教程](http://m.blog.sina.com.cn/s/blog_e11ac5850102xurp.html)"></a>

### 1.3.2. 配置[教程](http://m.blog.sina.com.cn/s/blog_e11ac5850102xurp.html)

```
Tools：Network：MobaSSHTunnel
```
稍复杂一些，其后的详细参数配置可戳小标题里的链接。

<a id="toc_anchor" name="#2-更新CUDA版本"></a>

# 2. 更新CUDA版本

这里以卸载10.0，升级到10.1为例。

```shell
$ cd /usr/local/cuda-8.0/bin # uninstall
$ sudo ./uninstall_cuda_10.0.pl
```

新版10.1驱动下载：[地址](https://developer.nvidia.com/cuda-10.1-download-archive-base)（根据自己的系统版本选择安装包）

```shell
$ wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.105_418.39_linux.run
$ sudo sh "/usr/cuda_10.1.105_418.39_linux.run" # exit X server first: sudo service lightdm stop
```

安装时第一个驱动不用选，因为之前就有（[参考](https://zhuanlan.zhihu.com/p/72298520)）。不过要确保该版本显卡驱动支持CUDA的Toolkit，若出错，附单独的驱动安装方法：

```shell
$ sudo add-apt-repository ppa:graphics-drivers/ppa  # 添加ppa源
$ sudo apt-get update
$ ubuntu-drivers devices # 查看可安装的驱动版本
$ sudo apt install nvidia-430 # install (430)
```

<a id="toc_anchor" name="#3-VSCode:编辑远程文件"></a>

# 3. VSCode: 编辑远程文件

- 安装Remote-SSH插件

- 添加配置文件，通常位置为`~/.ssh/config`

- config文件中配置SSH参数，详见官方文档：https://code.visualstudio.com/docs/remote/ssh

  ```
  Host 连接的主机的名称，可自定
  Hostname 远程主机的IP地址
  User 用于登录远程主机的用户名
  Port 用于登录远程主机的端口
  IdentityFile 本地的私钥的路径
  ForwardX11
  ProxyCommand
  ```

  解决`CreateProcessW failed error:2`：指定ProxyCommand中ssh.exe的的完整路径，如`ProxyCommand C:\Program Files\Git\usr\bin\ssh.exe ...`

- 关闭连接：File - Close Folder

<a id="toc_anchor" name="#4-Anaconda3安装OpenCV"></a>

# 4. Anaconda3安装OpenCV

1. 安装依赖项

  ```shell
  $ sudo apt-get install build-essential
  $ sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
  $ sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev # 处理图像所需的包
  $ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev liblapacke-dev
  $ sudo apt-get install libxvidcore-dev libx264-dev # 处理视频所需的包
  $ sudo apt-get install libatlas-base-dev gfortran # 优化opencv功能
  $ sudo apt-get install ffmpeg
  $ sudo apt-get install libjasper-dev
  ```

2. 从[官网](https://anaconda.org/menpo/opencv3/files)下载所需的包

3. conda命令执行`conda install opencv3-3.1.0-py36_0.tar.bz2`

<a id="toc_anchor" name="#5-Linux常用命令"></a>

# 5. Linux常用命令

<a id="toc_anchor" name="#51-执行`.sh`文件"></a>

## 5.1. 执行`.sh`文件

`.sh`文件就是文本文件，如果要执行，需要先使用`chmod a+x filename.sh`来给可执行权限；

然后在其所在目录下执行`sudo ./filename.sh`

<a id="toc_anchor" name="#52-tar:tapearchive"></a>

## 5.2. tar: tape archive

<a id="toc_anchor" name="#521-主要选项"></a>

### 5.2.1. 主要选项

> -c, --create 打包
>
> -x, --extract 解压
>
> -t,  --list 查看
>
> -A, --catenate, --concatenate 合并tar
>
> -r, --append 向归档文件末尾追加
>
> -u, --update 仅追加比归档中副本更新的文件
>
> --delete 从归档中删除
>
> -d, --diff 比较归档和文件系统的差异

<a id="toc_anchor" name="#522-辅助选项"></a>

### 5.2.2. 辅助选项

> -v, --verbose 详细列出处理的文件
>
> -f, --file=`ARCHIVE`
>
> -C 解压到指定目录

<a id="toc_anchor" name="#523-压缩选项(notsupportcompressdirectly)"></a>

### 5.2.3. 压缩选项 (not support compress directly)

> -a, --auto-compress 根据归档后缀名决定
>
> -j, --bzip2 **【.tar.bz, .tar.bz2】**
>
> -J, --xz **【.tar.xz】**  
>
> -z, --gzip **【.tar.gz, .tgz】**
>
> -Z, --compress **【.tar.Z】**

<figure><img src="{{site.url}}/images/fullsizerender.jpg" style="zoom:50%;"></figure>

* 打包（压缩）

```shell
$ tar -cvf /source_path/collection.tar file1, file2 
$ tar -zcvf collection.tar.gz # -z 压缩为.tar.gz文件
```

* 解压

```shell
$ tar -xvf collection.tar
$ tar -zxvf collection.tar.gz -C /destination_path/ # -C 指定输出路径
```

* 更新（假设修改了file1，新创建了file3）

```shell
$ tar --update -v -f collection.tar file1, file2, file3
file1
file3
```

<a id="toc_anchor" name="#524-附：（其它）压缩/解压"></a>

### 5.2.4. 附：（其它）压缩 / 解压

* .gz

  `gzip file_name`

  `gunzip file_name.gz`

* .bz2, .bz

  `bzip2 file_name`

  `bunzip2 file_name.bz2`

* .Z

  `compress file_name`

  `uncompress file_name.Z`

* .zip

  `zip file_name.zip dir_name`

  `unzip file_name.zip`

* .rar

  `rar a file_name.rar dir_name`

  `rar x file_name.rar`

<a id="toc_anchor" name="#53-统计目录下文件/目录个数"></a>

## 5.3. 统计目录下 文件/目录 个数

```shell
$ ll -R dir_name|grep ^-|wc -l
$ ll -R dir_name|grep ^d|wc -l
```

注：ll的输出信息中，目录以d开头，文件以-开头。若统计所有条目则无需中间一项。

<a id="toc_anchor" name="#54-实时监测命令的运行结果"></a>

## 5.4. 实时监测命令的运行结果

-n, --interval=	周期（秒）

-d, --differences	高亮显示变动

```shell
$ watch -n 1 -d nvidia-smi
```

注：nvidia-smi为CUDA命令，用于查看GPU使用情况

<a id="toc_anchor" name="#55-kill进程"></a>

## 5.5. kill进程

```shell
$ kill -u user # 杀死指定用户的所有进程
$ ps -ef|grep string|grep -v grep|awk ‘{print $2}’|xargs kill -9 # 批量杀死包含某字符串（string）的进程：awk中的脚本打印文本每行的第二列，即进程号；xargs将之前获得的进程号作为kill -9的参数并执行
```

<a id="toc_anchor" name="#56-修改环境变量"></a>

## 5.6. 修改环境变量

修改用户级别的环境变量`vim ~/.bashrc`（系统级别`vim /etc/profile`），写入：

```bash
export PATH="$PATH:/home/username/example"
```

用`source ~/.bashrc`命令以生效

<a id="toc_anchor" name="#57-查看系统版本"></a>

## 5.7. 查看系统版本

```shell
$ cat /etc/issue
```

<a id="toc_anchor" name="#58-系统管理"></a>

## 5.8. 系统管理

<a id="toc_anchor" name="#581-sudo:superuserdo"></a>

### 5.8.1. sudo: superuser do

<a id="toc_anchor" name="#582-su:switchuser"></a>

### 5.8.2. su: switch user

<a id="toc_anchor" name="#583-用户与用户组"></a>

### 5.8.3. 用户与用户组

- 增加用户 `useradd -d /usr/username -m username`

- 为用户增加密码 `passwd username`

- 新建工作组 `groupadd groupname`

- 将用户添加进工作组 `usermod -G groupname username`

  直接用`usermod -G groupA`会离开其他用户组，仅仅做为这个用户组 groupA 的成员。应该加上 -a 选项： 

  ```shell
  $ usermod -a -G groupA user
  ```

- 删除用户 `userdel username`

<a id="toc_anchor" name="#584-ps:processstatus"></a>

### 5.8.4. ps: process status

常用参数：-aux -ef

五种进程状态：

1. R：运行 runnable

2. S：中断 sleeping

3. D：不可中断 uninterruptible sleep (usually IO)

4. Z: 僵死 a zombie process

5. T: 停止 traced or stopped

<a id="toc_anchor" name="#585-nohup:nohangup"></a>

### 5.8.5. nohup: no hang up

常和 & 符号配合，使程序在后台永久执行【用**screen**更优】

<a id="toc_anchor" name="#586-screen"></a>

### 5.8.6. screen

`screen -ls`

`screen -r id `  **ctrl+a+d**：返回之前的shell

> screen -ls | grep Detached | cut -d. -f1 | awk '{print $1}' | xargs kill	# kill all screen

<a id="toc_anchor" name="#587-reboot"></a>

### 5.8.7. reboot

重启（需root）

<a id="toc_anchor" name="#59-文档编辑"></a>

## 5.9. 文档编辑

<a id="toc_anchor" name="#591-grep:globalregularexpressionprint"></a>

### 5.9.1. grep: global regular expression print

-v 显示不包含匹配文本的所有行

查找文件里符合条件的字符串，输出为文件中包含该字符串的行

`grep string *file` 查找后缀有file的文件中包含string字符串的文件

<a id="toc_anchor" name="#592-wc:wordcount"></a>

### 5.9.2. wc: word count

计算文件字数，输出三个数字分别表示行数(-l)、单词数(-w)、字节数(-c, --chars)

<a id="toc_anchor" name="#593-[Vim](https://www.openvim.com/)编辑器"></a>

### 5.9.3. [Vim](https://www.openvim.com/)编辑器

三种模式：Insert, Normal, (Visual便于选取文本)

Normal模式下：

> 命令前添加数字表示多遍重复操作；`f`表示行内正向查找，`F`反向

- 查找字符串：按`/`，默认大小写敏感；正则匹配使用`?pattern`

  ```
  n：查找下一个匹配
  N：查找上一个匹配
  2n：查找下面第二个匹配
  ```

- 移动光标

  ```
  0：到行首
  ^：到行首第一个字符，如果前面有空格的话
  $：到行尾
  gg：快速到文件头
  G：快速到文件尾
  20j：向下移动 20 行
  50G：跳转到第 50 行
  ```

- 复制y，粘贴p，剪切x，删除d

  ```
  yy：复制一行
  8yy：向下复制8行
  yw：复制光标开始的一个单词
  y$：复制光标到行尾
  yfA：复制光标到第一个大写A中间的内容
  y2fA：复制光标到第二个大写A中间的内容
  ```

[Vim快捷键键位图](https://cloud.tencent.com/developer/article/1369567)

- 经典版（翻译）

<figure><img src="https://www.runoob.com/wp-content/uploads/2015/10/vi-vim-cheat-sheet-sch1.gif" alt="img" style="zoom: 67%;" /></figure>

- 进阶版

<figure><img src="http://michael.peopleofhonoronly.com/vim/vim_cheat_sheet_for_programmers_print.png" alt="img"  /></figure>

<a id="toc_anchor" name="#510-文件管理"></a>

## 5.10. 文件管理

<a id="toc_anchor" name="#5101-chown:changeowner"></a>

### 5.10.1. chown: change owner

改变文件所有者，-R指定目录以及其子目录下所有文件

`chown user[:group] file_name`

`chmod -R user[:group] *`（当前目录）

<a id="toc_anchor" name="#5102-chgrp:changegroup"></a>

### 5.10.2. chgrp: change group

改变文件用户组

<a id="toc_anchor" name="#5103-chmod:changemode"></a>

### 5.10.3. chmod: change mode

`chmod 777 file_name`
* 读r=4, 写w=2, 执行x=1【rwx : 7, rw- : 6, r-x : 5】
* 三个数字分别对应User, Group, Other的权限	

<a id="toc_anchor" name="#5104-cat:concatenate"></a>

### 5.10.4. cat: concatenate

连接文件并打印到标准输出设备上，-n由1开始对输出行编号（-b空白行不编）

`cat file1 file2 > file3` （输入到file3中，若>>则为追加，不打印在控制台）

<a id="toc_anchor" name="#5105-more/less"></a>

### 5.10.5. more / less

分页浏览文件（less可随意浏览）

`history | less` 查看命令使用历史并通过less分页显示（Q退出）

<a id="toc_anchor" name="#5106-ln:link"></a>

### 5.10.6. ln: link

为某文件在另一个位置建立同步链接。需要在不同目录用到同一文件时，不必重复占用磁盘空间。-s 创建软链接（可跨文件系统，类似于快捷方式）

`ln -s file_name link_name`

<a id="toc_anchor" name="#5107-cp:copy"></a>

### 5.10.7. cp: copy

复制目录时必须加**-r**

<a id="toc_anchor" name="#5108-rm:remove"></a>

### 5.10.8. rm: remove

删除目录必须加-r，-f对只读文件也直接删除

<a id="toc_anchor" name="#5109-mv:move"></a>

### 5.10.9. mv: move

`mv source dest`

<a id="toc_anchor" name="#51010-scp:securecopy"></a>

### 5.10.10. scp: secure copy

linux系统下基于ssh登录进行安全的远程文件拷贝

`scp local_file remote_username@remote_ip:remote:folder`

<a id="toc_anchor" name="#51011-wget&curl"></a>

### 5.10.11. wget & curl

1. 下载文件

```shell
$ curl -O http://man.linuxde.net/text.iso                    #O大写，不用O只是打印内容不会下载
$ wget http://www.linuxde.net/text.iso                       #不用参数，直接下载文件
```

2. 下载文件并重命名

```shell
$ curl -o rename.iso http://man.linuxde.net/text.iso         #o小写
$ wget -O rename.zip http://www.linuxde.net/text.iso         #O大写
```

3. 断点续传

```shell
$ curl -O -C -URL http://man.linuxde.net/text.iso            #C大写
$ wget -c http://www.linuxde.net/text.iso                    #c小写
```

4. 显示响应头部信息

```shell
$ curl -I http://man.linuxde.net/text.iso
$ wget --server-response http://www.linuxde.net/test.iso
```

5. wget打包下载网站

```shell
$ wget --mirror -p --convert-links -P /var/www/html http://man.linuxde.net/
```

<a id="toc_anchor" name="#51012-locate"></a>

### 5.10.12. locate

用于查找符合条件的文档（文件或目录名中包含指定字符串）

<a id="toc_anchor" name="#51013-whereis"></a>

### 5.10.13. whereis

只查找二进制文件、源代码文件或帮助文件。一般文件的定位用locate

<a id="toc_anchor" name="#51014-which"></a>

### 5.10.14. which

在环境变量$PATH$设置的目录中查找文件

<a id="toc_anchor" name="#51015-split"></a>

### 5.10.15. split

将大文件分割成较小的文件（默认间隔1000行，可用-<行数>指定）

<a id="toc_anchor" name="#511-磁盘管理"></a>

## 5.11. 磁盘管理

<a id="toc_anchor" name="#5111-cd:changedirectory"></a>

### 5.11.1. cd: change directory

<a id="toc_anchor" name="#5112-pwd:printworkdirectory"></a>

### 5.11.2. pwd: print work directory

打印当前工作目录的绝对路径 

<a id="toc_anchor" name="#5113-ls:list"></a>

### 5.11.3. ls: list

列出指定目录下的内容

-a 显示隐藏文件

-l 除文件名外，还有文件权限、所有者、大小、修改时间 （简写为**ll**）

-R 将目录下所有的子目录的文件都列出来（递归）

<a id="toc_anchor" name="#5114-df:diskfree"></a>

### 5.11.4. df: disk free

显示文件系统的磁盘使用情况

<a id="toc_anchor" name="#5115-du:diskusage"></a>

### 5.11.5. du: disk usage

显示指定文件所占的磁盘空间

<a id="toc_anchor" name="#5116-mkdir"></a>

### 5.11.6. mkdir

<a id="toc_anchor" name="#512-网络通讯"></a>

## 5.12. 网络通讯

<a id="toc_anchor" name="#5121-netstat"></a>

### 5.12.1. netstat

`netstat -apu` 显示UDP（-u）端口号的使用，若TCP则为-t

`netstat -l` 显示监听的套接字

<a id="toc_anchor" name="#5122-tcpdump"></a>

### 5.12.2. tcpdump

显示TCP包信息

<a id="toc_anchor" name="#5123-ifconfig"></a>

### 5.12.3. ifconfig

显示或设置网络设备

<a id="toc_anchor" name="#513-常用符号"></a>

## 5.13. 常用符号

```
|	（管道，pipeline）上一条命令的输出，作为下一条命令的参数
	`echo ‘yes’ | wc -l`
|| 	上一条命令执行失败后，才执行下一条命令
	`cat nofile || echo “fail”`
&&	上一条命令执行成功时，才执行下一条命令
&	任务在后台执行
~ 	家目录（root用户为root，普通用户为home）
.	当前目录
..	上一级目录
>	覆盖重写某个文件
>> 	追加到某个文件
;	连续命令间的分隔
```

<a id="toc_anchor" name="#514-正则表达式"></a>

## 5.14. 正则表达式

<a id="toc_anchor" name="#5141-限定符"></a>

### 5.14.1. 限定符

```
*	>=0个匹配
+	>=1个匹配
?	0或1个匹配
{n}	指定数目n的匹配
{n, }	不少于指定数目n
{n, m}	匹配范围（m<=255）
```

<a id="toc_anchor" name="#5142-定位符"></a>

### 5.14.2. 定位符

```
^	文本的开始
$	文本的结尾
\b	单词边界
\B	非单词边界
```

<a id="toc_anchor" name="#5143-非打印字符"></a>

### 5.14.3. 非打印字符

```
\f	换页符
\n	换行符
\r	回车符
\t	制表符
\v	垂直制表符
\s	任何空白字符（空格及以上五个），等价于 [ \f\n\r\t\v]。
	注意Unicode正则表达式会匹配全角空格字符。
```

<a id="toc_anchor" name="#5144-其它特殊字符"></a>

### 5.14.4. 其它特殊字符

```
\	转义字符
.	除换行符（\n）外的任何单字符
|	两项之间的选择
( )	一个子表达式的开始和结束
```

<a id="toc_anchor" name="#5145-字符簇"></a>

### 5.14.5. 字符簇

放在一个方括号（[ ]）里，连字符（-）可表示一个范围。方括号里的^符号表示“非”。
```
[a-z]	所有小写字母
[A-Z]	所有大写字母
[a-zA-Z]	所有字母
[0-9]	所有数字
[0-9\.\-]	所有数字、句号和减号
[^a-z]	所有非小写字母的字符
^[a-zA-Z0-9_]+	所有包含一个及以上字母、数字或下划线的字符串
^[1-9][0-9]*	所有正整数
^\-?[0-9]+	所有整数
^[-]?[0-9]+(\.[0-9]+)?	所有浮点数
```
