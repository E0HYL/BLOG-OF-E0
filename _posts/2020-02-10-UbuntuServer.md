---
layout: post
title: Ubuntu GPU Server常用操作指南
description: "Commom operations for Ubuntu server."
modified: 2020-02-10
tags: [Skills, Linux]
image:
  feature: abstract-4.jpg
  entry: abstract-4.jpg
  credit: DarGadgetZ
  creditlink: http://www.dargadgetz.com/ios-7-abstract-wallpaper-pack-for-iphone-5-and-ipod-touch-retina/
---

# SSH神器：MobaXterm

> Free X server for Windows with tabbed SSH terminal, telnet, RDP, VNC and X11-forwarding  
MobaXterm是Windows下的一个远程连接客户端，功能十分强大，它不仅仅只支持ssh连接，它支持许多的远程连接方式，包括SSH，X11，RDP，VNC，FTP，MOSH

## 基本场景0: [SSH](https://zhuanlan.zhihu.com/p/56341917)
> 图形化的SSH隧道管理  
> SSH登陆后左边会自动列出sftp文件传输窗口，可以随手进行文件上传和下载  
> 免费的绿色免安装版本就可以满足日常工作的需求  

## 场景1: 通过跳板机的远程登录
### 需求介绍
客户端A使用校园网IP（10.开头）；<br>
校园网为路由器分配IP，路由器又为实验室的电脑分配IP。服务器B和服务器C在实验室内网中（IP为192.168.1.x）。<br>
其中服务器B做过端口转发，可以通过校园网访问。但客户端访问服务器C则需要通过B来做跳板机。
### 配置[教程](https://blog.csdn.net/xuyuqingfeng953/article/details/96180642)
Session：SSH：Network Setting：Connect through SSH gateway

## 场景2: 本地浏览器访问远程端口
### 需求介绍
远端服务器无浏览器界面，但希望使用jupyter notebook、tensorboard等。
### 配置[教程](http://m.blog.sina.com.cn/s/blog_e11ac5850102xurp.html)
Tools：Network：MobaSSHTunnel

# Linux常用命令
## tar: tape archive
### 主要选项
> **-c, --create 打包**  
> **-x, --extract 解压**  
> -t,  --list 查看  
> -A, --catenate, --concatenate 合并tar  
> -r, --append 向归档文件末尾追加  
> -u, --update 仅追加比归档中副本更新的文件  
> --delete 从归档中删除  
> -d, --diff 比较归档和文件系统的差异  

### 辅助选项
> **-v, --verbose 详细列出处理的文件**   
> **-f, --file=`ARCHIVE`**   
> -C 解压到指定目录  

### 压缩选项 (not support compress directly)
> -a, --auto-compress 根据归档后缀名决定  
> -j, --bzip2 **【.tar.bz, .tar.bz2】**  
> -J, --xz **【.tar.xz】**  
> -z, --gzip **【.tar.gz, .tgz】**  
> -Z, --compress **【.tar.Z】**  
![](Linux%E5%B8%B8%E7%94%A8%E5%91%BD%E4%BB%A4/%E7%85%A7%E7%89%87%202020%E5%B9%B42%E6%9C%889%E6%97%A5%20%E4%B8%8B%E5%8D%8822649.jpg)

* 打包（压缩）
```shell
$ tar -cvf /source_path/collection.tar file1, file2
# -z 压缩为.tar.gz文件
$ tar -zcvf collection.tar.gz
```
* 解压
```sh
$ tar -xvf collection.tar
# -C 指定输出路径
$ tar -zxvf collection.tar.gz -C /destination_path/
```
* 更新（假设修改了file1，新创建了file3）
```shell
$ tar --update -v -f collection.tar file1, file2, file3
file1
file3
```

### 附：（其它）压缩 / 解压
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

## 统计目录下 文件/目录 个数
```shell
$ ll dir_name|grep ^-|wc -l
$ ll dir_name|grep ^d|wc -l
```
注：ll的输出信息中，目录以d开头，文件以-开头。若统计所有条目则无需中间一项。

## 实时监测命令的运行结果
-n, --interval=	周期（秒）
-d, --differences	高亮显示变动
```shell
$ watch -n 1 -d nvidia-smi
```
注：nvidia-smi为CUDA命令，用于查看GPU使用情况

## kill进程
```shell
# 杀死指定用户的所有进程
$ kill -u user
# 批量杀死包含某字符串（string）的进程：awk中的脚本打印文本每行的第二列，即进程号；xargs将之前获得的进程号作为kill -9的参数并执行
$ ps -ef|grep string|grep -v grep|awk ‘{print $2}’|xargs kill -9
```

## 系统管理
### sudo: superuser do
### su: switch user
### ps: process status
常用参数：-aux -ef
五种进程状态
1. R：运行 runnable
2. S：中断 sleeping
3. D：不可中断 uninterruptible sleep (usually IO)
4. Z: 僵死 a zombie process
5. T: 停止 traced or stopped

### nohup: no hang up
常和 & 符号配合，使程序在后台永久执行【用**screen**更优】
### screen
`screen -ls`
<br>`screen -r id `  **ctrl+a+d**：返回之前的shell
### reboot
重启（需root）

## 文档编辑
### grep: global regular expression print
-v 显示不包含匹配文本的所有行
查找文件里符合条件的字符串，输出为文件中包含该字符串的行
`grep string *file` 查找后缀有file的文件中包含string字符串的文件
### wc: word count
计算文件字数，输出三个数字分别表示行数(-l)、单词数(-w)、字节数(-c, --chars)

## 文件管理
### chown: change owner
改变文件所有者，-R指定目录以及其子目录下所有文件
<br>`chown user[:group] file_name`
<br>`chmod -R user[:group] *`（当前目录）
### chgrp: change group
改变文件用户组
### chmod: change mode
`chmod 777 file_name`
* 读r=4, 写w=2, 执行x=1【rwx : 7, rw- : 6, r-x : 5】
* 三个数字分别对应User, Group, Other的权限		
### cat: concatenate
连接文件并打印到标准输出设备上，-n由1开始对输出行编号（-b空白行不编）
<br>`cat file1 file2 > file3` （输入到file3中，若>>则为追加，不打印在控制台）
### more / less
分页浏览文件（less可随意浏览）
<br>`history | less` 查看命令使用历史并通过less分页显示（Q退出）
### ln: link
为某文件在另一个位置建立同步链接。需要在不同目录用到同一文件时，不必重复占用磁盘空间。-s 创建软链接（可跨文件系统，类似于快捷方式）
<br>`ln -s file_name link_name`
### cp: copy
复制目录时必须加**-r**
### rm: remove
删除目录必须加-r，-f对只读文件也直接删除
### mv: move
`mv source dest`
### scp: secure copy
linux系统下基于ssh登录进行安全的远程文件拷贝
<br>`scp local_file remote_username@remote_ip:remote:folder`
### locate
用于查找符合条件的文档（文件或目录名中包含指定字符串）
### whereis
只查找二进制文件、源代码文件或帮助文件。一般文件的定位用locate
### which
在环境变量$PATH$设置的目录中查找文件
### split
将大文件分割成较小的文件（默认间隔1000行，可用-<行数>指定）

## 磁盘管理
### cd: change directory
### pwd: print work directory
打印当前工作目录的绝对路径 
### ls: list
列出指定目录下的内容
-a 显示隐藏文件
-l 除文件名外，还有文件权限、所有者、大小、修改时间 （简写为**ll**）
### df: disk free
显示文件系统的磁盘使用情况
### du: disk usage
显示指定文件所占的磁盘空间
### mkdir

## 网络通讯
### netstat
`netstat -apu` 显示UDP（-u）端口号的使用，若TCP则为-t
<br>`netstat -l` 显示监听的套接字
### tcpdump
显示TCP包信息
### ifconfig
显示或设置网络设备

## 常用符号
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
## 正则表达式
### 限定符
```
*	>=0个匹配
+	>=1个匹配
?	0或1个匹配
{n}	指定数目n的匹配
{n, }	不少于指定数目n
{n, m}	匹配范围（m<=255）
```
### 定位符
```
^	文本的开始
$	文本的结尾
\b	单词边界
\B	非单词边界
```
### 非打印字符
```
\f	换页符
\n	换行符
\r	回车符
\t	制表符
\v	垂直制表符
\s	任何空白字符（空格及以上五个），等价于 [ \f\n\r\t\v]。
	注意Unicode正则表达式会匹配全角空格字符。
```
### 其它特殊字符
```
\	转义字符
.	除换行符（\n）外的任何单字符
|	两项之间的选择
( )	一个子表达式的开始和结束
```
### 字符簇
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
