---
layout: post
title: 阿里云服务器SSH登录及云盘挂载
description: "Instructions for Alibaba Cloud."
modified: 2020-2-13
tags: [Skills]
image:
  feature: abstract-5.jpg
---
# 网页远程登录

1. VNC远程连接，输入**远程连接密码**

<img src="{{site.url}}/images/weblogin.png" alt="">

2. 显示命令行后，如果是全黑屏就点击左上角任意发送一个命令

3. 控制台输入用户名（默认root）和密码（**实例的密码**）

- 注意两个密码不一样，忘记了就重置/修改吧，但改完后一定要重启！

<img src="{{site.url}}/images/passwd.png" alt="">

# 配置SSH密钥

## Server

作为一个学网络安全的，还要做两件事：修改SSH端口，不用默认的22；禁止单纯的密码认证。

1. 修改sshd_config文件：`vim /etc/ssh/sshd_config`

```yaml
Port 自定义端口号
PasswordAuthentication yes 允许密码认证
```

2. 重启ssh服务：`/etc/init.d/ssh restart`
3. 为开放的SSH端口添加入方向的规则（删除原有的端口22规则，好像买服务器时候可选SSH端口，但当时不是我买的= =）

<img src="{{site.url}}/images/rule.PNG" alt="">

<img src="{{site.url}}/images/rule1.PNGg" alt="">

## Client

假设`~/.ssh`下有公私钥对（id_rsa.pub，id_rsa）。没有的话先生成一下，Win下用git。

1. `cd ~/.ssh`

2. `ssh-copy-id id_rsa.pub -p 端口号 用户名(root)@ip地址`

   中间会让输入一次密码（**实例的密码**），显示added则成功。

   <img src="{{site.url}}/images/clientkey.png" alt="">
   
   > 打开**Server**的.ssh文件夹下的**authorized_keys**文件，可以看到公钥的内容。所以其实也可以**直接把id_rsa.pub的内容复制进来**，添加新用户的key只要换行再粘贴就行。

# 挂载云盘

用`df -l`查看磁盘大小，不符合买之前提的要求。原来是因为云盘要自己挂载。

1. 查看未挂载的盘，我这里是`/dev/vdb`

   ```shell
   fdisk -l
   ```

2. 将`/dev/vdc`挂载到`/data`目录中

   - 先创建目录`mkdir -p /data `
   - 硬盘分区`fdisk /dev/vdc`(依次输入`n` -> `p` -> `Enter` -> `Enter` -> `wq`)

3. `fdisk -l` ：查看磁盘的设备，我这里是`/dev/vdb1`

4. 格式化 `/dev/vdb1` 格式为ext4

   ```shell
   mkfs.ext4 /dev/vdb1
   ```

5. 挂载云盘到目录

   ```shell
   mount /dev/vdb1 /data
   ```

   <img src="{{site.url}}/images/mount.PNG" alt="">

## 大于2 TiB的数据盘

```
apt-get install parted
apt-get install e2fsprogs
```

1. 查看未挂载的盘，我这里是`/dev/vdc`

2. `parted /dev/vdd`开始分区
    - `mklabel gpt`:将默认的MBR分区格式转为GPT分区格式
    - `mkpart primary 1 100%`:划分一个主分区，并设置分区的开始位置和结束位置
    - `align-check optimal 1`:检查分区是否对齐。
    - `print`:查看分区表
    - `quit`:退出parted操作

3. `partprobe`:使系统重读分区表

4. `mkfs -t ext4 /dev/vdd1`:为`/dev/vdc1`分区创建一个ext4文件系统

5. `mkdir /data_1`:创建一个名为`/data_1`的挂载点

6. `mount /dev/vdc1 /data_1`:将分区/dev/vdc1挂载到/data_1

7. `df -h`:查看目前磁盘空间和使用情况

## （推荐）启动开机自动挂载分区

1. `cp /etc/fstab /etc/fstab.bak`，备份etc/fstab

2. `echo /dev/vdc1 /data_1 ext4 defaults 0 0 >> /etc/fstab`，向/etc/fstab里写入新分区信息

3. `cat /etc/fstab`，查看/etc/fstab的信息。如果返回结果里出现了写入的新分区信息，说明写入成功

