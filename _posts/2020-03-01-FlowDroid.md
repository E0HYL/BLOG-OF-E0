---
layout: post
title: 编译运行FlowDroid
description: "How to use FlowDroid."
modified: 2020-3-1
tags: [Android]
image:
  feature: abstract-5.jpg
---
<!-- TOC -->

- [用Maven构建Flowdroid](#用maven构建flowdroid)
    - [主要问题和解决方法](#主要问题和解决方法)
    - [Maven 手动添加 JAR 包到本地仓库](#maven-手动添加-jar-包到本地仓库)
- [使用jar包运行Flowdroid](#使用jar包运行flowdroid)

<!-- /TOC -->
<!--more-->

## 用Maven构建Flowdroid

1. 下载`apache-maven-3.6.3.zip`并解压，将其下`bin`文件的路径添加到环境变量。
2. git clone到本地后，使用命令`mvn -DskipTests install`
3. 编译好的jar包在`soot-infoflow`和`soot-infoflow-android`下的`target`目录下。其中`apidocs`是Javadoc根据代码里的注释规范生成的（宝藏！）。

### 主要问题和解决方法

- Maven下载过慢：apache-maven-3.6.3\conf\settings.xml下配置镜像。阿里云的很快。我的镜像配置长下面这样，放在mirrors标签中。

  ```xml
    <mirror>
        <id>nexus-aliyun</id>
        <name>Nexus aliyun</name>
        <url>http://maven.aliyun.com/nexus/content/groups/public/</url>
        <mirrorOf>central</mirrorOf>
    </mirror>
    <!-- 中央仓库1 -->
    <mirror>
        <id>repo1</id>
        <mirrorOf>central</mirrorOf>
        <name>Human Readable Name for this Mirror.</name>
        <url>http://repo1.maven.org/maven2/</url>
    </mirror>
    <!-- 中央仓库2 -->
    <mirror>
        <id>repo2</id>
        <mirrorOf>central</mirrorOf>
        <name>Human Readable Name for this Mirror.</name>
        <url>http://repo2.maven.org/maven2/</url>
    </mirror>
  ```

- 有的jar包仍然下载很慢甚至失败。参考下节的方法先下载jar包到本地再Maven安装。

- 出现以下信息，大概是说失败记录缓存在本地了，在一段时间里都不会重试。可以直接在命令后加`-U`，或者是在settings.xml中修改[**updatePolicy**](https://stackoverflow.com/questions/4856307/when-maven-says-resolution-will-not-be-reattempted-until-the-update-interval-of)。

  ```shell
  xxx was cached in the local repository, resolution will not be reattempted until the update interval of xx has elapsed or updates are forced -> [Help 1]
  ```

- 构建soot-infoflow-android时出的错，`pom.xml`中AXMLPrinter的repository地址好像访问不了了，挂梯子也不行。这个包的主要作用是对Android的二进制格式的Androidmanifest.xml进行解析。

  我到开头说的那个搜索jar包的网上找dependency中对应的版本，有一个artifactId是AXMLPrinter2的repository，但下载链接也是无效的，感觉是很早的工具可能很久没维护了。

  最后我用的是com.android的AXMLPrinter的1.0.0，然后把repository的url修改一下。在FlowDroid项目的issues里自问自答了一下...[见下](https://github.com/secure-software-engineering/FlowDroid/issues/237)...

  > I did the following two steps, and it worked for me.
  >
  > - use the jar file [here](https://mvnrepository.com/artifact/com.android/AXMLPrinter/1.0.0), and modify the `dependency`:
  >
  > ```xml
  > <dependency>
  > 	<groupId>com.google.protobuf</groupId>
  > 	<artifactId>protobuf-java</artifactId>
  > 	<version>3.4.0</version>
  > </dependency>
  > ```
  >
  >
  > - modify the `repositories` like this:
  >
  > ```xml
  > <repositories>
  > 	<repository>
  > 		<id>soot-snapshot</id>
  > 		<name>Soot snapshot server</name>
  > 		<url>http://dev.91xmy.com/nexus/content/repositories/releases/</url>
  > 	</repository>
  > 	<!-- <repository>
  > 		<id>soot-release</id>
  > 		<name>Soot release server</name>
  > 		<url>https://soot-build.cs.uni-paderborn.de/nexus/repository/soot-release/</url>
  > 	</repository> -->
  > </repositories>
  > ```
  >
  > Build Success. Hope there would be no problems in the future.😄

### [Maven 手动添加 JAR 包到本地仓库](http://www.blogjava.net/fancydeepin/archive/2012/06/12/maven3-install-jar.html)

> [MVNRepository](http://mvnrepository.com/) 搜索可用的 JAR 包信息并下载

以spring-context-support为例，pom.xml中可用的Maven信息如下。

```xml
<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-context-support</artifactId>
    <version>3.1.0.RELEASE</version>
</dependency>
```

Maven安装本地jar包命令：**`mvn install:install-file -Dfile=jar包的位置 -DgroupId=groupId -DartifactId=artifactId -Dversion=version -Dpackaging=jar`**。用上面的例子即

```shell
mvn install:install-file -Dfile=spring-context-support-3.1.0.RELEASE.jar -DgroupId=org.springframework -DartifactId=spring-context-support -Dversion=3.1.0.RELEASE -Dpackaging=jar
```

## 使用jar包运行Flowdroid

若不需要修改源码，推荐此方式：只要在[Release页面](https://github.com/secure-software-engineering/FlowDroid/releases)下载`soot-infoflow-cmd-jar-with-dependencies.jar`即可。使用的示例可参考：https://github.com/hao-fu/MyFlowAnalysis

> *在soot中，函数的signature就是由该函数的类名，函数名，参数类型，以及返回值类型组成的字符串*
