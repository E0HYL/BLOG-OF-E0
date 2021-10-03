---
layout: post
title: Android指纹认证框架
description: "结合安卓（Q+版本）源码解析指纹认证框架"
modified: 2021-09-17
tags: [Android, Code]
math: true
image:
  feature: abstract-1.jpg
---

<details open><!-- 可选open -->
<summary>Contents</summary>
<div markdown="1">
<!-- TOC -->

- [安卓生物识别架构](#%E5%AE%89%E5%8D%93%E7%94%9F%E7%89%A9%E8%AF%86%E5%88%AB%E6%9E%B6%E6%9E%84)
- [BiometricManager](#biometricmanager)
- [BiometricPrompt](#biometricprompt)
- [FingerprintService](#fingerprintservice)
- [Fingerprint HAL](#fingerprint-hal)

<!-- /TOC -->
</div>
</details>

> 上半年在实习没有更博，做了与手机指纹认证相关的项目，由于项目需要阅读了 AOSP 相应部分的源码，在这篇博文里做一个系统化的整理~

## [安卓生物识别架构](https://source.android.com/security/biometric)

<figure><img src="https://e0hyl.github.io/BLOG-OF-E0/images/2021-01-19-COVID_Trojan/image-20210915222935702.png" alt="image-20210915222935702" style="zoom:80%;" /></figure>

Android Q(10)  开始引入了 [`BiometricManager` 类](https://developer.android.com/reference/android/hardware/biometrics/BiometricManager)，本文基于这一架构，结合最新的 AOSP 源码介绍安卓的指纹识别架构，侧重与发起指纹认证相关的方法（非其它生物认证方式或录入流程）。

Android Code Search: https://cs.android.com/

------

## BiometricManager

`frameworks/base/core/java/android/hardware/biometrics/BiometricManager.java`

```java
/**
 * A class that contains biometric utilities. For authentication, see {@link BiometricPrompt}.
 */
```

- static final 成员变量：定义在 `BiometricConstants` 类中的 `SUCCESS`, `ERROR_HW_UNAVAILABLE`, `ERROR_NONE_ENROLLED`, `NO_HARDWARE`

- 构造函数：传入 `Context`, `IAuthService` (Communication channel from `BiometricPrompt` and `BiometricManager` to `AuthService`) 

  > AuthService: System service that provides an interface for authenticating with biometrics and PIN/pattern/password to BiometricPrompt and lock screen.
  >
  > 在系统层，根据认证类型注册相应的服务（FingerprintService, FaceService, IrisService），做检查权限等

- 接口 `Authenticators`：认证类型的结合(e.g. <code>DEVICE_CREDENTIAL | BIOMETRIC_WEAK</code>)，包含生物认证的强度（`STRONG`, `WEAK`），用于设备保护的非生物认证方式（`DEVICE_CREDENTIAL`: PIN, pattern, password）

- `canAuthenticate`, `hasEnrolledBiometrics` 方法：检查设备是否支持生物识别身份验证，是否有录入的生物特征，以及相应的权限检查

<!--more-->

## BiometricPrompt

`frameworks/base/core/java/android/hardware/biometrics/BiometricPrompt.java`

```java
/**
 * A class that manages a system-provided biometric dialog.
 */
```

- 私有成员变量

  - 用于请求：`IAuthService` 类型的 `mService`，在构造函数中根据认证类型被赋值

    ```java
    mService = IAuthService.Stub.asInterface( // 得到一个IAuthService实例
                    ServiceManager.getService(Context.AUTH_SERVICE));
    ```

  - 用于接收：`IBiometricServiceReceiver` 类型的 final 变量 `mBiometricServiceReceiver`，其中的方法实现为根据系统服务返回的认证结果选择 `mAuthenticationCallback` 中的回调执行；

    ```java
    private final IBiometricServiceReceiver mBiometricServiceReceiver =
                new IBiometricServiceReceiver.Stub() {
    
            @Override
            public void onAuthenticationSucceeded(@AuthenticationResultType int authenticationType)
                    throws RemoteException {
                mExecutor.execute(() -> {
                    final AuthenticationResult result =
                            new AuthenticationResult(mCryptoObject, authenticationType);
                    mAuthenticationCallback.onAuthenticationSucceeded(result);
                });
            }
    
            @Override
            ...
    }    
    ```

    `mAuthenticationCallback` 为 ` AuthenticationCallback` 类型，在私有方法 `authenticateInternal` 中被赋值

  - ...

- 抽象内部类 `AuthenticationCallback`：包含 `onAuthenticationError`，`onAuthenticationHelp` (可恢复的错误，如“传感器脏了”)，`onAuthenticationSucceed`，`onAuthenticationFailed`等方法，用户可以定制化实现监听到相应事件时要做的；在调用`authenticate(User)`方法时传入

- 内部类 `OnAuthenticationCancelListener`： `CancellationSignal.OnCancelListener`接口函数的实现，调用私有方法`cancelAuthentication`，其中通过`mService`调用系统服务中的`cancelAuthentication`方法

- `authenticate`/`authenticateUser` 方法：主要传入的参数有 `CancellationSignal` 类型的 `cancel` 和`AuthenticationCallback` 类型的 `callback`，调用私有方法 `authenticateInternal` 完成

  (1) 给`cancel`注册一个监听器，为`OnAuthenticationCancelListener` 类的实例； 

  (2) 将 `mBiometricServiceReceiver` 依赖的 `mAuthenticationCallback` 赋值为应用实现的`callback`；

  (3) 通过 `mService`，传 `mBiometricServiceReceiver` 去调用系统服务中的`authenticate`方法；

  (4) 若在上述过程中有错误发生，则使用 `callback` 报告“硬件不可达”的错误

- 内部类 `Builder`和`ButtonInfo`，以及返回值类型是它们的方法：处理与系统对话框相关的事务

总结：应用创建 `BiometricPrompt` 实例时，会通过 [AIDL](https://developer.android.com/guide/components/aidl?hl=zh-cn) 拥有一个和系统层认证服务沟通的对象 `mService`。当调用 `authenticate(User)` 发起认证时，需要传入一个能终止进行中的系统操作的 `cancel` “开关”、定义了不同认证结果回调行为的 `callback`，接下来 `BiometricPrompt` 类内部会做的事主要有

(1) 给 `cancel` 开关装上监听器，用于开关被按下时通过 `mService` 告知系统服务去取消认证

(2) 将 `callback` 的约定包装进用于传回系统认证结果的（AIDL 定义的）接收器

(3) 将包装好的接收器作为参数，通过 `mService` 向系统服务发起认证请求

------

至此的两个类是应用可单独创建实例的，接下来介绍系统服务。与上层直接沟通的 `AuthService` 实现了 AIDL 定义的 `authenticate`, `cancelAuthentication` 等方法。

```java
private final class AuthServiceImpl extends IAuthService.Stub { // 实现 .aidl 生成的接口：扩展生成的 Binder 接口（<Interface>.Stub），并实现继承自 .aidl 文件的方法。
    @Override
    public void authenticate(IBinder token, long sessionId, int userId,
                             IBiometricServiceReceiver receiver, String opPackageName, Bundle bundle)
        throws RemoteException {
		...
        try {
            mBiometricService.authenticate(
                token, sessionId, userId, receiver, opPackageName, bundle, callingUid,
                callingPid, callingUserId);
        } finally {
            Binder.restoreCallingIdentity(identity);
        }
    }
    ...
}
```

在 `AuthService.authenticate` 方法中调用了 `BiometricService.authenticate`，其中实例 `mBiometricService` 在服务启动时通过 `registerAuthenticator`，使用认证类型对应的服务（如`FingerprintService`）进行了注册。

## FingerprintService

`frameworks/base/services/core/java/com/android/server/biometrics/fingerprint/FingerprintService.java`

```java
/**
 * A service to manage multiple clients that want to access the fingerprint HAL API.
 * The service is responsible for maintaining a list of clients and dispatching all
 * fingerprint-related events.
 *
 * @hide
 */
```

- 构造函数：初始化与错误 延时/总数 计数器相关的变量 `mTimedLockoutCleared`，`mFailedAttempts`

- 私有成员变量

  - `mDaemon`：`IBiometricsFingerprint` 类型，用于请求厂商库（支持`authenticate`, `cancel`等方法）

  - `mDaemonCallback`：`IBiometricsFingerprintClientCallback` 类型

    ```java
    /**
         * Receives callbacks from the HAL.
         */
    private IBiometricsFingerprintClientCallback mDaemonCallback =
        new IBiometricsFingerprintClientCallback.Stub() {
        @Override
        // onAuthenticated: FingerprintService.super.handleAuthenticated(authenticated, fp, token);
        // onError: FingerprintService.super.handleError(deviceId, error, vendorCode);
    	...
    };
    ```

    收到认证结果时调用父类 `BiometricServiceBase` 中的方法 `handleAuthenticated` 或 `handleError`

    ```java
    protected void handleAuthenticated(boolean authenticated,
                                       BiometricAuthenticator.Identifier identifier, ArrayList<Byte> token) {
        ...
        /*
            if (authenticated) {
                mPerformanceStats.accept++;
            } else {
                mPerformanceStats.reject++; // 失败次数+1
            }
        */
    }
    
    protected void handleError(long deviceId, int error, int vendorCode) {
        ...
        if (client != null && client.onError(deviceId, error, vendorCode)) {
            removeClient(client);
        }
    
        if (error == BiometricConstants.BIOMETRIC_ERROR_CANCELED) {
            mHandler.removeCallbacks(mResetClientState);
            ...
        }
    }
    ```

- 私有方法 `getFingerprintDaemon` ：根据 HIDL 的定义，获得厂商库支持的认证服务，并用 `mDaemonCallback` 接收 HAL 的返回

  ```java
      /** Gets the fingerprint daemon */
      private synchronized IBiometricsFingerprint getFingerprintDaemon() {
          if (mDaemon == null) {
              Slog.v(TAG, "mDaemon was null, reconnect to fingerprint");
              try {
                  mDaemon = IBiometricsFingerprint.getService();
              } ...
  
              try {
                  mHalDeviceId = mDaemon.setNotify(mDaemonCallback);
              } // Failed to open fingerprint HAL, try again later!
              }...
          }
          return mDaemon;
      }
  ```

- <Deprecated: 通信上层 `FingerprintManager` 时使用> 私有内部类 `FingerprintAuthClient`：

  包含 `isStrongBiometric`, `resetFailedAttempts`, `handleFailedAttempt` 等方法

  ```java
  public int handleFailedAttempt() { // 给用户的失败次数计数器+1
      final int currentUser = ActivityManager.getCurrentUser();
      mFailedAttempts.put(currentUser, mFailedAttempts.get(currentUser, 0) + 1);
      mTimedLockoutCleared.put(ActivityManager.getCurrentUser(), false);
  
      if (getLockoutMode() != AuthenticationClient.LOCKOUT_NONE) {
          scheduleLockoutResetForUser(currentUser);
      }
  
      return super.handleFailedAttempt();
  }
  ```

------

## Fingerprint HAL

`hardware/interfaces/biometrics/fingerprint/2.1/`

<figure><img src="{{ site.url }}/images/2021-01-19-COVID_Trojan/image-20210917180731158.png" alt="image-20210917180731158" style="zoom:80%;" /></figure>

IBiometricsFingerprint.hal 中的方法：供应商库中负责实现

IBiometricsFingerprintClientCallback.hal 中的方法：在 `FingerprintService` 中实现（参见`mDaemonCallback`）

------

参考：

1. source.android.com: [生物识别](https://source.android.com/security/biometric), [指纹识别身份验证 HIDL](https://source.android.com/security/authentication/fingerprint-hal)
2. CSDN: [Android Q 上的Biometric生物识别之Fingerprint指纹识别流程](https://blog.csdn.net/Easyhood/article/details/104278886)，[理解Aidl中Stub和Stub.Proxy](https://blog.csdn.net/scnuxisan225/article/details/49970217)
3. Medium: [Android Fingerprint Framework (1): FingerprintManager](https://jazzbeer1984.medium.com/android-fingerprint-framework-%E4%B8%80-fingerprintmanager-b72f5c8cd7ae)，[Android Fingerprint Framework (2): FingerprintService](https://jazzbeer1984.medium.com/android-fingerprint-framework-%E4%BA%8C-fingerprintservice-f6e7ed685e21)，[Android Fingerprint Framework (3): Fingerprint HAL](https://jazzbeer1984.medium.com/android-fingerprint-framework-3-fingerprint-hal-8cce5de0f0fb)，

