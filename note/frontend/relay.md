## Relay

Relay IR 解决了普通 DL 框架不支持 control flow 的特点。之前借用Python control flow会带来性能问题，尤其是op数量较多+op计算不是那么密集的时候，python op级别的调度开销就不可忽视了

### 表示方式

变量：Global variables are prefixed with @ and local variables with %.
函数：一等公民


[参考](https://zhuanlan.zhihu.com/p/390087648)