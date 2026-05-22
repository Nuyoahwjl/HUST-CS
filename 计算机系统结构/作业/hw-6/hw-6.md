## 9.9

设：$N=32=2^5$

处理器编号用 $$5$$ 位二进制表示。定义：

$$Cube_i(x)=x\oplus 2^i$$

$$\sigma(b_4b_3b_2b_1b_0)=(b_3b_2b_1b_0b_4)$$

$$\beta(b_4b_3b_2b_1b_0)=(b_0b_3b_2b_1b_4)$$

$$PM2I_{+i}(x)=(x+2^i)\bmod N$$


### (1)
$$Cube_2(12)=Cube_2(01100_2)=01000_2=8$$

$$\sigma(8)=\sigma(01000_2)=10000_2=16$$

$$\beta(9)=\beta(01001_2)=11000_2=24$$

$$PM2I_{+3}(28)=(28+2^3)\bmod 32=36\bmod 32=4$$

$$Cube_0(\sigma(4))=Cube_0(\sigma(00100_2))=Cube_0(01000_2)=01001_2=9$$

$$\sigma(Cube_0(18))=\sigma(Cube_0(10010_2))=\sigma(10011_2)=00111_2=7$$

### (2)

混洗交换网由 $$Cube_0$$ 和 $$\sigma$$ 构成。

对 $$N=32=2^5$$，混洗交换网直径为：

$$D=2\log_2N-1=2\times5-1=9$$

因此网络直径为：$9$

从处理机 $$5$$ 到处理机 $$7$$ 的一条最短路径为：

$$5\rightarrow4\rightarrow8\rightarrow9\rightarrow18\rightarrow19\rightarrow7$$

对应操作是：

$$5\xrightarrow{Cube_0}4\xrightarrow{\sigma}8\xrightarrow{Cube_0}9\xrightarrow{\sigma}18\xrightarrow{Cube_0}19\xrightarrow{\sigma}7$$

所以最短路径经过：$6$步。

## 9.13

对于 $$N=8$$ 的三级 Omega 网络，有：$\log_2 8=3$

每级有：$\frac{N}{2}=4$个 $$2\times2$$ 开关。

设第 $$k$$ 级第 $$j$$ 个开关记为：$S_{k,j}$

其中：$k=1,2,3$ ; $j=0,1,2,3$

Omega 网络按目的地址位逐级选路。目的处理机编号为：$d=d_2d_1d_0$

第 $$0$$ 级看 $$d_2$$，第 $$1$$ 级看 $$d_1$$，第 $$2$$ 级看 $$d_0$$。

![image-20260506221445085](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20260506221445085.png)

### 一、先看 $$P6\rightarrow P0\sim P4$$

处理机 $$P6$$ 的数据从左边编号 $$6$$ 输入。

根据图中连线，$$P6$$ 进入第一级第 $$2$$ 个开关的下输入端，即：$P6\rightarrow S_{1,2}\text{下输入}$

目标是：$P0,P1,P2,P3,P4$

所以路径可以分解为：$P6\rightarrow P0,P1,P2,P3$ 和 $P6\rightarrow P4$

因此第一级开关 $$S_{1,2}$$ 要把下输入复制到上下两个输出：$S_{1,2}:\ 下输入\rightarrow 上输出,\ 下输出$

也就是广播状态。

继续沿图走：$S_{1,2}\text{上输出}\rightarrow S_{2,0}\text{下输入}$

这一路要到 $$P0,P1,P2,P3$$，所以 $$S_{2,0}$$ 也要广播：$S_{2,0}:\ 下输入\rightarrow 上输出,\ 下输出$

然后：$S_{2,0}\text{上输出}\rightarrow S_{3,0}\text{上输入}\rightarrow P0,P1$

所以：$S_{3,0}:\ 上输入\rightarrow 上输出,\ 下输出$

即：$P0,P1$

同时：$S_{2,0}\text{下输出}\rightarrow S_{3,1}\text{上输入}\rightarrow P2,P3$

所以：$S_{3,1}:\ 上输入\rightarrow 上输出,\ 下输出$

即：$P2,P3$

另一支路：$S_{1,2}\text{下输出}\rightarrow S_{2,1}\text{下输入}$

这一路只去 $$P4$$，所以：$S_{2,1}:\ 下输入\rightarrow 上输出$

再到第三级：$S_{2,1}\text{上输出}\rightarrow S_{3,2}\text{上输入}\rightarrow P4$

因此：$S_{3,2}:\ 上输入\rightarrow 上输出$


### 二、再看 $$P3\rightarrow P5\sim P7$$

处理机 $$P3$$ 的数据从左边编号 $$3$$ 输入。

根据图中连线：$P3\rightarrow S_{1,3}\text{上输入}$

目标是：$P5,P6,P7$

所以第一级只需要走下输出：$S_{1,3}:\ 上输入\rightarrow 下输出$

继续沿图走：$S_{1,3}\text{下输出}\rightarrow S_{2,3}\text{下输入}$

这一路要分成：$P5$ 和 $P6,P7$

所以第二级开关 $$S_{2,3}$$ 要广播：$S_{2,3}:\ 下输入\rightarrow 上输出,\ 下输出$

其中：$S_{2,3}\text{上输出}\rightarrow S_{3,2}\text{下输入}\rightarrow P5$

所以：$S_{3,2}:\ 下输入\rightarrow 下输出$

另一支路：$S_{2,3}\text{下输出}\rightarrow S_{3,3}\text{下输入}\rightarrow P6,P7$

所以：$S_{3,3}:\ 下输入\rightarrow 上输出,\ 下输出$


### 三、开关状态汇总表

| 开关        | 状态                                                    |
| ----------- | ------------------------------------------------------- |
| $$S_{1,0}$$ | 未用                                                    |
| $$S_{1,1}$$ | 未用                                                    |
| $$S_{1,2}$$ | $$下输入\rightarrow 上输出,\ 下输出$$                   |
| $$S_{1,3}$$ | $$上输入\rightarrow 下输出$$                            |
| $$S_{2,0}$$ | $$下输入\rightarrow 上输出,\ 下输出$$                   |
| $$S_{2,1}$$ | $$下输入\rightarrow 上输出$$                            |
| $$S_{2,2}$$ | 未用                                                    |
| $$S_{2,3}$$ | $$下输入\rightarrow 上输出,\ 下输出$$                   |
| $$S_{3,0}$$ | $$上输入\rightarrow 上输出,\ 下输出$$                   |
| $$S_{3,1}$$ | $$上输入\rightarrow 上输出,\ 下输出$$                   |
| $$S_{3,2}$$ | $$上输入\rightarrow 上输出,\ 下输入\rightarrow 下输出$$ |
| $$S_{3,3}$$ | $$下输入\rightarrow 上输出,\ 下输出$$                   |


### 四、是否能同时实现？

能同时实现。

关键是看是否有两个数据流抢同一个开关输出。这里唯一同时被两个播送请求使用的是第三级的 $$S_{3,2}$$：$P6\rightarrow P4$

需要：$S_{3,2}:\ 上输入\rightarrow 上输出$

而：$P3\rightarrow P5$

需要：$S_{3,2}:\ 下输入\rightarrow 下输出$

这两个连接互不冲突，刚好是直通状态。
