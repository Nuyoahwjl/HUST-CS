## 1.

- 乘法：$1 \rightarrow 2 \rightarrow 5$，用时$\Delta t+2\Delta t+\Delta t=4\Delta t$
- 加法：$1 \rightarrow 3 \rightarrow 4 \rightarrow 5$，用时$\Delta t+\Delta t+\Delta t+\Delta t=4\Delta t$

取一种最优调度：
$$
P_1=A_1B_1,\quad P_2=A_2B_2,\quad P_3=A_3B_3,\quad P_4=A_4B_4
$$
先算 4 次乘法，再按平衡树相加：
$$
S_{12}=P_1+P_2,\qquad S_{34}=P_3+P_4,\qquad S=S_{12}+S_{34}
$$
各操作启动时刻：

- $P_1$: $0$
- $P_2$: $2\Delta t$
- $P_3$: $4\Delta t$
- $P_4$: $6\Delta t$
- $S_{12}$: $7\Delta t$
- $S_{34}$: $10\Delta t$
- $S$: $14\Delta t$

时空图：

- $P_1:\quad 1[0,1] \rightarrow 2[1,3] \rightarrow 5[3,4]$
- $P_2:\quad 1[2,3] \rightarrow 2[3,5] \rightarrow 5[5,6]$
- $P_3:\quad 1[4,5] \rightarrow 2[5,7] \rightarrow 5[7,8]$
- $P_4:\quad 1[6,7] \rightarrow 2[7,9] \rightarrow 5[9,10]$
- $S_{12}:\quad 1[7,8] \rightarrow 3[8,9] \rightarrow 4[9,10] \rightarrow 5[10,11]$
- $S_{34}:\quad 1[10,11] \rightarrow 3[11,12] \rightarrow 4[12,13] \rightarrow 5[13,14]$
- $S:\quad 1[14,15] \rightarrow 3[15,16] \rightarrow 4[16,17] \rightarrow 5[17,18]$

所以总完成时间为：
$$
T_p=18\Delta t
$$
吞吐率：
$$
TP=\frac{7}{18\Delta t}
$$
非流水顺序执行时：

- 4 次乘法：$4\times 4\Delta t=16\Delta t$
- 3 次加法：$3\times 4\Delta t=12\Delta t$

所以：$T_s=16\Delta t+12\Delta t=28\Delta t$

加速比：
$$
Speed=\frac{T_s}{T_p}
=\frac{28\Delta t}{18\Delta t}
=\frac{14}{9}\approx 1.56
$$
流水线共 5 段，所以效率：
$$
\eta=\frac{S}{5}
=\frac{14/9}{5}
=\frac{14}{45}
\approx 0.311
\approx 31.1\%
$$

## 2.

### (1) 状态转移图

各段被占用的时刻为：

- $S_1:{1,7}$
- $S_2:{2,5}$
- $S_3:{3,4}$
- $S_4:{4,7}$
- $S_5:{5,6}$

禁止延迟：

- $S_1: 7-1=6$
- $S_2: 5-2=3$
- $S_3: 4-3=1$
- $S_4: 7-4=3$
- $S_5: 6-5=1$

所以禁止延迟集为：
$$
F={1,3,6}
$$
最大延迟考虑到 6，因此初始冲突向量为：
$$
C_0=(c_1c_2c_3c_4c_5c_6)=(1,0,1,0,0,1)=101001
$$
从初始状态**A (101001)** 出发：

- 取延迟 2：

$$
A \xrightarrow{2} B=101101
$$

- 取延迟 4：

$$
A \xrightarrow{4} C=111001
$$

- 取延迟 5：

$$
A \xrightarrow{5} A=101001
$$

状态 **B(101101)**：

- 取延迟 2：

$$
B \xrightarrow{2} D=111101
$$

- 取延迟 5：

$$
B \xrightarrow{5} A=101001
$$

状态 **C(111001)**：

- 取延迟 4：

$$
C \xrightarrow{4} C=111001
$$

- 取延迟 5：

$$
C \xrightarrow{5} A=101001
$$

状态 **D(111101)**：

- 取延迟 5：

$$
D \xrightarrow{5} A=101001
$$

状态转移图：

```text
A={1,3,6}
├─2→ B={1,3,4,6}
│    ├─2→ D={1,2,3,4,6}
│    │    └─5→ A
│    └─5→ A
├─4→ C={1,2,3,6}
│    ├─4→ C
│    └─5→ A
└─5→ A
```

### (2) 两种最优调度策略及最大吞吐率

<u>不等时间间隔调度</u>：要找**平均延迟最小的循环**

从状态图可见可能的循环主要有：

- $A \xrightarrow{5} A$ ，延迟为：$\bar{L}=5$

- $C \xrightarrow{4} C$，延迟为：$\bar{L}=4$

- $A \xrightarrow{2} B \xrightarrow{2} D \xrightarrow{5} A$，延迟为：$\bar{L}=\frac{2+2+5}{3}=3$

最优不等间隔调度策略为：
$$
2,\ 2,\ 5,\ 2,\ 2,\ 5,\ \cdots
$$
最大吞吐率：
$$
TP_{\max}=\frac{1}{3\Delta t}
$$
<u>等时间间隔调度（固定启动间隔）</u>：要求固定延迟 (d) 能一直重复使用。

从状态图看：

- 固定 $5$：可行($A \to A$)
- 固定 $4$：可行(先到 $C$，之后$C \to C$）
- 固定 $2$：不行，不能一直持续
- 固定 $1,3,6$：本来就禁止

因此最小可行固定间隔为：$4$

最优等间隔调度策略：
$$
4,\ 4,\ 4,\ 4,\ \cdots
$$
最大吞吐率：
$$
TP_{\max}=\frac{1}{4\Delta t}
$$

### (3) 连续输入 10 个任务，求实际吞吐率和加速比

单个任务从启动到结束共占 7 个时间单位，所以：
$$
\text{单任务非流水执行时间}=7\Delta t
$$
10 个任务顺序执行总时间：
$$
T_s=10\times 7\Delta t=70\Delta t
$$
<u>不等时间间隔最优策略：(2,2,5) 循环</u>

10 个任务共有 9 个启动间隔。

取最优周期序列：
$$
2,2,5,2,2,5,2,2,5
$$
总启动间隔和：
$$
9\text{ 个间隔之和}=3\times(2+2+5)=27
$$
最后一个任务完成总时间：
$$
T_p=27\Delta t+7\Delta t=34\Delta t
$$
实际吞吐率：
$$
TP=\frac{10}{34\Delta t}=\frac{5}{17\Delta t}
\approx 0.294\frac{1}{\Delta t}
$$
加速比：
$$
S=\frac{T_s}{T_p}=\frac{70}{34}=\frac{35}{17}\approx 2.06
$$
<u>等时间间隔最优策略：固定间隔 4</u>

9 个启动间隔全为 4：
$$
4\times 9=36
$$
总完成时间：
$$
T_p=36\Delta t+7\Delta t=43\Delta t
$$
实际吞吐率：
$$
TP=\frac{10}{43\Delta t}
$$
加速比：
$$
S=\frac{70}{43}\approx 1.63
$$

## 3.

设初始时 `R2 = x`，则 `R3 = x + 396`。
每次循环里先执行

```
ADDI R2, R2, #4
```

所以第 (k) 次迭代结束后：$R2 = x + 4k$
随后：$R4 = R3 - R2 = (x+396) - (x+4k) = 396 - 4k$

`BNZ R4, LOOP` 在 `R4 != 0` 时跳转，因此当$396 - 4k = 0 \Rightarrow k = 99
$时退出循环。

所以：

- 一共执行 **99 次循环**
- 前 **98 次跳转成功**
- 第 **99 次不跳转**

### (1) 无任何定向硬件，分支采用“排空流水线”

指令序列记为：

- `I1 = LW R1,0(R2)`
- `I2 = ADDI R1,R1,#1`
- `I3 = SW R1,0(R2)`
- `I4 = ADDI R2,R2,#4`
- `I5 = SUB R4,R3,R2`
- `I6 = BNZ R4,LOOP`

无定向时，相关必须等到前一条在 **WB** 写回后，后一条才能在 **ID** 读到。
又因为题目允许“同周期先写后读”，所以后一条可以在前一条 `WB` 的同一个周期做 `ID`。

时空图（一轮）：

```text
周期 →      1   2   3   4   5   6   7   8   9  10   11  12  13  14  15  16
I1: LW     IF  ID  EX MEM  WB
I2: ADDI       IF  ID  ID  ID  EX MEM  WB
I3: SW                     IF  ID  ID  ID  EX MEM
I4: ADDI                               IF  ID  EX MEM  WB
I5: SUB                                    IF  ID  ID  ID  EX MEM  WB
I6: BNZ                                                IF  ID  ID  ID
下一轮 LW                                                               IF
```

这里的停顿原因：

- `LW -> ADDI R1`：等 `LW` 写回
- `ADDI R1 -> SW`：等 `ADDI` 写回
- `ADDI R2 -> SUB`：等 `ADDI` 写回
- `SUB -> BNZ`：等 `SUB` 写回
- 分支“排空流水线”，所以在分支确定前不取下一轮

因此一轮循环从本轮 `LW` 的 `IF` 到下一轮 `LW` 的 `IF`，正好隔 **15 个周期**。

所以总周期数：
$$
99 \times 15 = 1485
$$

### (2) 有正常定向路径，分支预测失败（predict not taken）

正常定向可消除大多数 RAW 相关，只剩下：

- `LW -> ADDI R1`：仍需要 **1 个气泡**
- `SUB -> BNZ`：仍需要 **1 个气泡**（分支在 ID 判定）
- 由于采用“预测不跳转”，而实际前 98 次都跳转，所以每次 taken branch 会带来 1 个错误取指槽位

时空图（一轮）：

```text
周期 →      1   2   3   4   5   6   7   8   9  10
I1: LW     IF  ID  EX MEM  WB
I2: ADDI       IF  ID  ID  EX MEM  WB
I3: SW                 IF  ID  EX MEM
I4: ADDI                   IF  ID  EX MEM  WB
I5: SUB                        IF  ID  EX MEM  WB
I6: BNZ                            IF  ID  ID
下一轮 LW                                       IF
```

说明：

- 第 3→4 拍之间，`LW` 的结果还没到，`I2` 停 1 拍
- `SUB -> BNZ` 之间再停 1 拍
- 分支在第 9 拍判定后，下一轮从第 10 拍开始取指

于是：

- 第一轮完成到第 **10** 拍
- 相邻两轮的起点间隔是 **9** 拍

总周期数：
$$
10 + (99-1)\times 9 = 10 + 98\times 9 = 892
$$

### (3) 有正常定向路径时，对循环调度

目标是把相关隔开，消掉停顿。

原循环：

```asm
LOOP: LW   R1, 0(R2)
      ADDI R1, R1, #1
      SW   R1, 0(R2)
      ADDI R2, R2, #4
      SUB  R4, R3, R2
      BNZ  R4, LOOP
```

可以改写为：

```asm
LOOP: LW   R1, 0(R2)
      ADDI R2, R2, #4
      ADDI R1, R1, #1
      SUB  R4, R3, R2
      SW   R1, -4(R2)
      BNZ  R4, LOOP
```

把 `ADDI R2,R2,#4` 提前后：

- `LW` 仍然访问旧地址 `0(R2)`，没变
- `SW` 原来要写回旧地址 `0(R2_old)`
- 现在 `R2` 已经先加了 4，所以要改成 `SW R1,-4(R2)`，这样地址仍是旧地址

同时：

- `LW` 和 `ADDI R1` 中间塞入了 `ADDI R2`
- `SUB` 和 `BNZ` 中间塞入了 `SW`

这样正好把两处关键相关都错开。时空图：

```text
周期 →          1   2   3   4   5   6   7   8
LW             IF  ID  EX MEM  WB
ADDI R2            IF  ID  EX MEM  WB
ADDI R1                IF  ID  EX MEM  WB
SUB                        IF  ID  EX MEM  WB
SW                             IF  ID  EX MEM
BNZ                                IF  ID
下一轮 LW                                   IF
```

这时本轮到下一轮的启动间隔为 **7 个周期**。

因此总周期数：
$$
8 + (99-1)\times 7 = 8 + 98\times 7 = 694
$$