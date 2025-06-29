## 1.

#### *转移矩阵 \( M \)*

$$
M= \begin{bmatrix}
\frac{1}{3} & \frac{1}{2} & 0 \\
\frac{1}{3} & 0 & \frac{1}{2} \\
\frac{1}{3} & \frac{1}{2} & \frac{1}{2}
\end{bmatrix}
$$

#### *初始PageRank*
$$
r^{(0)} = \begin{bmatrix}
\frac{1}{3} \\
\frac{1}{3} \\
\frac{1}{3}
\end{bmatrix}
$$

#### *迭代计算*
$$
r^{(1)} = M \cdot r^{(0)}
$$

$$
r^{(1)}=
\begin{bmatrix}
\frac{1}{3} & \frac{1}{2} & 0 \\
\frac{1}{3} & 0 & \frac{1}{2} \\
\frac{1}{3} & \frac{1}{2} & \frac{1}{2}
\end{bmatrix}
\cdot
\begin{bmatrix}
\frac{1}{3} \\
\frac{1}{3} \\
\frac{1}{3}
\end{bmatrix}
=
\begin{bmatrix}
\frac{5}{18} \\
\frac{5}{18} \\
\frac{4}{9}
\end{bmatrix}
$$

---

## 2.

#### *转移矩阵 \( M \)*

$$
M= \begin{bmatrix}
\frac{1}{3} & \frac{1}{2} & 0 \\
\frac{1}{3} & 0 & \frac{1}{2} \\
\frac{1}{3} & \frac{1}{2} & \frac{1}{2}
\end{bmatrix}
$$

#### *初始PageRank*

$$
r^{(0)} = \begin{bmatrix}
\frac{1}{3} \\
\frac{1}{3} \\
\frac{1}{3}
\end{bmatrix}
$$

#### *迭代计算*

$$
r^{(1)} =\beta M \cdot r^{(0)} + \frac{(1 - \beta)}{N} \mathbf{e}
$$

$$
r^{(1)}=
\frac{4}{5}
\cdot
\begin{bmatrix}
\frac{1}{3} & \frac{1}{2} & 0 \\
\frac{1}{3} & 0 & \frac{1}{2} \\
\frac{1}{3} & \frac{1}{2} & \frac{1}{2}
\end{bmatrix}
\cdot
\begin{bmatrix}
\frac{1}{3} \\
\frac{1}{3} \\
\frac{1}{3}
\end{bmatrix}
+
\begin{bmatrix}
\frac{1}{15} \\
\frac{1}{15} \\
\frac{1}{15}
\end{bmatrix}
=
\begin{bmatrix}
\frac{13}{45} \\
\frac{13}{45} \\
\frac{19}{45}
\end{bmatrix}
$$

---

## 3.

#### *转移矩阵 \( M \)*

$$
M= \begin{bmatrix}
0 & \frac{1}{2} & 1 & 0\\
\frac{1}{3} & 0 & 0 & \frac{1}{2}\\
\frac{1}{3} & 0 & 0 & \frac{1}{2}\\
\frac{1}{3} & \frac{1}{2} & 0 & 0
\end{bmatrix}
$$

#### *初始PageRank*

$$
r^{(0)} = 
\begin{bmatrix}
\frac{1}{4} \\
\frac{1}{4} \\
\frac{1}{4} \\
\frac{1}{4}
\end{bmatrix}
$$

#### *迭代计算*

$$
r^{(1)} =\beta M \cdot r^{(0)} + \frac{(1 - \beta)}{|S|} v
$$

$$
r^{(1)}=
\frac{4}{5}
\cdot
\begin{bmatrix}
0 & \frac{1}{2} & 1 & 0\\
\frac{1}{3} & 0 & 0 & \frac{1}{2}\\
\frac{1}{3} & 0 & 0 & \frac{1}{2}\\
\frac{1}{3} & \frac{1}{2} & 0 & 0
\end{bmatrix}
\cdot
\begin{bmatrix}
\frac{1}{4} \\
\frac{1}{4} \\
\frac{1}{4} \\
\frac{1}{4}
\end{bmatrix}
+
\begin{bmatrix}
\frac{1}{10} \\
0 \\
\frac{1}{10} \\
0
\end{bmatrix}
=
\begin{bmatrix}
\frac{2}{5} \\
\frac{1}{6} \\
\frac{4}{15} \\
\frac{1}{6}
\end{bmatrix}
$$

---

## 4.

#### *原始情况（所有支持网页链向目标网页）*

- 设目标网页的 **PageRank** 为  **y **
- 设从其他可访问页面贡献给目标页面的 PageRank 为  **x**
- 设 **farm** 页面数量为 **M**, 总页面数为 **N**
- 阻尼因子为 **β**

每个 **farm** 页面的 **PageRank** 为：
$$
\text{Rank of farm page} = \frac{\beta y}{M} + \frac{1 - \beta}{N}
$$
目标页面的 **PageRank** 方程为：
$$
y = x + \beta M \left( \frac{\beta y}{M} + \frac{1 - \beta}{N} \right) + \frac{1 - \beta}{N}
$$
解得：
$$
y = \frac{x}{1 - \beta^2} + c \frac{M}{N}, \quad \text{其中 } c = \frac{\beta}{1 + \beta}
$$

#### *变体 (a)：每个支持网页只链向自己*

在这种情况下：
- 每个 **farm** 页面的出链指向自己。
- 目标页面不再从 **farm** 页面获得任何 **PageRank**

每个 **farm** 页面只从自己获得 **PageRank**：
$$
r_{\text{farm}} = \beta \cdot r_{\text{farm}} + \frac{1 - \beta}{N}
$$
解得：
$$
r_{\text{farm}} = \frac{1 - \beta}{N} \cdot \frac{1}{1 - \beta} = \frac{1}{N}
$$
目标页面只从可访问页面和随机跳转获得 **PageRank**：
$$
y = x + \frac{1 - \beta}{N}
$$

#### *变体 (b)：每个支持网页不链向任何网页（dangling 节点）*

在这种情况下：
- 每个 **farm** 页面没有出链，它们的 **PageRank** 会均匀分配给所有页面（包括目标页面）
- 目标页面除了从可访问页面获得 **PageRank**，还会从 **farm** 页面获得额外的贡献

所有 **farm** 页面都是 dangling 节点，它们的 **PageRank** 最终会收敛到：
$$
r_{\text{farm}} = \frac{1 - \beta}{N} + \beta \cdot \left( \text{来自 dangling 节点的均匀分配} \right)
$$
由于 **farm** 页面互相不传递 **PageRank**，最终：
$$
r_{\text{farm}} = \frac{1}{N}
$$

目标页面从以下来源获得 **PageRank**：
1. **可访问页面**（贡献 **x**）
2. **随机跳转**
3. **Farm 页面的 dangling 分配：**

$$
M \cdot \left( \frac{\beta}{N} \cdot \frac{1}{N} \right) = \frac{\beta M}{N^2}
$$
因此，目标页面的 **PageRank** 为：
$$
y = x + \frac{1 - \beta}{N} + \frac{\beta M}{N^2}
$$

#### *变体 (c)：每个支持网页同时链向自己和目标网页*

在这种情况下：
- 每个 **farm** 页面有 **两条出链**：
  - 一条指向 **自己**（权重**1/2**）
  - 一条指向 **目标页面**（权重**1/2**）
- 目标页面从 **farm** 页面获得部分 **PageRank**，同时 **farm** 页面也会保留部分 **PageRank**

每个 **farm** 页面的 **PageRank** 来源：
1. **来自自己**（权重**1/2** ）
2. **随机跳转**（权重**(1-β)/N**\)

因此：
$$
r_{\text{farm}} = \frac{\beta}{2} r_{\text{farm}} + \frac{1 - \beta}{N}
$$
解得：
$$
r_{\text{farm}} = \frac{1 - \beta}{N} \cdot \frac{1}{1 - \frac{\beta}{2}} = \frac{2(1 - \beta)}{N(2 - \beta)}
$$

目标页面的 **PageRank** 来源：
1. **可访问页面**
2. **随机跳转**
3. **farm 页面的贡献**

$$
M \cdot \left( \frac{\beta}{2} \cdot \frac{2(1 - \beta)}{N(2 - \beta)} \right) = \frac{\beta M (1 - \beta)}{N(2 - \beta)}
$$
因此，目标页面的 **PageRank** 为：
$$
y = x + \frac{1 - \beta}{N} + \frac{\beta M (1 - \beta)}{N(2 - \beta)}
$$
可以进一步整理为：
$$
y = x + \frac{1 - \beta}{N} \left( 1 + \frac{\beta M}{2 - \beta} \right)
$$

---

## 5.

- 导航度得分(**hub**)是链出网页的权威度得分之和
- 权威度得分(**authority**)是链入网页的导航度之和

#### *邻接矩阵 \( A \)*

$$
A = 
\begin{bmatrix}
0 & 1 & 1 & 0\\
1 & 0 & 0 & 1\\
1 & 0 & 0 & 1\\
1 & 1 & 0 & 0
\end{bmatrix}
\ \ \ \ \ \ \
A^\top =
\begin{bmatrix}
0 & 1 & 1 & 1\\
1 & 0 & 0 & 1\\
1 & 0 & 0 & 0\\
0 & 1 & 1 & 0
\end{bmatrix}
$$

#### *初始导航度和权威度*

$$
Initialize \ \ \ h_{\text{i}} = a_{\text{i}} = \frac{1}{\sqrt{N}}
$$

$$
h^{(0)} = a^{(0)} =
\begin{bmatrix}
\frac{1}{2} \\
\frac{1}{2} \\
\frac{1}{2} \\
\frac{1}{2}
\end{bmatrix}
$$

#### *迭代计算*

$$
h^{(1)} = A \cdot a^{(0)} =
\begin{bmatrix}
1 \\
1 \\
1 \\
1
\end{bmatrix}
\ \ \ \ \ \ \
a^{(1)} = A^\top \cdot h^{(1)} =
\begin{bmatrix}
\frac{3}{2} \\
1 \\
\frac{1}{2} \\
1
\end{bmatrix}
$$

$$
Normalize \ \ \ 
h^{(1)} =
\begin{bmatrix}
\frac{1}{2} \\
\frac{1}{2} \\
\frac{1}{2} \\
\frac{1}{2}
\end{bmatrix}
\ \ \ 
a^{(1)} = 
\begin{bmatrix}
\frac{\sqrt{2}}{2} \\
\frac{\sqrt{2}}{3} \\
\frac{\sqrt{2}}{6} \\
\frac{\sqrt{2}}{3}
\end{bmatrix}
$$

