# FAT TREE(k=4) 

![](./FatTree.png)

- ✅由于不使用控制器，交换机需要配置为 **`standalone` 模式**，以启用二层自学习和转发功能
- ✅为防止广播风暴，必须启用 **`STP`（生成树协议）**，以消除拓扑中的环路
- ✅避免在 **`Mininet CLI`** 中手动配置交换机，在脚本中完成所有设置
- ✅脚本自动执行 **`pingall`** 测试，验证全网连通性，并显示所有交换机MAC表

---

## 运行脚本
```bash
cd ~/桌面/lab1
sudo python3 fat_tree_topo.py
``` 

## 测试结果
在 Mininet CLI 中查看：
```bash
*** Ping: testing ping reachability
h1 -> h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 
h2 -> h1 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 
h3 -> h1 h2 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 
h4 -> h1 h2 h3 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 
h5 -> h1 h2 h3 h4 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 
h6 -> h1 h2 h3 h4 h5 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 
h7 -> h1 h2 h3 h4 h5 h6 h8 h9 h10 h11 h12 h13 h14 h15 h16 
h8 -> h1 h2 h3 h4 h5 h6 h7 h9 h10 h11 h12 h13 h14 h15 h16 
h9 -> h1 h2 h3 h4 h5 h6 h7 h8 h10 h11 h12 h13 h14 h15 h16 
h10 -> h1 h2 h3 h4 h5 h6 h7 h8 h9 h11 h12 h13 h14 h15 h16 
h11 -> h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h12 h13 h14 h15 h16 
h12 -> h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h13 h14 h15 h16 
h13 -> h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h14 h15 h16 
h14 -> h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h15 h16 
h15 -> h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h16 
h16 -> h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 
*** Results: 0% dropped (240/240 received)
*** SUCCESS: All hosts are connected!
```
```bash
*** Starting CLI:
mininet> links
a11-eth3<->e11-eth3 (OK OK) 
a11-eth4<->e12-eth3 (OK OK) 
a12-eth3<->e11-eth4 (OK OK) 
a12-eth4<->e12-eth4 (OK OK) 
a21-eth3<->e21-eth3 (OK OK) 
a21-eth4<->e22-eth3 (OK OK) 
a22-eth3<->e21-eth4 (OK OK) 
a22-eth4<->e22-eth4 (OK OK) 
a31-eth3<->e31-eth3 (OK OK) 
a31-eth4<->e32-eth3 (OK OK) 
a32-eth3<->e31-eth4 (OK OK) 
a32-eth4<->e32-eth4 (OK OK) 
a41-eth3<->e41-eth3 (OK OK) 
a41-eth4<->e42-eth3 (OK OK) 
a42-eth3<->e41-eth4 (OK OK) 
a42-eth4<->e42-eth4 (OK OK) 
c1-eth1<->a11-eth1 (OK OK) 
c1-eth2<->a21-eth1 (OK OK) 
c1-eth3<->a31-eth1 (OK OK) 
c1-eth4<->a41-eth1 (OK OK) 
c2-eth1<->a11-eth2 (OK OK) 
c2-eth2<->a21-eth2 (OK OK) 
c2-eth3<->a31-eth2 (OK OK) 
c2-eth4<->a41-eth2 (OK OK) 
c3-eth1<->a12-eth1 (OK OK) 
c3-eth2<->a22-eth1 (OK OK) 
c3-eth3<->a32-eth1 (OK OK) 
c3-eth4<->a42-eth1 (OK OK) 
c4-eth1<->a12-eth2 (OK OK) 
c4-eth2<->a22-eth2 (OK OK) 
c4-eth3<->a32-eth2 (OK OK) 
c4-eth4<->a42-eth2 (OK OK) 
e11-eth1<->h1-eth0 (OK OK) 
e11-eth2<->h2-eth0 (OK OK) 
e12-eth1<->h3-eth0 (OK OK) 
e12-eth2<->h4-eth0 (OK OK) 
e21-eth1<->h5-eth0 (OK OK) 
e21-eth2<->h6-eth0 (OK OK) 
e22-eth1<->h7-eth0 (OK OK) 
e22-eth2<->h8-eth0 (OK OK) 
e31-eth1<->h9-eth0 (OK OK) 
e31-eth2<->h10-eth0 (OK OK) 
e32-eth1<->h11-eth0 (OK OK) 
e32-eth2<->h12-eth0 (OK OK) 
e41-eth1<->h13-eth0 (OK OK) 
e41-eth2<->h14-eth0 (OK OK) 
e42-eth1<->h15-eth0 (OK OK) 
e42-eth2<->h16-eth0 (OK OK) 
```
综上分析，h1到h16的一条路径为：
`h1-eth0` <-> `e11-eth1` <-> `e11-eth3` <-> `a11-eth3` <-> `a11-eth1` <-> `c1-eth1` <-> `c1-eth4` <-> `a41-eth1` <-> `a41-eth4` <-> `e42-eth3` <-> `e42-eth2` <-> `h16-eth0`
