# 采用RISC-V代理内核的操作系统和系统能力培养实验

[前言](preliminary.md)

## 第一部分：操作系统实验

[第一章. RISC-V体系结构](chapter1_riscv.md) ----- 课程资源：    [PPT](./resources/第一章.RISC-V体系结构.pptx)        [视频讲解](https://www.bilibili.com/video/BV1ca411K7Zy)

- [1.1 RISC-V发展历史](chapter1_riscv.md#history)  
- [1.2 RISC-V汇编语言](chapter1_riscv.md#assembly)  
- [1.3 机器的特权状态](chapter1_riscv.md#machinestates)  
- [1.4 中断和中断处理](chapter1_riscv.md#traps)  
- [1.5 页式虚存管理](chapter1_riscv.md#paging)  
- [1.6 相关工具软件](chapter1_riscv.md#toolsoftware)  

[第二章. 实验环境配置与实验构成](chapter2_installation.md)  ----- 课程资源：    [PPT](./resources/第二章.实验环境配置与实验构成.pptx)        [视频讲解](https://www.bilibili.com/video/BV1hS4y147ZP)

 - [2.1 操作系统部分实验环境安装](chapter2_installation.md#environments)  
 - [2.2 riscv-pke（实验）代码的获取](chapter2_installation.md#preparecode)  
 - [2.3 PKE实验的组成](chapter2_installation.md#pke_experiemnts)  

[第三章. PKE实验1：系统调用、异常和外部中断](chapter3_traps.md)    ----- 课程资源：    [PPT](./resources/第三章.实验1：系统调用、异常和外部中断.pptx)        [视频讲解](https://www.bilibili.com/video/BV1aW4y1a71T)

 - [3.1 实验1的基础知识](chapter3_traps.md#fundamental)   
 - [3.2 lab1_1 系统调用](chapter3_traps.md#syscall)  
 - [3.3 lab1_2 异常处理](chapter3_traps.md#exception)  
 - [3.4 lab1_3（外部）中断](chapter3_traps.md#irq)  
 - [3.5 lab1_challenge1 打印用户程序调用栈（难度：&#9733;&#9733;&#9733;&#9734;&#9734;）](chapter3_traps.md#lab1_challenge1_backtrace) 
 - [3.6 lab1_challenge2 打印异常代码行（难度：&#9733;&#9733;&#9733;&#9734;&#9734;）](chapter3_traps.md#lab1_challenge2_errorline)
 - [3.7 lab1_challenge3 多核启动及运行（难度：&#9733;&#9733;&#9733;&#9733;&#9734;）](chapter3_traps.md#lab1_challenge3_multicore)

[第四章. PKE实验2：内存管理](chapter4_memory.md)  ----- 课程资源：    [PPT](./resources/第四章.实验2：内存管理.pptx)        [视频讲解](https://www.bilibili.com/video/BV1yd4y1T77a)

 - [4.1 实验2的基础知识](chapter4_memory.md#fundamental)  
 - [4.2 lab2_1 虚实地址转换](chapter4_memory.md#lab2_1_pagetable)  
 - [4.3 lab2_2 简单内存分配和回收](chapter4_memory.md#lab2_2_allocatepage)  
 - [4.4 lab2_3 缺页异常](chapter4_memory.md#lab2_3_pagefault)  
 - [4.5 lab2_challenge1 复杂缺页异常（难度：&#9733;&#9734;&#9734;&#9734;&#9734;）](chapter4_memory.md#lab2_challenge1_pagefault)
 - [4.6 lab2_challenge2 堆空间管理（难度：&#9733;&#9733;&#9733;&#9733;&#9734;）](chapter4_memory.md#lab2_challenge2_singlepageheap)
 - [4.7 lab2_challenge3 多核内存管理（难度：&#9733;&#9733;&#9734;&#9734;&#9734;）](chapter4_memory.md#lab2_challenge3_multicoremem)

[第五章. PKE实验3：进程管理](chapter5_process.md)  ----- 课程资源：    [PPT](./resources/第五章.实验3：进程管理.pptx)        [视频讲解](https://www.bilibili.com/video/BV1Qe4y1D7dv)

 - [5.1 实验3的基础知识](chapter5_process.md#fundamental)  
 - [5.2 lab3_1 进程创建](chapter5_process.md#lab3_1_naive_fork)  
 - [5.3 lab3_2 进程yield](chapter5_process.md#lab3_2_yield)  
 - [5.4 lab3_3 循环轮转调度](chapter5_process.md#lab3_3_rrsched)  
 - [5.5 lab3_challenge1 进程等待和数据段复制（难度：&#9733;&#9733;&#9734;&#9734;&#9734;）](chapter5_process.md#lab3_challenge1_wait) 
 - [5.6 lab3_challenge2 实现信号量（难度：&#9733;&#9733;&#9733;&#9734;&#9734;）](chapter5_process.md#lab3_challenge2_semaphore) 
 - [5.7 lab3_challenge3 写时复制（Copy On Write）（难度：&#9733;&#9733;&#9733;&#9734;&#9734;）](chapter5_process.md#lab3_challenge3_cow) 

[第六章. PKE实验4：文件系统](chapter6_filesystem.md) ----- 课程资源：    [PPT](./resources/第六章.实验4：文件系统.pptx)        [视频讲解](https://www.bilibili.com/video/BV1Us4y1h7tT)

 - [6.1 实验4的基础知识](chapter6_filesystem.md#fundamental)  
 - [6.2 lab4_1 文件操作](chapter6_filesystem.md#lab4_1_file)  
 - [6.3 lab4_2 目录文件](chapter6_filesystem.md#lab4_2_dir)  
 - [6.4 lab4_3 硬链接](chapter6_filesystem.md#lab4_3_hardlink)  
 - [6.5 lab4_challenge1 相对路径（难度：&#9733;&#9733;&#9733;&#9734;&#9734;）](chapter6_filesystem.md#lab4_challenge1_pwd)
 - [6.6 lab4_challenge2 重载执行（难度：&#9733;&#9733;&#9733;&#9733;&#9734;）](chapter6_filesystem.md#lab4_challenge2_exec)
 - [6.7 lab4_challenge3 简易Shell（难度：&#9733;&#9733;&#9733;&#9733;&#9733;）](chapter6_filesystem.md#lab4_challenge3_shell)

## 第二部分：系统能力培养实验

[第七章. RISCV处理器在PYNQ上的部署和接口实验](chapter7_riscv_on_pynq.md)  ----- 课程资源：    [PPT](./resources/第七章.fpga实验：PYNQ上的部署.pptx)        [视频讲解](https://www.bilibili.com/video/BV1nt4y1n7dm)

- [7.1 系统能力培养部分实验环境安装](chapter7_riscv_on_pynq.md#environments)  
- [7.2 fpga实验1：在Rocket Chip上添加uart接口](chapter7_riscv_on_pynq.md#hardware_lab1)
- [7.3 fpga实验2：以中断方式实现uart通信](chapter7_riscv_on_pynq.md#hardware_lab2)
- [7.4 fpga实验3：配置连接到PS端的USB设备](chapter7_riscv_on_pynq.md#hardware_lab3)

[第八章. PKE实验5：设备和文件](chapter8_device.md)  ----- 课程资源：    [PPT](./resources/第八章.实验5：设备和文件.pptx)        [视频讲解](https://www.bilibili.com/video/BV1LB4y157Rb)

 - [8.1 实验5的基础知识](chapter8_device.md#fundamental)  
 - [8.2 lab5_1 轮询方式](chapter8_device.md#polling)  
 - [8.3 lab5_2 中断方式](chapter8_device.md#PLIC)  
 - [8.4 lab5_3 主机设备访问](chapter8_device.md#hostdevice)  



