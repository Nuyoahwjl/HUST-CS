# 第二章．实验环境配置与实验构成

### 目录

- [2.1 操作系统部分实验环境安装](#environments)
  - [2.1.1 安装开发环境](#subsec_osenvironments)
  - [2.1.2 安装支撑软件](#subsec_softwarepackages)
  - [2.1.3 头歌平台](#subsec_educoder)
- [2.2 riscv-pke（实验）代码的获取](#preparecode)
- [2.3 PKE实验构成](#pke_experiemnts)

`<a name="environments"></a>`

## 2.1 操作系统部分实验环境安装

`<a name="subsec_osenvironments"></a>`

### 2.1.1 安装开发环境

安装即将编译、构建（build）和执行riscv-pke的环境，我们给出以下3种选择：

**方案一：Windows Subversion Linux（WSL）**

如果读者的工作环境是Windows10的专业版，可采用WSL（Windows Subversion Linux）+MobaXterm的组合来搭建PKE的实验环境。在Windows10专业版上配置该环境的说明，可以参考[这里](https://zhuanlan.zhihu.com/p/81769058)。关于WSL，**应尽量使用WSL1不要升级到WSL2**，因为如果代码存放的位置在/mnt目录下，WSL1中运行编译、构建的速度比WSL2快很多（参考[这里](https://github.com/microsoft/WSL/issues/4197)）。

需要说明的是，PKE实验并不需要中文字体、图形界面或者JAVA的支持，所以读者在安装WSL的过程中无须安装于汉化相关的包，也无需安装xfce、JDK等。只需要安装WSL的基础环境后，再按照[下一节](#subsec_softwarepackages)的说明继续完成rsicv-pke开发环境的安装。

**方案二：Ubuntu或者其他Linux发行版**

读者可以采用VMware、VirtualBox等软件，在客户机中安装客户机，再在客户机中安装执行支撑软件来构建实验环境。客户机的选择我们推荐采用Ubuntu 16.04LTS（x86_64）或更高版本的Ubuntu发行版（如18.04LTS、20.04LTS或更高）操作系统，或者**华为openEuler操作系统**。我们未在其他系统（如arch，RHEL等）上做过测试，但理论上只要将实验中所涉及到的安装包替换成其他系统中的等效软件包，就可完成同样效果。

**方案三：Docker环境**

读者可以采用Docker作为开发环境。我们提供的 Docker 镜像中已经预装了完整的实验环境，实验源代码以及必要的辅助软件，读者无需手动安装任何支撑软件，便可轻松运行实验。我们建议读者使用 **x86_64**（例如，采用Intel或者ARM处理器的笔记本、台式机电脑） 或 **arm64** 处理器架构（例如，苹果公司新款**采用M系列处理器**的笔记本，如MacBook等）的客户机来运行 Docker 容器。具体的安装过程可以看[下一节](#subsec_docker)。

**方案四：头歌实验平台**

我们在EduCoder实验平台（网址：https://www.educoder.net ）上创建了riscv-pke的同步实验课程，课程的终端环境（docker虚拟机）中已完成实验所需软件工具的安装，所以如果读者是在头歌平台上选择的本课程，则可跳过本节的实验环境搭建过程以及执行支撑软件的安装过程，直接进入通过终端（命令行）进入实验环境，头歌实验平台的使用方法见[2.1.3节](#subsec_educoder)。

`<a name="subsec_softwarepackages"></a>`

### 2.1.2 安装支撑软件

riscv-pke的源代码的编译（compile）、构建（build）需要专门的risc-v交叉编译器，所构建的代理内核（及应用）的执行也需要专门的risc-v虚拟机（spike）的支持。这一节将介绍这些支撑软件的安装过程，需要说明的是，该安装过程适合操作系统环境（见[2.1.1节](#subsec_osenvironments)）为**WSL**，或者**Ubuntu发行版**的情况。对于头歌实验平台，则无需对执行支撑软件进行任何安装。

PKE实验涉及到的执行支撑软件有：

- RISC-V交叉编译器（及附带的主机编译器、构造工具等）；
- spike模拟器；

以下分别介绍这些执行支撑软件的安装过程，我们给出以下三种方案：

**方案一：下载自包含交叉编译器+执行环境**

● 第一步，安装依赖库

RISC-V交叉编译器的执行仍然需要一些本地支撑软件包，可使用以下命令安装：

`$ sudo apt-get install autoconf automake autotools-dev curl libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev device-tree-compiler`

● 第二步，下载自包含交叉编译器+执行环境

到[**这里**](./resources/riscv64-elf-gcc-20210923.tgz)下载自包含交叉编译器+执行环境的tgz包（大小约为72MB，感谢张杰老师为大家制作），采用以下命令解开tgz包：

$ tar xf riscv64-elf-gcc-20210923.tgz

解压完成后，将在当前目录产生名为**riscv64-elf-gcc**的目录，**该目录下包含RISC-V交叉编译器以及spike模拟器**。

● 第三步，设置环境变量

`$ export RISCV=$PWD/riscv64-elf-gcc`

`$ export PATH=$PATH:$RISCV/bin`

以上命令设置了RISCV环境变量，指向在第二步中的tgz包展开目录，并且将交叉编译器的可执行文件所在的目录加入到了系统路径中。这样，我们就可以在riscv-pke的工作目录调用RISC-V交叉编译器所包含的工具软件了。

**建议读者将以上两个export命令，加入到 `~/.bashrc`，`~/.profile`或 `/etc/profile`文件的末尾（注意替换$PWD环境变量）**。这样，每次重新启动（并打开终端程序）后，系统会自动设置这两个环境变量，而不用每次都要手动输入以上命令。

**方案二：完整自行构建交叉编译器和执行环境**

对于新安装的Ubuntu操作系统，我们强烈建议读者在新装环境中完整构建（build）RISC-V交叉编译器，以及spike模拟器。如果强调环境的可移植性，可以考虑在虚拟机中安装完整系统和环境，之后将虚拟机进行克隆和迁移。

RISC-V交叉编译器是与Linux自带的GCC编译器类似的一套工具软件集合，不同的是，x86_64平台上Linux自带的GCC编译器会将源代码编译、链接成为适合在x86_64平台上运行的二进制代码（称为native code），而RISC-V交叉编译器则会将源代码编译、链接成为在RISC-V平台上运行的代码。后者（RISC-V交叉编译器生成的二进制代码）是无法在x86_64平台（即x86_64架构的Ubuntu环境下）直接运行的，它的运行需要模拟器（我们采用的spike）的支持。

一般情况下，我们称x86_64架构的Ubuntu环境为host，而在host上执行spike后所虚拟出来的RISC-V环境，则被称为target。RISC-V交叉编译器的构建（build）、安装过程如下：

● 第一步，安装依赖库

RISC-V交叉编译器的构建需要一些本地支撑软件包，可使用以下命令安装：

`$ sudo apt-get install autoconf automake autotools-dev curl libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev device-tree-compiler`

● 第二步，获取RISC-V交叉编译器的源代码

有两种方式获得RISC-V交叉编译器的源代码：一种是通过源代码仓库获取，使用以下命令：

`$ git clone --recursive https://github.com/riscv/riscv-gnu-toolchain.git`

但由于RISC-V交叉编译器的仓库包含了Qemu模拟器的代码，下载后的目录占用的磁盘空间大小约为4.8GB，（从国内下载）整体下载所需的时间较长。为了方便国内用户，我们提供了另一种方式就是通过百度云盘获取源代码压缩包，链接和提取码如下：

`链接: https://pan.baidu.com/s/1cMGt0zWhRidnw7vNUGcZhg 提取码: qbjh`

从百度云盘下载RISCV-packages/riscv-gnu-toolchain-clean.tar.gz文件（大小为2.7GB），再在Ubuntu环境下解压这个.tar.gz文件，采用如下命令行：

`$ tar xf  riscv-gnu-toolchain-clean.tar.gz`

之后就能够看到和进入当前目录下的riscv-gnu-toolchain文件夹了。

● 第三步，构建（build）RISC-V交叉编译器

`$ cd riscv-gnu-toolchain`

`$ ./configure --prefix=[your.RISCV.install.path]`

`$ make`

以上命令中，[your.RISCV.install.path]指向的是你的RISC-V交叉编译器安装目录。如果安装是你home目录下的一个子目录（如~/riscv-install-dir），则最后的make install无需sudoer权限。但如果安装目录是系统目录（如/opt/riscv-install-dir），则需要sudoer权限（即在make install命令前加上sudo）。

● 第四步，设置环境变量

`$ export RISCV=[your.RISCV.install.path]`

`$ export PATH=$PATH:$RISCV/bin`

以上命令设置了RISCV环境变量，指向在第三步中的安装目录，并且将交叉编译器的可执行文件所在的目录加入到了系统路径中。这样，我们就可以在PKE的工作目录调用RISC-V交叉编译器所包含的工具软件了。

**建议读者将以上两个export命令，加入到 `~/.bashrc`，`~/.profile`或 `/etc/profile`文件的末尾**。这样，每次重新启动（并打开终端程序）后，系统会自动设置这两个环境变量，而不用每次都要手动输入以上命令。

● 第五步，安装spike模拟器

接下来，安装spkie模拟器。首先取得spike的源代码，有两个途径：一个是从github代码仓库中获取：

`$ git clone https://github.com/riscv/riscv-isa-sim.git`

也可以从百度云盘中下载spike-riscv-isa-sim.tar.gz文件（约4.2MB），然后用tar命令解压缩。百度云盘的地址，以及tar命令解压缩可以参考2.1.1节RISC-V交叉编译器的安装过程。获得spike源代码或解压后，将在本地目录看到一个riscv-isa-sim目录。

接下来构建（build）spike，并安装：

`$ cd riscv-isa-sim`

`$ ./configure --prefix=$RISCV`

`$ make`

`$ make install`

在以上命令中，我们假设RISCV环境变量已经指向了RISC-V交叉编译器的安装目录，即[your.RISCV.install.path]。

`<a name="subsec_docker"></a>`

**方案三：安装Docker桌面环境**

* 第一步，安装Docker

  * **Ubuntu**

  对于 **x86_64** 架构，安装具体的过程可以参考[这篇](https://blog.csdn.net/magic_ll/article/details/139985543?spm=1001.2014.3001.5506)文章。对于 **arm64** 架构，安装具体的过程可以参考[这篇](https://blog.csdn.net/sglin123/article/details/139754107?ops_request_misc=&request_id=&biz_id=102&utm_term=arm64%E5%AE%89%E8%A3%85docker&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-139754107.142^v100^pc_search_result_base8&spm=1018.2226.3001.4187)文章。

  * **Windows**

  Windows 版本的 Docker 安装可以参考[这篇](https://blog.csdn.net/Liuj666/article/details/126099982?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522EC5F862A-6D9C-439D-96AB-6CB77A783F13%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=EC5F862A-6D9C-439D-96AB-6CB77A783F13&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-126099982-null-null.142^v100^pc_search_result_base8&utm_term=windows%E5%AE%89%E8%A3%85docker&spm=1018.2226.3001.4187)文章。

  * **macOS**

  macOS 版本的 Docker 安装可以参考[这篇](https://blog.csdn.net/weixin_41860471/article/details/135048312?ops_request_misc=%257B%2522request%255Fid%2522%253A%25220015913A-5C64-4FD0-A192-3686BF8FE2C4%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=0015913A-5C64-4FD0-A192-3686BF8FE2C4&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-4-135048312-null-null.142^v100^pc_search_result_base8&utm_term=macos%E5%AE%89%E8%A3%85docker&spm=1018.2226.3001.4187)文章。
* 第二步，拉取镜像

  * **x86_64/amd64 版本镜像**

  *Dockerhub 镜像源*

  `$ docker pull docker.io/tjr9098/amd64_pke_mirrors:1.0`

  *阿里云镜像源*

  `$ docker pull crpi-vycj2ba2y82yi8d0.cn-hangzhou.personal.cr.aliyuncs.com/pke_mirrors/amd64_pke_mirrors:1.0`

  * **arm64 版本镜像**

  *Dockerhub 镜像源*

  `$ docker pull docker.io/tjr9098/arm64_pke_mirrors:1.0`

  *阿里云镜像源*

  `$ docker pull crpi-vycj2ba2y82yi8d0.cn-hangzhou.personal.cr.aliyuncs.com/pke_mirrors/arm64_pke_mirrors:1.0`
* 第三步，运行镜像

`$ docker run -it --name pke_mirror IMAGE:1.0`

    ●**`IMAGE`** 是镜像名称，名称可通过 `$ docker images`查看。

    ●**`-it`**: 交互式运行容器，分配一个伪终端。

    ●**`--name`**: 为容器命名，便于后续使用。

若出现下图所示则成功进入容器。

<img src="pictures/image1-docker.png" alt="image1-docker" style="zoom:80%;" />

* 第四步，验证环境

接下来切换到pke实验代码目录，并构造

`# cd riscv-pke`

`# make`

若得到以下输出则可正常进行实验

```bash
# make
compiling util/snprintf.c
compiling util/string.c
linking  obj/util.a ...
Util lib has been build into "obj/util.a"
compiling spike_interface/spike_file.c
compiling spike_interface/spike_htif.c
compiling spike_interface/spike_utils.c
compiling spike_interface/dts_parse.c
compiling spike_interface/spike_memory.c
linking  obj/spike_interface.a ...
Spike lib has been build into "obj/spike_interface.a"
compiling kernel/syscall.c
compiling kernel/kernel.c
compiling kernel/elf.c
compiling kernel/process.c
compiling kernel/strap.c
compiling kernel/machine/minit.c
compiling kernel/strap_vector.S
compiling kernel/machine/mentry.S
linking obj/riscv-pke ...
PKE core has been built into "obj/riscv-pke"
compiling user/app_helloworld.c
compiling user/user_lib.c
linking obj/app_helloworld ...
User app has been built into "obj/app_helloworld"
```

* 再次进入容器

  * 第一步，启动容器

  `$ docker container start pke_mirror`

  * 第二步，进入容器

  `$ docker exec -it pke_mirror /bin/bash`

  若出现下图所示则成功进入容器。

  <img src="pictures/image2-docker.png" alt="image2-docker" style="zoom:80%;" />

  **注意：以上代码默认容器名称为 `pke_mirror`，请读者根据自己的容器名称做修改**。

`<a name="subsec_educoder"></a>`

### 2.1.3 头歌平台

为支撑《操作系统原理》课程的教学，PKE系列实验在[头歌平台（educoder.net）](https://www.educoder.net/)上进行了部署。

<img src="pictures/educoder_1.png" alt="educoder_step1" style="zoom:80%;" />

读者可以在头歌平台的主页上，通过选择左上角的“实践课程”入口，然后在搜索框中输入“基于RISCV的操作系统实验”找到PKE实验对应的实践课程：

<img src="pictures/educoder_2.png" alt="educoder_step2" style="zoom:80%;" />

进入课程后，可以先阅读课程实验说明，翻到课程下的实验后就可以通过点击“开始实战”来进入实验了。

<img src="pictures/educoder_3.png" alt="educoder_step3" style="zoom:80%;" />

进入实战后，会看到如下的界面：

<img src="pictures/educoder_4.png" alt="educoder_step4" style="zoom:80%;" />

以上界面中，左侧显示的是实验的任务描述以及参考文献链接。右侧是做实验要用到（修改）的源代码文件，在右侧界面的顶端，有一个“命令行”标签，点击进入后可进入命令行模式；顶端的右侧的文件夹图标点击后会出现整个文件目录树，读者可以通过目录树选择要阅读的文件；最后，在右下角有个“评测”按钮，读者可以在完成实验后点击该按钮对自己的实验结果进行评测。

**实际上，读者看到的这个界面背后是一个Linux docker虚拟机，在该虚拟机中已经安装好了实验所需要的环境**。包括各种开发工具包、RISC-V交叉编译器等；而评测按钮按下后，会触发虚拟机中的构建（build）脚本，该脚本会自动对代码树进行构建、执行给定的用户程序（如lab1_1中的user/app_helloworld）、最后根据输出的内容对实验结果的正确性进行判断。

**然而，由于在头歌平台上的课程，每个实验都对应了一个独立的代码仓库，所以从一个实验到另一个实验的切换无法通过 `git merge`命令实现对上一个实验的实验结果的继承**。学生在进行实验切换时，需要将之前实验的代码手动地贴入下一个实验的对应文件中。这一点，我们在课程的“课程须知”中做了说明。一个比较好的方法，是采用类似Notepad++这样类型的软件，记录每个实验的答案（包括答案所在的文件），并在切换到下一个实验时，将答案拷贝粘贴到对应文件。

考虑到头歌平台的用户身份（主要有老师和学生）的不同，以及老师开课和学生选课的过程的不同，头歌平台给出了[教师使用手册](pictures/EduCoder平台（教师）使用手册v5.0.0_20200924.pdf)和[学生使用手册](pictures/EduCoder平台（学生）使用手册v5.0.0_20200924.pdf)。**简单来说，为了将头歌平台上的PKE实验课程应用到《操作系统原理》课程教学，教师需要在头歌平台上建立一个教学课堂，并在该教学课堂中选用“基于RISCV的操作系统实验”课程，将该实验课程中的实验有选择性的导入到所建立的教学课堂中，并最后吧所建立的教学课堂的邀请码发给学生**。对于学生而言，只需要用老师所给的邀请码在头歌平台上加入教学课堂即可。以上过程都可以通过头歌主页右上侧的“+”符号（将鼠标移动到该符号上）对应的弹出菜单中选择即可。

“基于RISCV的操作系统实验”课程的设计，充分考虑到了实验的模块化以及学制安排的问题。课程共设计了15个实验，其中包含9个基础实验和6个挑战实验（随着时间推移，这个列表可能会进一步增加），实验间的关联见[PKE实验的组成](#pke_experiemnts)。**教师开设教学课堂的时候，可以根据学生的水平、教学预期按需选择不同的实验内容，分期分批地给学生安排实验任务**。

`<a name="preparecode"></a>`

## 2.2 riscv-pke（实验）代码的获取

以下讨论，我们假设读者是使用的Ubuntu/openEuler操作系统，且已按照[2.1.2](#subsec_softwarepackages)的说明安装了需要的工具软件。对于头歌平台而言，代码已经部署到实验环境，读者可以通过头歌网站界面或者界面上的“命令行”选项看到实验代码，所以头歌平台用户无须考虑代码获取环节。

#### 代码获取

在Ubuntu/WSL/openEuler操作系统，可以通过以下命令下载riscv-pke的实验代码：

（克隆代码仓库）

```bash
`$ git clone https://gitee.com/hustos/riscv-pke.git
Cloning into 'riscv-pke-prerelease'...
remote: Enumerating objects: 195, done.
remote: Counting objects: 100% (195/195), done.
remote: Compressing objects: 100% (195/195), done.
remote: Total 227 (delta 107), reused 1 (delta 0), pack-reused 32
Receiving objects: 100% (227/227), 64.49 KiB | 335.00 KiB/s, done.
Resolving deltas: 100% (107/107), done.`
```

克隆完成后，将在当前目录应该能看到riscv-pke-prerelease目录。这时，可以到riscv-pke目录下查看文件结构，例如：

`$ cd riscv-pke`

切换到lab1_1_syscall分支（因为lab1_1_syscall是默认分支，这里也可以不切换）
`$ git checkout lab1_1_syscall`

`$ tree -L 3`

（将看到如下输出）

```
.
├── LICENSE.txt
├── Makefile
├── README.md
├── kernel
│   ├── config.h
│   ├── elf.c
│   ├── elf.h
│   ├── kernel.c
│   ├── kernel.lds
│   ├── machine
│   │   ├── mentry.S
│   │   └── minit.c
│   ├── process.c
│   ├── process.h
│   ├── riscv.h
│   ├── strap.c
│   ├── strap.h
│   ├── strap_vector.S
│   ├── syscall.c
│   └── syscall.h
├── spike_interface
│   ├── atomic.h
│   ├── dts_parse.c
│   ├── dts_parse.h
│   ├── spike_file.c
│   ├── spike_file.h
│   ├── spike_htif.c
│   ├── spike_htif.h
│   ├── spike_memory.c
│   ├── spike_memory.h
│   ├── spike_utils.c
│   └── spike_utils.h
├── user
│   ├── app_helloworld.c
│   ├── user.lds
│   ├── user_lib.c
│   └── user_lib.h
└── util
    ├── functions.h
    ├── load_store.S
    ├── snprintf.c
    ├── snprintf.h
    ├── string.c
    ├── string.h
    └── types.h
```

在代码的根目录有以下文件：

- Makefile文件，它是make命令即将使用的“自动化编译”脚本；
- LICENSE.txt文件，即riscv-pke的版权文件，里面包含了所有参与开发的人员信息。riscv-pke是开源软件，你可以以任意方式自由地使用，前提是使用时包含LICENSE.txt文件即可；
- README.md文件，一个简要的英文版代码说明。

另外是一些子目录，其中：

- kernel目录包含了riscv-pke的内核部分代码；
- spike_interface目录是riscv-pke内核与spike模拟器的接口代码（如设备树DTB、主机设备接口HTIF等），用于接口的初始化和调用；
- user目录包含了实验给定应用（如lab1_1中的app_helloworld.c），以及用户态的程序库文件（如lab1_1中的user_lib.c）；
- util目录包含了一些内核和用户程序公用的代码，如字符串处理（string.c），类型定义（types.h）等。

#### 环境验证

对于Ubuntu/WSL/openEuler用户（对于头歌用户，可以通过选择“命令行”标签，进入shell环境、进入提示的代码路径，开始构造过程），可以在代码的根目录（进入riscv-pke-prerelease子目录后）输入以下构造命令，应看到如下输出：

```bash
$ make
compiling util/snprintf.c
compiling util/string.c
linking  obj/util.a ...
Util lib has been build into "obj/util.a"
compiling spike_interface/dts_parse.c
compiling spike_interface/spike_htif.c
compiling spike_interface/spike_utils.c
compiling spike_interface/spike_file.c
compiling spike_interface/spike_memory.c
linking  obj/spike_interface.a ...
Spike lib has been build into "obj/spike_interface.a"
compiling kernel/syscall.c
compiling kernel/elf.c
compiling kernel/process.c
compiling kernel/strap.c
compiling kernel/kernel.c
compiling kernel/machine/minit.c
compiling kernel/strap_vector.S
compiling kernel/machine/mentry.S
linking obj/riscv-pke ...
PKE core has been built into "obj/riscv-pke"
compiling user/app_helloworld.c
compiling user/user_lib.c
linking obj/app_helloworld ...
User app has been built into "obj/app_helloworld"
```

如果环境安装不对（如缺少必要的支撑软件包），以上构造过程可能会在中间报错，如果碰到报错情况，请回到[2.2](#environments)的环境安装过程检查实验环境的正确性。

构造完成后，在代码根目录会出现一个obj子目录，该子目录包含了构造过程中所生成的所有对象文件（.o）、编译依赖文件（.d）、静态库（.a）文件，和最终目标ELF文件（如./obj/riscv-pke和./obj/app_helloworld）。

这时，我们可以尝试借助riscv-pke内核运行app_helloworld的“Hello world!”程序：

```bash
$ spike ./obj/riscv-pke ./obj/app_helloworld
In m_start, hartid:0
HTIF is available!
(Emulated) memory size: 2048 MB
Enter supervisor mode...
Application: ./obj/app_helloworld
Application program entry point (virtual address): 0x0000000081000000
Switching to user mode...
call do_syscall to accomplish the syscall and lab1_1 here.

System is shutting down with exit code -1.
```

如果能看到以上输出，riscv-pke的代码获取（和验证）就已经完成，可以开始实验了。

`<a name="pke_experiemnts"></a>`

## 2.3 PKE实验的组成

对于《操作系统原理》课堂来说，PKE实验由3组基础实验以及基础试验后的挑战实验组成（见图1.2）：

- **对于基础实验而言**，第一组基础实验重点涉及系统调用、异常和外部中断的知识；第二组基础实验重点涉及主存管理方面的知识；第三组实验重点涉及进程管理方面的知识。基础实验部分实验指导文档较为详细，学生（读者）需要填写的代码量很小，可以看作是“阅读理解+填空”题，涉及的知识也非常基础。
- **对于挑战实验而言**，每一组实验的挑战实验都可以理解为在该组实验上的挑战性内容，只给出题目（应用程序）和预期的效果（需要做到的目标）。这一部分实验指导文档只给出大的方向，需要学生（读者）查阅和理解较多课外内容，为实现预期的效果需要填写的代码量也较大，可以看作是“作文题”。

如图1.2所示，**基础实验部分存在继承性**，即学生（读者）需要按照实线箭头的顺序依次完成实验，是不可跳跃的！这是因为PKE的实验设计，后一个实验依赖前一个实验的答案，在开始后一个实验前需要先将之前的答案继承下来（通过git commit/git merge命令）。

**而挑战部分的实验只依赖于每组实验的最后一个实验**，例如，如果学生（读者）完成了实验1的3个基础试验后，就可以开始做实验1的挑战实验了，不需要等到完成所有基础实验再开始。

<img src="pictures/experiment_organization_withlab4.png" alt="experiment_organization" style="zoom:100%;" />

图1.2 PKE实验的组织结构

**PKE实验即可用于自学目的，也可以用于教学目的**。对于自学的读者，可以完全按照[PKE文档](https://gitee.com/hustos/pke-doc)，以及在gitee的代码仓库中所获得的代码进行；出于教学目的，所有的PKE实验，我们都在[头歌平台](https://www.educoder.net/)进行了部署，实验结果的检测全部在头歌平台上进行，实验完成后头歌平台会生成实验情况的简报。教师可以根据简报，对学生的实验完成情况给出具体的分数。

考虑到《操作系统原理》课堂的实验安排，很多学校（例如华中科技大学计算机学院）是将实验分成了两部分：课内实验和课程设计。如果采纳PKE实验，根据学生的水平，我们给出两个方案：

**方案一（学生其他课程负担较重，或不希望实验太难的情况）：**

- 课内实验包括所有的基础实验；
- 课程设计学生可在每组实验中，选择（学生自选）一个挑战实验。

这种情况，对于课内实验我们的建议是：每个基础实验=30分，每组实验总分90分；3组实验取平均分（总分仍然是90）后，总分加上实验报告的10分，等于课内实验的总分数；对于课程设计我们的建议是：每个挑战实验=30分，3个挑战实验总分求和（总分90），最后加上实验报告的10分，等于课程设计的总分数。

**方案二（学生平均能力较强，且希望实验分数有区分度的情况）：**

- 课内实验包括所有的基础实验，外加基础实验的1个（学生自选）挑战实验；
- 课程设计学生可在每组实验中，选择之前未完成的一个挑战实验。

这种情况，对于课内实验我们的建议是：每个基础实验=20分，外加一个挑战实验，每组实验总分是3*20+30=90分；3组实验取平均分（总分仍然是90）后，总分加上实验报告的10分，等于课内实验的总分数；对于课程设计我们的建议是：每个挑战实验=30分，3个挑战实验总分求和（总分90），最后加上实验报告的10分，等于课程设计的总分数。
