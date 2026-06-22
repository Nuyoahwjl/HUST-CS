 #!/bin/bash

# 请用 用clang编译器把bar.c“翻译”成优化的(优化级别O2)armv7架构，linux系统，符合gnueabihf嵌入式二进制接口规则，
# 并支持arm硬浮点的汇编代码（程序中并没有浮点数）。汇编代码文件名为bar.clang.arm.s
# 请使用恰当的编译选项以完成上述任务：
clang -O2 -S -target armv7-linux-gnueabihf bar.c -o bar.clang.arm.s

