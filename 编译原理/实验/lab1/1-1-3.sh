 #!/bin/bash

# 用arm-linux-gnueabihf-gcc 将iplusf.c编译成arm汇编代码iplusf.arm.s
arm-linux-gnueabihf-gcc -S iplusf.c -o iplusf.arm.s
# 再次用arm-linux-gnueabihf-gcc 汇编iplusf.arm.s，同时连接SysY2022的运行时库sylib.a，生成arm的可执行代码iplusf.arm
arm-linux-gnueabihf-gcc iplusf.arm.s sylib.a -o iplusf.arm
# 用qemu-arm运行iplusf.arm
qemu-arm -L /usr/arm-linux-gnueabihf/ iplusf.arm
