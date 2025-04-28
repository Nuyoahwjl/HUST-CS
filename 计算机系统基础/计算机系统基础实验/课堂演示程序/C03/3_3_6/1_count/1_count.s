# CX, 1的个数计数
# AX, 用作循环移位

.section    .data
y:      .word 0x55
count:  .word 0

.section    .text
.global     _start
_start:
    mov $0, %cx
    mov y, %ax
loop2:
    test $0xffff, %ax
    jz exit
    jns loop1
    inc %cx
loop1:
    shl $1, %ax
    jmp loop2
exit:
    mov %cx, count           

    mov $1, %eax
    mov $0, %ebx
    int $0x80
