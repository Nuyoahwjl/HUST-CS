# AL, 用作ASCII字符转化
# BX, 用作循环移位
# CH, 用作循环计数
# EDI, 用作输出串变址

.section    .data
buf1:   .word 0x1234
buf2:   .byte 0, 0, 0, 0, '\n'

.section    .text
.global     _start
_start:
    mov buf1, %bx
    mov $4, %ch
    mov $0, %edi
loop1:    
    rol $4, %bx
    mov %bl, %al
    and $0xf, %al
    add $0x30, %al
    cmp $0x3a, %al
    jl out1
    add $7, %al
out1:
    mov %al, buf2(%edi)
    inc %edi
    dec %ch
    jnz loop1

    mov $4, %eax
    mov $1, %ebx
    mov $buf2, %ecx
    mov $5, %edx
    int $0x80

    mov $1, %eax
    mov $0, %ebx
    int $0x80
