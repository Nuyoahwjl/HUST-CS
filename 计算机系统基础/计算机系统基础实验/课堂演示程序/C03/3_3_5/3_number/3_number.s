# 3_number.s
.section    .data
    buf:    .word 12, 12, 12
    r1:     .byte 0, '\n'

.section    .text
.global     _start
_start:
    mov $0, %dl
    mov buf, %ax
    cmp buf + 2, %ax
    jnz l1
    inc %dl
l1:
    mov buf + 2, %bx
    cmp buf + 4, %bx
    jz  l2
    cmp buf + 4, %ax
    jnz l3     
l2:
    inc %dl
l3:
    add $0x30, %dl
    mov %dl, r1
    
    mov $4, %eax
    mov $1, %ebx
    mov $r1, %ecx
    mov $2, %edx
    int $0x80

    mov $1, %eax
    mov $0, %ebx
    int $0x80
