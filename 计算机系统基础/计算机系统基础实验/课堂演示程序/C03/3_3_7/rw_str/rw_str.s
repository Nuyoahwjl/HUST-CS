.section    .data
str:    .fill 100, 1, 0

.section    .text
.global _start
_start:
    mov $3, %eax
    mov $0, %ebx
    mov $str, %ecx
    mov $100, %edx
    int $0x80
    
    mov %eax, %edx
    mov $4, %eax
    mov $str, %ecx
    mov $1, %ebx
    int $0x80

    movl $1, %eax
    movl $0, %ebx
    int $0x80
    