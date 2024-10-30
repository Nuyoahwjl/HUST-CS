.section .data
sum:  .word 0

.section .text
.globl _start
_start:
    movl $50, %ecx
    movl $0, %eax
    movl $1, %ebx
next:
    addl %ebx, %eax
    incl %ebx
    incl %ebx
    decl %ecx    
    jne next
    movl %eax, sum
    movl $1, %eax
    movl $0, %ebx
    int $0x80
