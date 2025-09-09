.section .data
msg:	.ascii "abc", "123", "\n\0"
		.equ  con, 0x500
a1:	    .word  0x77aa
mark	= 0x100
b1:	    .long 1122
buf:	.byte  1, 2, 3, '1', 'a'
d1:     .long b1
len     = . - msg



.section .text
.global  _start
_start:
    mov     a1, %al
    movb    a1, %al
    mov     a1, %ax
    #movb    a1, %ax
    movw    a1, %ax
    mov     a1, %eax
    mov     (%eax), %eax
    mov     $12, a1

    addl    %dx, %ax
    
    inc     %al
    incb    %al
    inc     a1

    movsx a1, %eax
    movzx a1, %ebx
    movswl a1, %eax
    movzwl a1, %ebx
    movsx %bx, %eax

    movl $1, %eax
    movl $0, %ebx
    int $0x80
