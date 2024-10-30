.section .data
msg:	.ascii "abc", "123", "\n\0"
		.equ  con, 0x500
a1:	    .word  0xffaa
mark	= 0x100
b1:	    .long 1122
buf:	.byte  1, 2, 3, '1', 'a'
d1:     .long b1
len     = . - msg



.section .text
.global  _start
_start:
    movl $a1, %eax
    mov  $a1, %ebx
    mov  $5, (%eax)

    addl $a1, %eax
    add $a1, %ebx
    addl $a1, (%eax)

    movw  $0x89ab, a1

    movl $1, %eax
    movl $0, %ebx
    int $0x80
