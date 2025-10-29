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
    movl $a1, %eax
    movl .size a1, %ebx 

    movl $1, %eax
    movl $0, %ebx
    int $0x80
