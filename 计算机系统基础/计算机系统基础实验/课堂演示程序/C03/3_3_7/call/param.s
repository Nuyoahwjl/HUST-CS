.section  .data
x:  .long  0x1234
y:  .long  0x5000

.section  .text
.global  _start
_start:

	pushl	x
	pushl	y
	call	f_sub

    mov     $1, %eax
    mov     $0, %ebx
    int     $0x80

.type   f_sub, @function
f_sub:
	mov	%esp, %ebp
	mov	4(%ebp), %ebx
	mov	8(%ebp), %eax
	add	%ebx, %eax
	ret
