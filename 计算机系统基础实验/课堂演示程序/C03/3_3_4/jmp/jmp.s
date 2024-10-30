.section  .data
v1: .long 0x0

.section  .text
.global  _start
_start:
    jmp     l1
l1:
    mov     $l2, %eax
    jmp     *%eax
l2:
    movl    $l3, v1
    mov     $v1, %ebx
    jmp     *(%ebx)
l3:
    mov $1, %eax
    mov $0, %ebx
    int $0x80
