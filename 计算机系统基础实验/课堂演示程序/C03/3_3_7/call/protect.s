.section  .data

.section  .text
.global  _start
_start:
    mov     $0x935a,  %eax
    mov     $0x44bc,  %ebx

    call    f_1

    mov     $1, %eax
    mov     $0, %ebx
    int     $0x80

.type   f_1, @function
f_1:
    push   %eax
    push   %ebx

    add   %ebx,  %eax
    mov   $0,  %ebx
    
    pop   %ebx
    pop   %eax
    ret
