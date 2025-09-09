.section    .data


.section    .text
.global     _start
_start:
    mov     $0x1234, %ax
    mov     $0x5678, %bx
    mov     $0x9abc, %cx

    push    %ax
    push    %bx
    push    %cx
    pop     %dx
    pop     %ax

    mov     $1, %eax
    mov     $0, %ebx
    int $0x80
