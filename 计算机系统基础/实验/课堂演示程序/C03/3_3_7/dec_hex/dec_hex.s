.section    .data
    in_buf: .byte 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    out_buf:    .byte 0, 0, 0, 0, 0, 0, 0, 0, '\n'
    lf:     .byte '\n'    

.section    .text
.global     _start
_start:
    call    decibin
    call    binihex
    mov     $1, %eax
    mov     $0, %ebx
    int     $0x80

.type       decibin, @function
decibin:
    mov     $3, %eax
    mov     $0, %ebx
    mov     $in_buf, %ecx
    mov     $10, %edx
    int     $0x80

    mov     %eax, %ecx
    mov     $0, %eax
    mov     $0, %esi
    dec     %ecx
next_1:
    movzxb  in_buf(%esi), %ebx
    sub     $'0', %ebx
    cmp     $9, %ebx
    jg      err_1
    cmp     $0, %ebx
    jl      err_1
    mov     $10, %edx
    mul     %edx
    add     %ebx, %eax
    inc     %esi
    dec     %ecx
    jnz     next_1
    jmp     exit_1     
err_1:
    mov     $0, %eax    
exit_1:
    ret

.type       binihex, @function
binihex:
    mov     $8, %ecx
    mov     $0, %edi
next_2:
    rol     $4, %eax
    mov     %al, %dl
    and     $0xf, %dl
    add     $'0', %dl
    cmp     $0x3a, %dl
    jl      label_2
    add     $7, %dl
label_2:
    mov     %dl, out_buf(%edi)
    inc     %edi
    dec     %ecx
    jnz     next_2

    mov     $4, %eax
    mov     $1, %ebx
    mov     $out_buf, %ecx
    mov     $9, %edx
    int     $0x80   
    ret
