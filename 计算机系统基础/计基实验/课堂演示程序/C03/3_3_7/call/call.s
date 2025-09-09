.section  .data
v1: .long 0x0

.section  .text
.global  _start
_start:
    call    sort

    movl    $sort, v1
    call    *v1
    
    mov     $v1, %ebx
    call    *(%ebx)

    mov     $1, %eax
    mov     $0, %ebx
    int     $0x80

.type   sort, @function
sort:
    ret
