.section  .data
helloworld:
    .ascii "hello world\n\0"

.section  .text
.global  _start
_start:
    pushl   $helloworld
    call    printf

    pushll  $0
    call    exit
