GAS LISTING var_def.s 			page 1


   1              	.section .data
   2 0000 61626331 	msg:	.ascii "abc", "123", "\n\0"
   2      32330A00 
   3              			.equ  con, 0x500
   4 0008 AAFF     	a1:	    .word  0xffaa
   5              	mark	= 0x100
   6 000a 62040000 	b1:	    .long 1122
   7 000e 01026202 	buf:	.byte  1, 2, 'b', 2, 'b', '1', '2', '3', 1
   7      62313233 
   7      01
   8 0017 0A000000 	d1:     .long b1
   9              	len     = . - msg
  10              	
  11              	
  12              	
  13              	.section .text
  14              	;.global _start
  15              	;_start:
  16 0000 B8010000 	    movl $1, %eax
  16      00
  17 0005 BB000000 	    movl $0, %ebx
  17      00
  18 000a CD80     	    int $0x80
GAS LISTING var_def.s 			page 2


DEFINED SYMBOLS
           var_def.s:2      .data:0000000000000000 msg
           var_def.s:3      *ABS*:0000000000000500 con
           var_def.s:4      .data:0000000000000008 a1
           var_def.s:5      *ABS*:0000000000000100 mark
           var_def.s:6      .data:000000000000000a b1
           var_def.s:7      .data:000000000000000e buf
           var_def.s:8      .data:0000000000000017 d1
           var_def.s:9      *ABS*:000000000000001b len
           var_def.s:15     .text:0000000000000000 _start

NO UNDEFINED SYMBOLS
