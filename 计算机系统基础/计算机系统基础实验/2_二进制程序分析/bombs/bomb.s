
bomb:     file format elf32-i386


Disassembly of section .init:

00001000 <_init>:
    1000:	f3 0f 1e fb          	endbr32 
    1004:	53                   	push   %ebx
    1005:	83 ec 08             	sub    $0x8,%esp
    1008:	e8 33 02 00 00       	call   1240 <__x86.get_pc_thunk.bx>
    100d:	81 c3 57 4f 00 00    	add    $0x4f57,%ebx
    1013:	8b 83 88 00 00 00    	mov    0x88(%ebx),%eax
    1019:	85 c0                	test   %eax,%eax
    101b:	74 02                	je     101f <_init+0x1f>
    101d:	ff d0                	call   *%eax
    101f:	83 c4 08             	add    $0x8,%esp
    1022:	5b                   	pop    %ebx
    1023:	c3                   	ret    

Disassembly of section .plt:

00001030 <strcmp@plt-0x10>:
    1030:	ff b3 04 00 00 00    	pushl  0x4(%ebx)
    1036:	ff a3 08 00 00 00    	jmp    *0x8(%ebx)
    103c:	00 00                	add    %al,(%eax)
	...

00001040 <strcmp@plt>:
    1040:	ff a3 0c 00 00 00    	jmp    *0xc(%ebx)
    1046:	68 00 00 00 00       	push   $0x0
    104b:	e9 e0 ff ff ff       	jmp    1030 <_init+0x30>

00001050 <__libc_start_main@plt>:
    1050:	ff a3 10 00 00 00    	jmp    *0x10(%ebx)
    1056:	68 08 00 00 00       	push   $0x8
    105b:	e9 d0 ff ff ff       	jmp    1030 <_init+0x30>

00001060 <read@plt>:
    1060:	ff a3 14 00 00 00    	jmp    *0x14(%ebx)
    1066:	68 10 00 00 00       	push   $0x10
    106b:	e9 c0 ff ff ff       	jmp    1030 <_init+0x30>

00001070 <fflush@plt>:
    1070:	ff a3 18 00 00 00    	jmp    *0x18(%ebx)
    1076:	68 18 00 00 00       	push   $0x18
    107b:	e9 b0 ff ff ff       	jmp    1030 <_init+0x30>

00001080 <fgets@plt>:
    1080:	ff a3 1c 00 00 00    	jmp    *0x1c(%ebx)
    1086:	68 20 00 00 00       	push   $0x20
    108b:	e9 a0 ff ff ff       	jmp    1030 <_init+0x30>

00001090 <signal@plt>:
    1090:	ff a3 20 00 00 00    	jmp    *0x20(%ebx)
    1096:	68 28 00 00 00       	push   $0x28
    109b:	e9 90 ff ff ff       	jmp    1030 <_init+0x30>

000010a0 <sleep@plt>:
    10a0:	ff a3 24 00 00 00    	jmp    *0x24(%ebx)
    10a6:	68 30 00 00 00       	push   $0x30
    10ab:	e9 80 ff ff ff       	jmp    1030 <_init+0x30>

000010b0 <alarm@plt>:
    10b0:	ff a3 28 00 00 00    	jmp    *0x28(%ebx)
    10b6:	68 38 00 00 00       	push   $0x38
    10bb:	e9 70 ff ff ff       	jmp    1030 <_init+0x30>

000010c0 <__stack_chk_fail@plt>:
    10c0:	ff a3 2c 00 00 00    	jmp    *0x2c(%ebx)
    10c6:	68 40 00 00 00       	push   $0x40
    10cb:	e9 60 ff ff ff       	jmp    1030 <_init+0x30>

000010d0 <strcpy@plt>:
    10d0:	ff a3 30 00 00 00    	jmp    *0x30(%ebx)
    10d6:	68 48 00 00 00       	push   $0x48
    10db:	e9 50 ff ff ff       	jmp    1030 <_init+0x30>

000010e0 <getenv@plt>:
    10e0:	ff a3 34 00 00 00    	jmp    *0x34(%ebx)
    10e6:	68 50 00 00 00       	push   $0x50
    10eb:	e9 40 ff ff ff       	jmp    1030 <_init+0x30>

000010f0 <puts@plt>:
    10f0:	ff a3 38 00 00 00    	jmp    *0x38(%ebx)
    10f6:	68 58 00 00 00       	push   $0x58
    10fb:	e9 30 ff ff ff       	jmp    1030 <_init+0x30>

00001100 <__memmove_chk@plt>:
    1100:	ff a3 3c 00 00 00    	jmp    *0x3c(%ebx)
    1106:	68 60 00 00 00       	push   $0x60
    110b:	e9 20 ff ff ff       	jmp    1030 <_init+0x30>

00001110 <exit@plt>:
    1110:	ff a3 40 00 00 00    	jmp    *0x40(%ebx)
    1116:	68 68 00 00 00       	push   $0x68
    111b:	e9 10 ff ff ff       	jmp    1030 <_init+0x30>

00001120 <strlen@plt>:
    1120:	ff a3 44 00 00 00    	jmp    *0x44(%ebx)
    1126:	68 70 00 00 00       	push   $0x70
    112b:	e9 00 ff ff ff       	jmp    1030 <_init+0x30>

00001130 <write@plt>:
    1130:	ff a3 48 00 00 00    	jmp    *0x48(%ebx)
    1136:	68 78 00 00 00       	push   $0x78
    113b:	e9 f0 fe ff ff       	jmp    1030 <_init+0x30>

00001140 <__isoc99_sscanf@plt>:
    1140:	ff a3 4c 00 00 00    	jmp    *0x4c(%ebx)
    1146:	68 80 00 00 00       	push   $0x80
    114b:	e9 e0 fe ff ff       	jmp    1030 <_init+0x30>

00001150 <fopen@plt>:
    1150:	ff a3 50 00 00 00    	jmp    *0x50(%ebx)
    1156:	68 88 00 00 00       	push   $0x88
    115b:	e9 d0 fe ff ff       	jmp    1030 <_init+0x30>

00001160 <__errno_location@plt>:
    1160:	ff a3 54 00 00 00    	jmp    *0x54(%ebx)
    1166:	68 90 00 00 00       	push   $0x90
    116b:	e9 c0 fe ff ff       	jmp    1030 <_init+0x30>

00001170 <__printf_chk@plt>:
    1170:	ff a3 58 00 00 00    	jmp    *0x58(%ebx)
    1176:	68 98 00 00 00       	push   $0x98
    117b:	e9 b0 fe ff ff       	jmp    1030 <_init+0x30>

00001180 <socket@plt>:
    1180:	ff a3 5c 00 00 00    	jmp    *0x5c(%ebx)
    1186:	68 a0 00 00 00       	push   $0xa0
    118b:	e9 a0 fe ff ff       	jmp    1030 <_init+0x30>

00001190 <__fprintf_chk@plt>:
    1190:	ff a3 60 00 00 00    	jmp    *0x60(%ebx)
    1196:	68 a8 00 00 00       	push   $0xa8
    119b:	e9 90 fe ff ff       	jmp    1030 <_init+0x30>

000011a0 <gethostbyname@plt>:
    11a0:	ff a3 64 00 00 00    	jmp    *0x64(%ebx)
    11a6:	68 b0 00 00 00       	push   $0xb0
    11ab:	e9 80 fe ff ff       	jmp    1030 <_init+0x30>

000011b0 <strtol@plt>:
    11b0:	ff a3 68 00 00 00    	jmp    *0x68(%ebx)
    11b6:	68 b8 00 00 00       	push   $0xb8
    11bb:	e9 70 fe ff ff       	jmp    1030 <_init+0x30>

000011c0 <connect@plt>:
    11c0:	ff a3 6c 00 00 00    	jmp    *0x6c(%ebx)
    11c6:	68 c0 00 00 00       	push   $0xc0
    11cb:	e9 60 fe ff ff       	jmp    1030 <_init+0x30>

000011d0 <close@plt>:
    11d0:	ff a3 70 00 00 00    	jmp    *0x70(%ebx)
    11d6:	68 c8 00 00 00       	push   $0xc8
    11db:	e9 50 fe ff ff       	jmp    1030 <_init+0x30>

000011e0 <__ctype_b_loc@plt>:
    11e0:	ff a3 74 00 00 00    	jmp    *0x74(%ebx)
    11e6:	68 d0 00 00 00       	push   $0xd0
    11eb:	e9 40 fe ff ff       	jmp    1030 <_init+0x30>

000011f0 <__sprintf_chk@plt>:
    11f0:	ff a3 78 00 00 00    	jmp    *0x78(%ebx)
    11f6:	68 d8 00 00 00       	push   $0xd8
    11fb:	e9 30 fe ff ff       	jmp    1030 <_init+0x30>

Disassembly of section .plt.got:

00001200 <__cxa_finalize@plt>:
    1200:	ff a3 84 00 00 00    	jmp    *0x84(%ebx)
    1206:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00001210 <_start>:
    1210:	f3 0f 1e fb          	endbr32 
    1214:	31 ed                	xor    %ebp,%ebp
    1216:	5e                   	pop    %esi
    1217:	89 e1                	mov    %esp,%ecx
    1219:	83 e4 f0             	and    $0xfffffff0,%esp
    121c:	50                   	push   %eax
    121d:	54                   	push   %esp
    121e:	52                   	push   %edx
    121f:	e8 18 00 00 00       	call   123c <_start+0x2c>
    1224:	81 c3 40 4d 00 00    	add    $0x4d40,%ebx
    122a:	6a 00                	push   $0x0
    122c:	6a 00                	push   $0x0
    122e:	51                   	push   %ecx
    122f:	56                   	push   %esi
    1230:	ff b3 94 00 00 00    	pushl  0x94(%ebx)
    1236:	e8 15 fe ff ff       	call   1050 <__libc_start_main@plt>
    123b:	f4                   	hlt    
    123c:	8b 1c 24             	mov    (%esp),%ebx
    123f:	c3                   	ret    

00001240 <__x86.get_pc_thunk.bx>:
    1240:	8b 1c 24             	mov    (%esp),%ebx
    1243:	c3                   	ret    
    1244:	66 90                	xchg   %ax,%ax
    1246:	66 90                	xchg   %ax,%ax
    1248:	66 90                	xchg   %ax,%ax
    124a:	66 90                	xchg   %ax,%ax
    124c:	66 90                	xchg   %ax,%ax
    124e:	66 90                	xchg   %ax,%ax

00001250 <deregister_tm_clones>:
    1250:	e8 e4 00 00 00       	call   1339 <__x86.get_pc_thunk.dx>
    1255:	81 c2 0f 4d 00 00    	add    $0x4d0f,%edx
    125b:	8d 8a bc 03 00 00    	lea    0x3bc(%edx),%ecx
    1261:	8d 82 bc 03 00 00    	lea    0x3bc(%edx),%eax
    1267:	39 c8                	cmp    %ecx,%eax
    1269:	74 1d                	je     1288 <deregister_tm_clones+0x38>
    126b:	8b 82 7c 00 00 00    	mov    0x7c(%edx),%eax
    1271:	85 c0                	test   %eax,%eax
    1273:	74 13                	je     1288 <deregister_tm_clones+0x38>
    1275:	55                   	push   %ebp
    1276:	89 e5                	mov    %esp,%ebp
    1278:	83 ec 14             	sub    $0x14,%esp
    127b:	51                   	push   %ecx
    127c:	ff d0                	call   *%eax
    127e:	83 c4 10             	add    $0x10,%esp
    1281:	c9                   	leave  
    1282:	c3                   	ret    
    1283:	8d 74 26 00          	lea    0x0(%esi,%eiz,1),%esi
    1287:	90                   	nop
    1288:	c3                   	ret    
    1289:	8d b4 26 00 00 00 00 	lea    0x0(%esi,%eiz,1),%esi

00001290 <register_tm_clones>:
    1290:	e8 a4 00 00 00       	call   1339 <__x86.get_pc_thunk.dx>
    1295:	81 c2 cf 4c 00 00    	add    $0x4ccf,%edx
    129b:	55                   	push   %ebp
    129c:	89 e5                	mov    %esp,%ebp
    129e:	53                   	push   %ebx
    129f:	8d 8a bc 03 00 00    	lea    0x3bc(%edx),%ecx
    12a5:	8d 82 bc 03 00 00    	lea    0x3bc(%edx),%eax
    12ab:	83 ec 04             	sub    $0x4,%esp
    12ae:	29 c8                	sub    %ecx,%eax
    12b0:	89 c3                	mov    %eax,%ebx
    12b2:	c1 e8 1f             	shr    $0x1f,%eax
    12b5:	c1 fb 02             	sar    $0x2,%ebx
    12b8:	01 d8                	add    %ebx,%eax
    12ba:	d1 f8                	sar    %eax
    12bc:	74 14                	je     12d2 <register_tm_clones+0x42>
    12be:	8b 92 98 00 00 00    	mov    0x98(%edx),%edx
    12c4:	85 d2                	test   %edx,%edx
    12c6:	74 0a                	je     12d2 <register_tm_clones+0x42>
    12c8:	83 ec 08             	sub    $0x8,%esp
    12cb:	50                   	push   %eax
    12cc:	51                   	push   %ecx
    12cd:	ff d2                	call   *%edx
    12cf:	83 c4 10             	add    $0x10,%esp
    12d2:	8b 5d fc             	mov    -0x4(%ebp),%ebx
    12d5:	c9                   	leave  
    12d6:	c3                   	ret    
    12d7:	8d b4 26 00 00 00 00 	lea    0x0(%esi,%eiz,1),%esi
    12de:	66 90                	xchg   %ax,%ax

000012e0 <__do_global_dtors_aux>:
    12e0:	f3 0f 1e fb          	endbr32 
    12e4:	55                   	push   %ebp
    12e5:	89 e5                	mov    %esp,%ebp
    12e7:	53                   	push   %ebx
    12e8:	e8 53 ff ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    12ed:	81 c3 77 4c 00 00    	add    $0x4c77,%ebx
    12f3:	83 ec 04             	sub    $0x4,%esp
    12f6:	80 bb bc 03 00 00 00 	cmpb   $0x0,0x3bc(%ebx)
    12fd:	75 27                	jne    1326 <__do_global_dtors_aux+0x46>
    12ff:	8b 83 84 00 00 00    	mov    0x84(%ebx),%eax
    1305:	85 c0                	test   %eax,%eax
    1307:	74 11                	je     131a <__do_global_dtors_aux+0x3a>
    1309:	83 ec 0c             	sub    $0xc,%esp
    130c:	ff b3 a0 00 00 00    	pushl  0xa0(%ebx)
    1312:	e8 e9 fe ff ff       	call   1200 <__cxa_finalize@plt>
    1317:	83 c4 10             	add    $0x10,%esp
    131a:	e8 31 ff ff ff       	call   1250 <deregister_tm_clones>
    131f:	c6 83 bc 03 00 00 01 	movb   $0x1,0x3bc(%ebx)
    1326:	8b 5d fc             	mov    -0x4(%ebp),%ebx
    1329:	c9                   	leave  
    132a:	c3                   	ret    
    132b:	8d 74 26 00          	lea    0x0(%esi,%eiz,1),%esi
    132f:	90                   	nop

00001330 <frame_dummy>:
    1330:	f3 0f 1e fb          	endbr32 
    1334:	e9 57 ff ff ff       	jmp    1290 <register_tm_clones>

00001339 <__x86.get_pc_thunk.dx>:
    1339:	8b 14 24             	mov    (%esp),%edx
    133c:	c3                   	ret    

0000133d <main>:
    133d:	8d 4c 24 04          	lea    0x4(%esp),%ecx
    1341:	83 e4 f0             	and    $0xfffffff0,%esp
    1344:	ff 71 fc             	pushl  -0x4(%ecx)
    1347:	55                   	push   %ebp
    1348:	89 e5                	mov    %esp,%ebp
    134a:	56                   	push   %esi
    134b:	53                   	push   %ebx
    134c:	51                   	push   %ecx
    134d:	83 ec 0c             	sub    $0xc,%esp
    1350:	e8 eb fe ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    1355:	81 c3 0f 4c 00 00    	add    $0x4c0f,%ebx
    135b:	8b 01                	mov    (%ecx),%eax
    135d:	8b 71 04             	mov    0x4(%ecx),%esi
    1360:	83 f8 01             	cmp    $0x1,%eax
    1363:	0f 84 13 01 00 00    	je     147c <main+0x13f>
    1369:	83 f8 02             	cmp    $0x2,%eax
    136c:	0f 85 3c 01 00 00    	jne    14ae <main+0x171>
    1372:	83 ec 08             	sub    $0x8,%esp
    1375:	8d 83 a4 d0 ff ff    	lea    -0x2f5c(%ebx),%eax
    137b:	50                   	push   %eax
    137c:	ff 76 04             	pushl  0x4(%esi)
    137f:	e8 cc fd ff ff       	call   1150 <fopen@plt>
    1384:	89 83 c0 03 00 00    	mov    %eax,0x3c0(%ebx)
    138a:	83 c4 10             	add    $0x10,%esp
    138d:	85 c0                	test   %eax,%eax
    138f:	0f 84 fa 00 00 00    	je     148f <main+0x152>
    1395:	e8 d8 06 00 00       	call   1a72 <initialize_bomb>
    139a:	83 ec 0c             	sub    $0xc,%esp
    139d:	8d 83 28 d1 ff ff    	lea    -0x2ed8(%ebx),%eax
    13a3:	50                   	push   %eax
    13a4:	e8 47 fd ff ff       	call   10f0 <puts@plt>
    13a9:	8d 83 64 d1 ff ff    	lea    -0x2e9c(%ebx),%eax
    13af:	89 04 24             	mov    %eax,(%esp)
    13b2:	e8 39 fd ff ff       	call   10f0 <puts@plt>
    13b7:	e8 f5 07 00 00       	call   1bb1 <read_line>
    13bc:	89 04 24             	mov    %eax,(%esp)
    13bf:	e8 09 01 00 00       	call   14cd <phase_1>
    13c4:	e8 07 09 00 00       	call   1cd0 <phase_defused>
    13c9:	8d 83 90 d1 ff ff    	lea    -0x2e70(%ebx),%eax
    13cf:	89 04 24             	mov    %eax,(%esp)
    13d2:	e8 19 fd ff ff       	call   10f0 <puts@plt>
    13d7:	e8 d5 07 00 00       	call   1bb1 <read_line>
    13dc:	89 04 24             	mov    %eax,(%esp)
    13df:	e8 1b 01 00 00       	call   14ff <phase_2>
    13e4:	e8 e7 08 00 00       	call   1cd0 <phase_defused>
    13e9:	8d 83 dd d0 ff ff    	lea    -0x2f23(%ebx),%eax
    13ef:	89 04 24             	mov    %eax,(%esp)
    13f2:	e8 f9 fc ff ff       	call   10f0 <puts@plt>
    13f7:	e8 b5 07 00 00       	call   1bb1 <read_line>
    13fc:	89 04 24             	mov    %eax,(%esp)
    13ff:	e8 77 01 00 00       	call   157b <phase_3>
    1404:	e8 c7 08 00 00       	call   1cd0 <phase_defused>
    1409:	8d 83 fb d0 ff ff    	lea    -0x2f05(%ebx),%eax
    140f:	89 04 24             	mov    %eax,(%esp)
    1412:	e8 d9 fc ff ff       	call   10f0 <puts@plt>
    1417:	e8 95 07 00 00       	call   1bb1 <read_line>
    141c:	89 04 24             	mov    %eax,(%esp)
    141f:	e8 58 02 00 00       	call   167c <phase_4>
    1424:	e8 a7 08 00 00       	call   1cd0 <phase_defused>
    1429:	8d 83 bc d1 ff ff    	lea    -0x2e44(%ebx),%eax
    142f:	89 04 24             	mov    %eax,(%esp)
    1432:	e8 b9 fc ff ff       	call   10f0 <puts@plt>
    1437:	e8 75 07 00 00       	call   1bb1 <read_line>
    143c:	89 04 24             	mov    %eax,(%esp)
    143f:	e8 bb 02 00 00       	call   16ff <phase_5>
    1444:	e8 87 08 00 00       	call   1cd0 <phase_defused>
    1449:	8d 83 0a d1 ff ff    	lea    -0x2ef6(%ebx),%eax
    144f:	89 04 24             	mov    %eax,(%esp)
    1452:	e8 99 fc ff ff       	call   10f0 <puts@plt>
    1457:	e8 55 07 00 00       	call   1bb1 <read_line>
    145c:	89 04 24             	mov    %eax,(%esp)
    145f:	e8 f5 02 00 00       	call   1759 <phase_6>
    1464:	e8 67 08 00 00       	call   1cd0 <phase_defused>
    1469:	83 c4 10             	add    $0x10,%esp
    146c:	b8 00 00 00 00       	mov    $0x0,%eax
    1471:	8d 65 f4             	lea    -0xc(%ebp),%esp
    1474:	59                   	pop    %ecx
    1475:	5b                   	pop    %ebx
    1476:	5e                   	pop    %esi
    1477:	5d                   	pop    %ebp
    1478:	8d 61 fc             	lea    -0x4(%ecx),%esp
    147b:	c3                   	ret    
    147c:	8b 83 8c 00 00 00    	mov    0x8c(%ebx),%eax
    1482:	8b 00                	mov    (%eax),%eax
    1484:	89 83 c0 03 00 00    	mov    %eax,0x3c0(%ebx)
    148a:	e9 06 ff ff ff       	jmp    1395 <main+0x58>
    148f:	ff 76 04             	pushl  0x4(%esi)
    1492:	ff 36                	pushl  (%esi)
    1494:	8d 83 a6 d0 ff ff    	lea    -0x2f5a(%ebx),%eax
    149a:	50                   	push   %eax
    149b:	6a 01                	push   $0x1
    149d:	e8 ce fc ff ff       	call   1170 <__printf_chk@plt>
    14a2:	c7 04 24 08 00 00 00 	movl   $0x8,(%esp)
    14a9:	e8 62 fc ff ff       	call   1110 <exit@plt>
    14ae:	83 ec 04             	sub    $0x4,%esp
    14b1:	ff 36                	pushl  (%esi)
    14b3:	8d 83 c3 d0 ff ff    	lea    -0x2f3d(%ebx),%eax
    14b9:	50                   	push   %eax
    14ba:	6a 01                	push   $0x1
    14bc:	e8 af fc ff ff       	call   1170 <__printf_chk@plt>
    14c1:	c7 04 24 08 00 00 00 	movl   $0x8,(%esp)
    14c8:	e8 43 fc ff ff       	call   1110 <exit@plt>

000014cd <phase_1>:
    14cd:	53                   	push   %ebx
    14ce:	83 ec 10             	sub    $0x10,%esp
    14d1:	e8 6a fd ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    14d6:	81 c3 8e 4a 00 00    	add    $0x4a8e,%ebx
    14dc:	8d 83 e0 d1 ff ff    	lea    -0x2e20(%ebx),%eax
    14e2:	50                   	push   %eax
    14e3:	ff 74 24 1c          	pushl  0x1c(%esp)
    14e7:	e8 2e 05 00 00       	call   1a1a <strings_not_equal>
    14ec:	83 c4 10             	add    $0x10,%esp
    14ef:	85 c0                	test   %eax,%eax
    14f1:	75 05                	jne    14f8 <phase_1+0x2b>
    14f3:	83 c4 08             	add    $0x8,%esp
    14f6:	5b                   	pop    %ebxe
    14f7:	c3                   	ret    
    14f8:	e8 35 06 00 00       	call   1b32 <explode_bomb>
    14fd:	eb f4                	jmp    14f3 <phase_1+0x26>

000014ff <phase_2>:
    14ff:	57                   	push   %edi
    1500:	56                   	push   %esi
    1501:	53                   	push   %ebx
    1502:	83 ec 28             	sub    $0x28,%esp
    1505:	e8 36 fd ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    150a:	81 c3 5a 4a 00 00    	add    $0x4a5a,%ebx
    1510:	65 a1 14 00 00 00    	mov    %gs:0x14,%eax
    1516:	89 44 24 24          	mov    %eax,0x24(%esp)
    151a:	31 c0                	xor    %eax,%eax
    151c:	8d 44 24 0c          	lea    0xc(%esp),%eax
    1520:	50                   	push   %eax
    1521:	ff 74 24 3c          	pushl  0x3c(%esp)
    1525:	e8 3d 06 00 00       	call   1b67 <read_six_numbers>
    152a:	83 c4 10             	add    $0x10,%esp
    152d:	83 7c 24 04 00       	cmpl   $0x0,0x4(%esp)
    1532:	75 07                	jne    153b <phase_2+0x3c>
    1534:	83 7c 24 08 01       	cmpl   $0x1,0x8(%esp)
    1539:	74 05                	je     1540 <phase_2+0x41>
    153b:	e8 f2 05 00 00       	call   1b32 <explode_bomb>
    1540:	8d 74 24 04          	lea    0x4(%esp),%esi
    1544:	8d 7c 24 14          	lea    0x14(%esp),%edi
    1548:	eb 07                	jmp    1551 <phase_2+0x52>
    154a:	83 c6 04             	add    $0x4,%esi
    154d:	39 fe                	cmp    %edi,%esi
    154f:	74 11                	je     1562 <phase_2+0x63>
    1551:	8b 46 04             	mov    0x4(%esi),%eax
    1554:	03 06                	add    (%esi),%eax
    1556:	39 46 08             	cmp    %eax,0x8(%esi)
    1559:	74 ef                	je     154a <phase_2+0x4b>
    155b:	e8 d2 05 00 00       	call   1b32 <explode_bomb>
    1560:	eb e8                	jmp    154a <phase_2+0x4b>
    1562:	8b 44 24 1c          	mov    0x1c(%esp),%eax
    1566:	65 2b 05 14 00 00 00 	sub    %gs:0x14,%eax
    156d:	75 07                	jne    1576 <phase_2+0x77>
    156f:	83 c4 20             	add    $0x20,%esp
    1572:	5b                   	pop    %ebx
    1573:	5e                   	pop    %esi
    1574:	5f                   	pop    %edi
    1575:	c3                   	ret    
    1576:	e8 c5 13 00 00       	call   2940 <__stack_chk_fail_local>

0000157b <phase_3>:
    157b:	53                   	push   %ebx
    157c:	83 ec 18             	sub    $0x18,%esp
    157f:	e8 bc fc ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    1584:	81 c3 e0 49 00 00    	add    $0x49e0,%ebx
    158a:	65 a1 14 00 00 00    	mov    %gs:0x14,%eax
    1590:	89 44 24 0c          	mov    %eax,0xc(%esp)
    1594:	31 c0                	xor    %eax,%eax
    1596:	8d 44 24 08          	lea    0x8(%esp),%eax
    159a:	50                   	push   %eax
    159b:	8d 44 24 08          	lea    0x8(%esp),%eax
    159f:	50                   	push   %eax
    15a0:	8d 83 cb d3 ff ff    	lea    -0x2c35(%ebx),%eax
    15a6:	50                   	push   %eax
    15a7:	ff 74 24 2c          	pushl  0x2c(%esp)
    15ab:	e8 90 fb ff ff       	call   1140 <__isoc99_sscanf@plt>
    15b0:	83 c4 10             	add    $0x10,%esp
    15b3:	83 f8 01             	cmp    $0x1,%eax
    15b6:	7e 16                	jle    15ce <phase_3+0x53>
    15b8:	83 7c 24 04 07       	cmpl   $0x7,0x4(%esp)
    15bd:	77 5d                	ja     161c <.L18+0x7>
    15bf:	8b 44 24 04          	mov    0x4(%esp),%eax
    15c3:	89 da                	mov    %ebx,%edx
    15c5:	03 94 83 5c d2 ff ff 	add    -0x2da4(%ebx,%eax,4),%edx
    15cc:	ff e2                	jmp    *%edx
    15ce:	e8 5f 05 00 00       	call   1b32 <explode_bomb>
    15d3:	eb e3                	jmp    15b8 <phase_3+0x3d>

000015d5 <.L25>:
    15d5:	b8 97 00 00 00       	mov    $0x97,%eax
    15da:	39 44 24 08          	cmp    %eax,0x8(%esp)
    15de:	75 4f                	jne    162f <.L29+0x7>
    15e0:	8b 44 24 0c          	mov    0xc(%esp),%eax
    15e4:	65 2b 05 14 00 00 00 	sub    %gs:0x14,%eax
    15eb:	75 49                	jne    1636 <.L29+0xe>
    15ed:	83 c4 18             	add    $0x18,%esp
    15f0:	5b                   	pop    %ebx
    15f1:	c3                   	ret    

000015f2 <.L24>:
    15f2:	b8 de 01 00 00       	mov    $0x1de,%eax
    15f7:	eb e1                	jmp    15da <.L25+0x5>

000015f9 <.L23>:
    15f9:	b8 35 00 00 00       	mov    $0x35,%eax
    15fe:	eb da                	jmp    15da <.L25+0x5>

00001600 <.L22>:
    1600:	b8 ab 02 00 00       	mov    $0x2ab,%eax
    1605:	eb d3                	jmp    15da <.L25+0x5>

00001607 <.L21>:
    1607:	b8 e7 00 00 00       	mov    $0xe7,%eax
    160c:	eb cc                	jmp    15da <.L25+0x5>

0000160e <.L20>:
    160e:	b8 0a 01 00 00       	mov    $0x10a,%eax
    1613:	eb c5                	jmp    15da <.L25+0x5>

00001615 <.L18>:
    1615:	b8 81 00 00 00       	mov    $0x81,%eax
    161a:	eb be                	jmp    15da <.L25+0x5>
    161c:	e8 11 05 00 00       	call   1b32 <explode_bomb>
    1621:	b8 00 00 00 00       	mov    $0x0,%eax
    1626:	eb b2                	jmp    15da <.L25+0x5>

00001628 <.L29>:
    1628:	b8 b4 02 00 00       	mov    $0x2b4,%eax
    162d:	eb ab                	jmp    15da <.L25+0x5>
    162f:	e8 fe 04 00 00       	call   1b32 <explode_bomb>
    1634:	eb aa                	jmp    15e0 <.L25+0xb>
    1636:	e8 05 13 00 00       	call   2940 <__stack_chk_fail_local>

0000163b <func4>:
    163b:	57                   	push   %edi
    163c:	56                   	push   %esi
    163d:	53                   	push   %ebx
    163e:	8b 5c 24 10          	mov    0x10(%esp),%ebx
    1642:	8b 7c 24 14          	mov    0x14(%esp),%edi
    1646:	b8 00 00 00 00       	mov    $0x0,%eax
    164b:	85 db                	test   %ebx,%ebx
    164d:	7e 29                	jle    1678 <func4+0x3d>
    164f:	89 f8                	mov    %edi,%eax
    1651:	83 fb 01             	cmp    $0x1,%ebx
    1654:	74 22                	je     1678 <func4+0x3d>
    1656:	83 ec 08             	sub    $0x8,%esp
    1659:	57                   	push   %edi
    165a:	8d 43 ff             	lea    -0x1(%ebx),%eax
    165d:	50                   	push   %eax
    165e:	e8 d8 ff ff ff       	call   163b <func4>
    1663:	83 c4 08             	add    $0x8,%esp
    1666:	8d 34 38             	lea    (%eax,%edi,1),%esi
    1669:	57                   	push   %edi
    166a:	83 eb 02             	sub    $0x2,%ebx
    166d:	53                   	push   %ebx
    166e:	e8 c8 ff ff ff       	call   163b <func4>
    1673:	83 c4 10             	add    $0x10,%esp
    1676:	01 f0                	add    %esi,%eax
    1678:	5b                   	pop    %ebx
    1679:	5e                   	pop    %esi
    167a:	5f                   	pop    %edi
    167b:	c3                   	ret    

0000167c <phase_4>:
    167c:	53                   	push   %ebx
    167d:	83 ec 18             	sub    $0x18,%esp
    1680:	e8 bb fb ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    1685:	81 c3 df 48 00 00    	add    $0x48df,%ebx
    168b:	65 a1 14 00 00 00    	mov    %gs:0x14,%eax
    1691:	89 44 24 0c          	mov    %eax,0xc(%esp)
    1695:	31 c0                	xor    %eax,%eax
    1697:	8d 44 24 04          	lea    0x4(%esp),%eax
    169b:	50                   	push   %eax
    169c:	8d 44 24 0c          	lea    0xc(%esp),%eax
    16a0:	50                   	push   %eax
    16a1:	8d 83 cb d3 ff ff    	lea    -0x2c35(%ebx),%eax
    16a7:	50                   	push   %eax
    16a8:	ff 74 24 2c          	pushl  0x2c(%esp)
    16ac:	e8 8f fa ff ff       	call   1140 <__isoc99_sscanf@plt>
    16b1:	83 c4 10             	add    $0x10,%esp
    16b4:	83 f8 02             	cmp    $0x2,%eax
    16b7:	75 0c                	jne    16c5 <phase_4+0x49>
    16b9:	8b 44 24 04          	mov    0x4(%esp),%eax
    16bd:	83 e8 02             	sub    $0x2,%eax
    16c0:	83 f8 02             	cmp    $0x2,%eax
    16c3:	76 05                	jbe    16ca <phase_4+0x4e>
    16c5:	e8 68 04 00 00       	call   1b32 <explode_bomb>
    16ca:	83 ec 08             	sub    $0x8,%esp
    16cd:	ff 74 24 0c          	pushl  0xc(%esp)
    16d1:	6a 09                	push   $0x9
    16d3:	e8 63 ff ff ff       	call   163b <func4>
    16d8:	83 c4 10             	add    $0x10,%esp
    16db:	39 44 24 08          	cmp    %eax,0x8(%esp)
    16df:	75 12                	jne    16f3 <phase_4+0x77>
    16e1:	8b 44 24 0c          	mov    0xc(%esp),%eax
    16e5:	65 2b 05 14 00 00 00 	sub    %gs:0x14,%eax
    16ec:	75 0c                	jne    16fa <phase_4+0x7e>
    16ee:	83 c4 18             	add    $0x18,%esp
    16f1:	5b                   	pop    %ebx
    16f2:	c3                   	ret    
    16f3:	e8 3a 04 00 00       	call   1b32 <explode_bomb>
    16f8:	eb e7                	jmp    16e1 <phase_4+0x65>
    16fa:	e8 41 12 00 00       	call   2940 <__stack_chk_fail_local>

000016ff <phase_5>:
    16ff:	57                   	push   %edi
    1700:	56                   	push   %esi
    1701:	53                   	push   %ebx
    1702:	e8 39 fb ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    1707:	81 c3 5d 48 00 00    	add    $0x485d,%ebx
    170d:	8b 74 24 10          	mov    0x10(%esp),%esi
    1711:	83 ec 0c             	sub    $0xc,%esp
    1714:	56                   	push   %esi
    1715:	e8 e2 02 00 00       	call   19fc <string_length>
    171a:	83 c4 10             	add    $0x10,%esp
    171d:	83 f8 06             	cmp    $0x6,%eax
    1720:	75 29                	jne    174b <phase_5+0x4c>
    1722:	89 f0                	mov    %esi,%eax
    1724:	83 c6 06             	add    $0x6,%esi
    1727:	b9 00 00 00 00       	mov    $0x0,%ecx
    172c:	8d bb 7c d2 ff ff    	lea    -0x2d84(%ebx),%edi
    1732:	0f b6 10             	movzbl (%eax),%edx
    1735:	83 e2 0f             	and    $0xf,%edx
    1738:	03 0c 97             	add    (%edi,%edx,4),%ecx
    173b:	83 c0 01             	add    $0x1,%eax
    173e:	39 f0                	cmp    %esi,%eax
    1740:	75 f0                	jne    1732 <phase_5+0x33>
    1742:	83 f9 35             	cmp    $0x35,%ecx
    1745:	75 0b                	jne    1752 <phase_5+0x53>
    1747:	5b                   	pop    %ebx
    1748:	5e                   	pop    %esi
    1749:	5f                   	pop    %edi
    174a:	c3                   	ret    
    174b:	e8 e2 03 00 00       	call   1b32 <explode_bomb>
    1750:	eb d0                	jmp    1722 <phase_5+0x23>
    1752:	e8 db 03 00 00       	call   1b32 <explode_bomb>
    1757:	eb ee                	jmp    1747 <phase_5+0x48>

00001759 <phase_6>:
    1759:	55                   	push   %ebp
    175a:	57                   	push   %edi
    175b:	56                   	push   %esi
    175c:	53                   	push   %ebx
    175d:	83 ec 74             	sub    $0x74,%esp
    1760:	e8 db fa ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    1765:	81 c3 ff 47 00 00    	add    $0x47ff,%ebx
    176b:	65 a1 14 00 00 00    	mov    %gs:0x14,%eax
    1771:	89 44 24 64          	mov    %eax,0x64(%esp)
    1775:	31 c0                	xor    %eax,%eax
    1777:	8d 44 24 34          	lea    0x34(%esp),%eax
    177b:	89 c7                	mov    %eax,%edi
    177d:	89 44 24 24          	mov    %eax,0x24(%esp)
    1781:	50                   	push   %eax
    1782:	ff b4 24 8c 00 00 00 	pushl  0x8c(%esp)
    1789:	e8 d9 03 00 00       	call   1b67 <read_six_numbers>
    178e:	89 7c 24 28          	mov    %edi,0x28(%esp)
    1792:	83 c4 10             	add    $0x10,%esp
    1795:	89 7c 24 10          	mov    %edi,0x10(%esp)
    1799:	c7 44 24 0c 00 00 00 	movl   $0x0,0xc(%esp)
    17a0:	00 
    17a1:	89 fd                	mov    %edi,%ebp
    17a3:	eb 23                	jmp    17c8 <phase_6+0x6f>
    17a5:	e8 88 03 00 00       	call   1b32 <explode_bomb>
    17aa:	eb 30                	jmp    17dc <phase_6+0x83>
    17ac:	83 c6 01             	add    $0x1,%esi
    17af:	83 fe 06             	cmp    $0x6,%esi
    17b2:	74 0f                	je     17c3 <phase_6+0x6a>
    17b4:	8b 44 b5 00          	mov    0x0(%ebp,%esi,4),%eax
    17b8:	39 07                	cmp    %eax,(%edi)
    17ba:	75 f0                	jne    17ac <phase_6+0x53>
    17bc:	e8 71 03 00 00       	call   1b32 <explode_bomb>
    17c1:	eb e9                	jmp    17ac <phase_6+0x53>
    17c3:	83 44 24 10 04       	addl   $0x4,0x10(%esp)
    17c8:	8b 44 24 10          	mov    0x10(%esp),%eax
    17cc:	89 c7                	mov    %eax,%edi
    17ce:	8b 00                	mov    (%eax),%eax
    17d0:	89 44 24 14          	mov    %eax,0x14(%esp)
    17d4:	83 e8 01             	sub    $0x1,%eax
    17d7:	83 f8 05             	cmp    $0x5,%eax
    17da:	77 c9                	ja     17a5 <phase_6+0x4c>
    17dc:	83 44 24 0c 01       	addl   $0x1,0xc(%esp)
    17e1:	8b 74 24 0c          	mov    0xc(%esp),%esi
    17e5:	83 fe 05             	cmp    $0x5,%esi
    17e8:	7e ca                	jle    17b4 <phase_6+0x5b>
    17ea:	8b 54 24 1c          	mov    0x1c(%esp),%edx
    17ee:	83 c2 18             	add    $0x18,%edx
    17f1:	b9 07 00 00 00       	mov    $0x7,%ecx
    17f6:	8b 44 24 18          	mov    0x18(%esp),%eax
    17fa:	89 ce                	mov    %ecx,%esi
    17fc:	2b 30                	sub    (%eax),%esi
    17fe:	89 30                	mov    %esi,(%eax)
    1800:	83 c0 04             	add    $0x4,%eax
    1803:	39 c2                	cmp    %eax,%edx
    1805:	75 f3                	jne    17fa <phase_6+0xa1>
    1807:	be 00 00 00 00       	mov    $0x0,%esi
    180c:	89 f7                	mov    %esi,%edi
    180e:	8b 4c b4 2c          	mov    0x2c(%esp,%esi,4),%ecx
    1812:	b8 01 00 00 00       	mov    $0x1,%eax
    1817:	8d 93 68 01 00 00    	lea    0x168(%ebx),%edx
    181d:	83 f9 01             	cmp    $0x1,%ecx
    1820:	7e 0a                	jle    182c <phase_6+0xd3>
    1822:	8b 52 08             	mov    0x8(%edx),%edx
    1825:	83 c0 01             	add    $0x1,%eax
    1828:	39 c8                	cmp    %ecx,%eax
    182a:	75 f6                	jne    1822 <phase_6+0xc9>
    182c:	89 54 bc 44          	mov    %edx,0x44(%esp,%edi,4)
    1830:	83 c6 01             	add    $0x1,%esi
    1833:	83 fe 06             	cmp    $0x6,%esi
    1836:	75 d4                	jne    180c <phase_6+0xb3>
    1838:	8b 74 24 44          	mov    0x44(%esp),%esi
    183c:	8b 44 24 48          	mov    0x48(%esp),%eax
    1840:	89 46 08             	mov    %eax,0x8(%esi)
    1843:	8b 54 24 4c          	mov    0x4c(%esp),%edx
    1847:	89 50 08             	mov    %edx,0x8(%eax)
    184a:	8b 44 24 50          	mov    0x50(%esp),%eax
    184e:	89 42 08             	mov    %eax,0x8(%edx)
    1851:	8b 54 24 54          	mov    0x54(%esp),%edx
    1855:	89 50 08             	mov    %edx,0x8(%eax)
    1858:	8b 44 24 58          	mov    0x58(%esp),%eax
    185c:	89 42 08             	mov    %eax,0x8(%edx)
    185f:	c7 40 08 00 00 00 00 	movl   $0x0,0x8(%eax)
    1866:	bf 05 00 00 00       	mov    $0x5,%edi
    186b:	eb 08                	jmp    1875 <phase_6+0x11c>
    186d:	8b 76 08             	mov    0x8(%esi),%esi
    1870:	83 ef 01             	sub    $0x1,%edi
    1873:	74 10                	je     1885 <phase_6+0x12c>
    1875:	8b 46 08             	mov    0x8(%esi),%eax
    1878:	8b 00                	mov    (%eax),%eax
    187a:	39 06                	cmp    %eax,(%esi)
    187c:	7d ef                	jge    186d <phase_6+0x114>
    187e:	e8 af 02 00 00       	call   1b32 <explode_bomb>
    1883:	eb e8                	jmp    186d <phase_6+0x114>
    1885:	8b 44 24 5c          	mov    0x5c(%esp),%eax
    1889:	65 2b 05 14 00 00 00 	sub    %gs:0x14,%eax
    1890:	75 08                	jne    189a <phase_6+0x141>
    1892:	83 c4 6c             	add    $0x6c,%esp
    1895:	5b                   	pop    %ebx
    1896:	5e                   	pop    %esi
    1897:	5f                   	pop    %edi
    1898:	5d                   	pop    %ebp
    1899:	c3                   	ret    
    189a:	e8 a1 10 00 00       	call   2940 <__stack_chk_fail_local>

0000189f <fun7>:
    189f:	53                   	push   %ebx
    18a0:	83 ec 08             	sub    $0x8,%esp
    18a3:	8b 54 24 10          	mov    0x10(%esp),%edx
    18a7:	8b 4c 24 14          	mov    0x14(%esp),%ecx
    18ab:	85 d2                	test   %edx,%edx
    18ad:	74 3a                	je     18e9 <fun7+0x4a>
    18af:	8b 1a                	mov    (%edx),%ebx
    18b1:	39 cb                	cmp    %ecx,%ebx
    18b3:	7f 0c                	jg     18c1 <fun7+0x22>
    18b5:	b8 00 00 00 00       	mov    $0x0,%eax
    18ba:	75 18                	jne    18d4 <fun7+0x35>
    18bc:	83 c4 08             	add    $0x8,%esp
    18bf:	5b                   	pop    %ebx
    18c0:	c3                   	ret    
    18c1:	83 ec 08             	sub    $0x8,%esp
    18c4:	51                   	push   %ecx
    18c5:	ff 72 04             	pushl  0x4(%edx)
    18c8:	e8 d2 ff ff ff       	call   189f <fun7>
    18cd:	83 c4 10             	add    $0x10,%esp
    18d0:	01 c0                	add    %eax,%eax
    18d2:	eb e8                	jmp    18bc <fun7+0x1d>
    18d4:	83 ec 08             	sub    $0x8,%esp
    18d7:	51                   	push   %ecx
    18d8:	ff 72 08             	pushl  0x8(%edx)
    18db:	e8 bf ff ff ff       	call   189f <fun7>
    18e0:	83 c4 10             	add    $0x10,%esp
    18e3:	8d 44 00 01          	lea    0x1(%eax,%eax,1),%eax
    18e7:	eb d3                	jmp    18bc <fun7+0x1d>
    18e9:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    18ee:	eb cc                	jmp    18bc <fun7+0x1d>

000018f0 <secret_phase>:
    18f0:	56                   	push   %esi
    18f1:	53                   	push   %ebx
    18f2:	83 ec 04             	sub    $0x4,%esp
    18f5:	e8 46 f9 ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    18fa:	81 c3 6a 46 00 00    	add    $0x466a,%ebx
    1900:	e8 ac 02 00 00       	call   1bb1 <read_line>
    1905:	83 ec 04             	sub    $0x4,%esp
    1908:	6a 0a                	push   $0xa
    190a:	6a 00                	push   $0x0
    190c:	50                   	push   %eax
    190d:	e8 9e f8 ff ff       	call   11b0 <strtol@plt>
    1912:	89 c6                	mov    %eax,%esi
    1914:	8d 40 ff             	lea    -0x1(%eax),%eax
    1917:	83 c4 10             	add    $0x10,%esp
    191a:	3d e8 03 00 00       	cmp    $0x3e8,%eax
    191f:	77 32                	ja     1953 <secret_phase+0x63>
    1921:	83 ec 08             	sub    $0x8,%esp
    1924:	56                   	push   %esi
    1925:	8d 83 14 01 00 00    	lea    0x114(%ebx),%eax
    192b:	50                   	push   %eax
    192c:	e8 6e ff ff ff       	call   189f <fun7>
    1931:	83 c4 10             	add    $0x10,%esp
    1934:	83 f8 03             	cmp    $0x3,%eax
    1937:	75 21                	jne    195a <secret_phase+0x6a>
    1939:	83 ec 0c             	sub    $0xc,%esp
    193c:	8d 83 18 d2 ff ff    	lea    -0x2de8(%ebx),%eax
    1942:	50                   	push   %eax
    1943:	e8 a8 f7 ff ff       	call   10f0 <puts@plt>
    1948:	e8 83 03 00 00       	call   1cd0 <phase_defused>
    194d:	83 c4 14             	add    $0x14,%esp
    1950:	5b                   	pop    %ebx
    1951:	5e                   	pop    %esi
    1952:	c3                   	ret    
    1953:	e8 da 01 00 00       	call   1b32 <explode_bomb>
    1958:	eb c7                	jmp    1921 <secret_phase+0x31>
    195a:	e8 d3 01 00 00       	call   1b32 <explode_bomb>
    195f:	eb d8                	jmp    1939 <secret_phase+0x49>

00001961 <sig_handler>:
    1961:	53                   	push   %ebx
    1962:	83 ec 14             	sub    $0x14,%esp
    1965:	e8 d6 f8 ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    196a:	81 c3 fa 45 00 00    	add    $0x45fa,%ebx
    1970:	8d 83 bc d2 ff ff    	lea    -0x2d44(%ebx),%eax
    1976:	50                   	push   %eax
    1977:	e8 74 f7 ff ff       	call   10f0 <puts@plt>
    197c:	c7 04 24 03 00 00 00 	movl   $0x3,(%esp)
    1983:	e8 18 f7 ff ff       	call   10a0 <sleep@plt>
    1988:	83 c4 08             	add    $0x8,%esp
    198b:	8d 83 7e d3 ff ff    	lea    -0x2c82(%ebx),%eax
    1991:	50                   	push   %eax
    1992:	6a 01                	push   $0x1
    1994:	e8 d7 f7 ff ff       	call   1170 <__printf_chk@plt>
    1999:	83 c4 04             	add    $0x4,%esp
    199c:	8b 83 90 00 00 00    	mov    0x90(%ebx),%eax
    19a2:	ff 30                	pushl  (%eax)
    19a4:	e8 c7 f6 ff ff       	call   1070 <fflush@plt>
    19a9:	c7 04 24 01 00 00 00 	movl   $0x1,(%esp)
    19b0:	e8 eb f6 ff ff       	call   10a0 <sleep@plt>
    19b5:	8d 83 86 d3 ff ff    	lea    -0x2c7a(%ebx),%eax
    19bb:	89 04 24             	mov    %eax,(%esp)
    19be:	e8 2d f7 ff ff       	call   10f0 <puts@plt>
    19c3:	c7 04 24 10 00 00 00 	movl   $0x10,(%esp)
    19ca:	e8 41 f7 ff ff       	call   1110 <exit@plt>

000019cf <invalid_phase>:
    19cf:	53                   	push   %ebx
    19d0:	83 ec 0c             	sub    $0xc,%esp
    19d3:	e8 68 f8 ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    19d8:	81 c3 8c 45 00 00    	add    $0x458c,%ebx
    19de:	ff 74 24 14          	pushl  0x14(%esp)
    19e2:	8d 83 8e d3 ff ff    	lea    -0x2c72(%ebx),%eax
    19e8:	50                   	push   %eax
    19e9:	6a 01                	push   $0x1
    19eb:	e8 80 f7 ff ff       	call   1170 <__printf_chk@plt>
    19f0:	c7 04 24 08 00 00 00 	movl   $0x8,(%esp)
    19f7:	e8 14 f7 ff ff       	call   1110 <exit@plt>

000019fc <string_length>:
    19fc:	8b 54 24 04          	mov    0x4(%esp),%edx
    1a00:	80 3a 00             	cmpb   $0x0,(%edx)
    1a03:	74 0f                	je     1a14 <string_length+0x18>
    1a05:	b8 00 00 00 00       	mov    $0x0,%eax
    1a0a:	83 c0 01             	add    $0x1,%eax
    1a0d:	80 3c 02 00          	cmpb   $0x0,(%edx,%eax,1)
    1a11:	75 f7                	jne    1a0a <string_length+0xe>
    1a13:	c3                   	ret    
    1a14:	b8 00 00 00 00       	mov    $0x0,%eax
    1a19:	c3                   	ret    

00001a1a <strings_not_equal>:
    1a1a:	57                   	push   %edi
    1a1b:	56                   	push   %esi
    1a1c:	53                   	push   %ebx
    1a1d:	8b 5c 24 10          	mov    0x10(%esp),%ebx
    1a21:	8b 74 24 14          	mov    0x14(%esp),%esi
    1a25:	53                   	push   %ebx
    1a26:	e8 d1 ff ff ff       	call   19fc <string_length>
    1a2b:	89 c7                	mov    %eax,%edi
    1a2d:	89 34 24             	mov    %esi,(%esp)
    1a30:	e8 c7 ff ff ff       	call   19fc <string_length>
    1a35:	83 c4 04             	add    $0x4,%esp
    1a38:	89 c2                	mov    %eax,%edx
    1a3a:	b8 01 00 00 00       	mov    $0x1,%eax
    1a3f:	39 d7                	cmp    %edx,%edi
    1a41:	75 2b                	jne    1a6e <strings_not_equal+0x54>
    1a43:	0f b6 03             	movzbl (%ebx),%eax
    1a46:	84 c0                	test   %al,%al
    1a48:	74 18                	je     1a62 <strings_not_equal+0x48>
    1a4a:	38 06                	cmp    %al,(%esi)
    1a4c:	75 1b                	jne    1a69 <strings_not_equal+0x4f>
    1a4e:	83 c3 01             	add    $0x1,%ebx
    1a51:	83 c6 01             	add    $0x1,%esi
    1a54:	0f b6 03             	movzbl (%ebx),%eax
    1a57:	84 c0                	test   %al,%al
    1a59:	75 ef                	jne    1a4a <strings_not_equal+0x30>
    1a5b:	b8 00 00 00 00       	mov    $0x0,%eax
    1a60:	eb 0c                	jmp    1a6e <strings_not_equal+0x54>
    1a62:	b8 00 00 00 00       	mov    $0x0,%eax
    1a67:	eb 05                	jmp    1a6e <strings_not_equal+0x54>
    1a69:	b8 01 00 00 00       	mov    $0x1,%eax
    1a6e:	5b                   	pop    %ebx
    1a6f:	5e                   	pop    %esi
    1a70:	5f                   	pop    %edi
    1a71:	c3                   	ret    

00001a72 <initialize_bomb>:
    1a72:	53                   	push   %ebx
    1a73:	83 ec 10             	sub    $0x10,%esp
    1a76:	e8 c5 f7 ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    1a7b:	81 c3 e9 44 00 00    	add    $0x44e9,%ebx
    1a81:	8d 83 fd b9 ff ff    	lea    -0x4603(%ebx),%eax
    1a87:	50                   	push   %eax
    1a88:	6a 02                	push   $0x2
    1a8a:	e8 01 f6 ff ff       	call   1090 <signal@plt>
    1a8f:	83 c4 18             	add    $0x18,%esp
    1a92:	5b                   	pop    %ebx
    1a93:	c3                   	ret    

00001a94 <initialize_bomb_solve>:
    1a94:	c3                   	ret    

00001a95 <blank_line>:
    1a95:	57                   	push   %edi
    1a96:	56                   	push   %esi
    1a97:	53                   	push   %ebx
    1a98:	e8 a3 f7 ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    1a9d:	81 c3 c7 44 00 00    	add    $0x44c7,%ebx
    1aa3:	8b 7c 24 10          	mov    0x10(%esp),%edi
    1aa7:	0f b6 37             	movzbl (%edi),%esi
    1aaa:	89 f0                	mov    %esi,%eax
    1aac:	84 c0                	test   %al,%al
    1aae:	74 1d                	je     1acd <blank_line+0x38>
    1ab0:	e8 2b f7 ff ff       	call   11e0 <__ctype_b_loc@plt>
    1ab5:	83 c7 01             	add    $0x1,%edi
    1ab8:	89 f2                	mov    %esi,%edx
    1aba:	0f be f2             	movsbl %dl,%esi
    1abd:	8b 00                	mov    (%eax),%eax
    1abf:	f6 44 70 01 20       	testb  $0x20,0x1(%eax,%esi,2)
    1ac4:	75 e1                	jne    1aa7 <blank_line+0x12>
    1ac6:	b8 00 00 00 00       	mov    $0x0,%eax
    1acb:	eb 05                	jmp    1ad2 <blank_line+0x3d>
    1acd:	b8 01 00 00 00       	mov    $0x1,%eax
    1ad2:	5b                   	pop    %ebx
    1ad3:	5e                   	pop    %esi
    1ad4:	5f                   	pop    %edi
    1ad5:	c3                   	ret    

00001ad6 <skip>:
    1ad6:	55                   	push   %ebp
    1ad7:	57                   	push   %edi
    1ad8:	56                   	push   %esi
    1ad9:	53                   	push   %ebx
    1ada:	83 ec 0c             	sub    $0xc,%esp
    1add:	e8 5e f7 ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    1ae2:	81 c3 82 44 00 00    	add    $0x4482,%ebx
    1ae8:	8d bb c0 03 00 00    	lea    0x3c0(%ebx),%edi
    1aee:	8d b3 3c 04 00 00    	lea    0x43c(%ebx),%esi
    1af4:	83 ec 04             	sub    $0x4,%esp
    1af7:	ff 37                	pushl  (%edi)
    1af9:	6a 50                	push   $0x50
    1afb:	8b 83 2c 04 00 00    	mov    0x42c(%ebx),%eax
    1b01:	8d 04 80             	lea    (%eax,%eax,4),%eax
    1b04:	c1 e0 04             	shl    $0x4,%eax
    1b07:	01 f0                	add    %esi,%eax
    1b09:	50                   	push   %eax
    1b0a:	e8 71 f5 ff ff       	call   1080 <fgets@plt>
    1b0f:	89 c5                	mov    %eax,%ebp
    1b11:	83 c4 10             	add    $0x10,%esp
    1b14:	85 c0                	test   %eax,%eax
    1b16:	74 10                	je     1b28 <skip+0x52>
    1b18:	83 ec 0c             	sub    $0xc,%esp
    1b1b:	50                   	push   %eax
    1b1c:	e8 74 ff ff ff       	call   1a95 <blank_line>
    1b21:	83 c4 10             	add    $0x10,%esp
    1b24:	85 c0                	test   %eax,%eax
    1b26:	75 cc                	jne    1af4 <skip+0x1e>
    1b28:	89 e8                	mov    %ebp,%eax
    1b2a:	83 c4 0c             	add    $0xc,%esp
    1b2d:	5b                   	pop    %ebx
    1b2e:	5e                   	pop    %esi
    1b2f:	5f                   	pop    %edi
    1b30:	5d                   	pop    %ebp
    1b31:	c3                   	ret    

00001b32 <explode_bomb>:
    1b32:	53                   	push   %ebx
    1b33:	83 ec 14             	sub    $0x14,%esp
    1b36:	e8 05 f7 ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    1b3b:	81 c3 29 44 00 00    	add    $0x4429,%ebx
    1b41:	8d 83 9f d3 ff ff    	lea    -0x2c61(%ebx),%eax
    1b47:	50                   	push   %eax
    1b48:	e8 a3 f5 ff ff       	call   10f0 <puts@plt>
    1b4d:	8d 83 a8 d3 ff ff    	lea    -0x2c58(%ebx),%eax
    1b53:	89 04 24             	mov    %eax,(%esp)
    1b56:	e8 95 f5 ff ff       	call   10f0 <puts@plt>
    1b5b:	c7 04 24 08 00 00 00 	movl   $0x8,(%esp)
    1b62:	e8 a9 f5 ff ff       	call   1110 <exit@plt>

00001b67 <read_six_numbers>:
    1b67:	53                   	push   %ebx
    1b68:	83 ec 08             	sub    $0x8,%esp
    1b6b:	e8 d0 f6 ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    1b70:	81 c3 f4 43 00 00    	add    $0x43f4,%ebx
    1b76:	8b 44 24 14          	mov    0x14(%esp),%eax
    1b7a:	8d 50 14             	lea    0x14(%eax),%edx
    1b7d:	52                   	push   %edx
    1b7e:	8d 50 10             	lea    0x10(%eax),%edx
    1b81:	52                   	push   %edx
    1b82:	8d 50 0c             	lea    0xc(%eax),%edx
    1b85:	52                   	push   %edx
    1b86:	8d 50 08             	lea    0x8(%eax),%edx
    1b89:	52                   	push   %edx
    1b8a:	8d 50 04             	lea    0x4(%eax),%edx
    1b8d:	52                   	push   %edx
    1b8e:	50                   	push   %eax
    1b8f:	8d 83 bf d3 ff ff    	lea    -0x2c41(%ebx),%eax
    1b95:	50                   	push   %eax
    1b96:	ff 74 24 2c          	pushl  0x2c(%esp)
    1b9a:	e8 a1 f5 ff ff       	call   1140 <__isoc99_sscanf@plt>
    1b9f:	83 c4 20             	add    $0x20,%esp
    1ba2:	83 f8 05             	cmp    $0x5,%eax
    1ba5:	7e 05                	jle    1bac <read_six_numbers+0x45>
    1ba7:	83 c4 08             	add    $0x8,%esp
    1baa:	5b                   	pop    %ebx
    1bab:	c3                   	ret    
    1bac:	e8 81 ff ff ff       	call   1b32 <explode_bomb>

00001bb1 <read_line>:
    1bb1:	57                   	push   %edi
    1bb2:	56                   	push   %esi
    1bb3:	53                   	push   %ebx
    1bb4:	e8 87 f6 ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    1bb9:	81 c3 ab 43 00 00    	add    $0x43ab,%ebx
    1bbf:	e8 12 ff ff ff       	call   1ad6 <skip>
    1bc4:	85 c0                	test   %eax,%eax
    1bc6:	74 47                	je     1c0f <read_line+0x5e>
    1bc8:	8b b3 2c 04 00 00    	mov    0x42c(%ebx),%esi
    1bce:	8d 04 b6             	lea    (%esi,%esi,4),%eax
    1bd1:	c1 e0 04             	shl    $0x4,%eax
    1bd4:	8d bc 03 3c 04 00 00 	lea    0x43c(%ebx,%eax,1),%edi
    1bdb:	83 ec 0c             	sub    $0xc,%esp
    1bde:	57                   	push   %edi
    1bdf:	e8 3c f5 ff ff       	call   1120 <strlen@plt>
    1be4:	83 c4 10             	add    $0x10,%esp
    1be7:	83 f8 4e             	cmp    $0x4e,%eax
    1bea:	0f 8f a4 00 00 00    	jg     1c94 <read_line+0xe3>
    1bf0:	8d 14 b6             	lea    (%esi,%esi,4),%edx
    1bf3:	c1 e2 04             	shl    $0x4,%edx
    1bf6:	01 d0                	add    %edx,%eax
    1bf8:	c6 84 03 3b 04 00 00 	movb   $0x0,0x43b(%ebx,%eax,1)
    1bff:	00 
    1c00:	83 c6 01             	add    $0x1,%esi
    1c03:	89 b3 2c 04 00 00    	mov    %esi,0x42c(%ebx)
    1c09:	89 f8                	mov    %edi,%eax
    1c0b:	5b                   	pop    %ebx
    1c0c:	5e                   	pop    %esi
    1c0d:	5f                   	pop    %edi
    1c0e:	c3                   	ret    
    1c0f:	8d 93 c0 03 00 00    	lea    0x3c0(%ebx),%edx
    1c15:	8b 83 8c 00 00 00    	mov    0x8c(%ebx),%eax
    1c1b:	8b 00                	mov    (%eax),%eax
    1c1d:	39 02                	cmp    %eax,(%edx)
    1c1f:	74 20                	je     1c41 <read_line+0x90>
    1c21:	83 ec 0c             	sub    $0xc,%esp
    1c24:	8d 83 ef d3 ff ff    	lea    -0x2c11(%ebx),%eax
    1c2a:	50                   	push   %eax
    1c2b:	e8 b0 f4 ff ff       	call   10e0 <getenv@plt>
    1c30:	83 c4 10             	add    $0x10,%esp
    1c33:	85 c0                	test   %eax,%eax
    1c35:	74 25                	je     1c5c <read_line+0xab>
    1c37:	83 ec 0c             	sub    $0xc,%esp
    1c3a:	6a 00                	push   $0x0
    1c3c:	e8 cf f4 ff ff       	call   1110 <exit@plt>
    1c41:	83 ec 0c             	sub    $0xc,%esp
    1c44:	8d 83 d1 d3 ff ff    	lea    -0x2c2f(%ebx),%eax
    1c4a:	50                   	push   %eax
    1c4b:	e8 a0 f4 ff ff       	call   10f0 <puts@plt>
    1c50:	c7 04 24 08 00 00 00 	movl   $0x8,(%esp)
    1c57:	e8 b4 f4 ff ff       	call   1110 <exit@plt>
    1c5c:	8b 83 8c 00 00 00    	mov    0x8c(%ebx),%eax
    1c62:	8b 10                	mov    (%eax),%edx
    1c64:	8d 83 c0 03 00 00    	lea    0x3c0(%ebx),%eax
    1c6a:	89 10                	mov    %edx,(%eax)
    1c6c:	e8 65 fe ff ff       	call   1ad6 <skip>
    1c71:	85 c0                	test   %eax,%eax
    1c73:	0f 85 4f ff ff ff    	jne    1bc8 <read_line+0x17>
    1c79:	83 ec 0c             	sub    $0xc,%esp
    1c7c:	8d 83 d1 d3 ff ff    	lea    -0x2c2f(%ebx),%eax
    1c82:	50                   	push   %eax
    1c83:	e8 68 f4 ff ff       	call   10f0 <puts@plt>
    1c88:	c7 04 24 00 00 00 00 	movl   $0x0,(%esp)
    1c8f:	e8 7c f4 ff ff       	call   1110 <exit@plt>
    1c94:	83 ec 0c             	sub    $0xc,%esp
    1c97:	8d 83 fa d3 ff ff    	lea    -0x2c06(%ebx),%eax
    1c9d:	50                   	push   %eax
    1c9e:	e8 4d f4 ff ff       	call   10f0 <puts@plt>
    1ca3:	8b 83 2c 04 00 00    	mov    0x42c(%ebx),%eax
    1ca9:	8d 50 01             	lea    0x1(%eax),%edx
    1cac:	89 93 2c 04 00 00    	mov    %edx,0x42c(%ebx)
    1cb2:	6b c0 50             	imul   $0x50,%eax,%eax
    1cb5:	8d 84 03 3c 04 00 00 	lea    0x43c(%ebx,%eax,1),%eax
    1cbc:	8d b3 15 d4 ff ff    	lea    -0x2beb(%ebx),%esi
    1cc2:	b9 04 00 00 00       	mov    $0x4,%ecx
    1cc7:	89 c7                	mov    %eax,%edi
    1cc9:	f3 a5                	rep movsl %ds:(%esi),%es:(%edi)
    1ccb:	e8 62 fe ff ff       	call   1b32 <explode_bomb>

00001cd0 <phase_defused>:
    1cd0:	53                   	push   %ebx
    1cd1:	83 ec 68             	sub    $0x68,%esp
    1cd4:	e8 67 f5 ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    1cd9:	81 c3 8b 42 00 00    	add    $0x428b,%ebx
    1cdf:	65 a1 14 00 00 00    	mov    %gs:0x14,%eax
    1ce5:	89 44 24 5c          	mov    %eax,0x5c(%esp)
    1ce9:	31 c0                	xor    %eax,%eax
    1ceb:	83 bb 2c 04 00 00 06 	cmpl   $0x6,0x42c(%ebx)
    1cf2:	74 16                	je     1d0a <phase_defused+0x3a>
    1cf4:	8b 44 24 5c          	mov    0x5c(%esp),%eax
    1cf8:	65 2b 05 14 00 00 00 	sub    %gs:0x14,%eax
    1cff:	0f 85 88 00 00 00    	jne    1d8d <phase_defused+0xbd>
    1d05:	83 c4 68             	add    $0x68,%esp
    1d08:	5b                   	pop    %ebx
    1d09:	c3                   	ret    
    1d0a:	83 ec 0c             	sub    $0xc,%esp
    1d0d:	8d 44 24 18          	lea    0x18(%esp),%eax
    1d11:	50                   	push   %eax
    1d12:	8d 44 24 18          	lea    0x18(%esp),%eax
    1d16:	50                   	push   %eax
    1d17:	8d 44 24 18          	lea    0x18(%esp),%eax
    1d1b:	50                   	push   %eax
    1d1c:	8d 83 25 d4 ff ff    	lea    -0x2bdb(%ebx),%eax
    1d22:	50                   	push   %eax
    1d23:	8d 83 2c 05 00 00    	lea    0x52c(%ebx),%eax
    1d29:	50                   	push   %eax
    1d2a:	e8 11 f4 ff ff       	call   1140 <__isoc99_sscanf@plt>
    1d2f:	83 c4 20             	add    $0x20,%esp
    1d32:	83 f8 03             	cmp    $0x3,%eax
    1d35:	74 14                	je     1d4b <phase_defused+0x7b>
    1d37:	83 ec 0c             	sub    $0xc,%esp
    1d3a:	8d 83 54 d3 ff ff    	lea    -0x2cac(%ebx),%eax
    1d40:	50                   	push   %eax
    1d41:	e8 aa f3 ff ff       	call   10f0 <puts@plt>
    1d46:	83 c4 10             	add    $0x10,%esp
    1d49:	eb a9                	jmp    1cf4 <phase_defused+0x24>
    1d4b:	83 ec 08             	sub    $0x8,%esp
    1d4e:	8d 83 2e d4 ff ff    	lea    -0x2bd2(%ebx),%eax
    1d54:	50                   	push   %eax
    1d55:	8d 44 24 18          	lea    0x18(%esp),%eax
    1d59:	50                   	push   %eax
    1d5a:	e8 bb fc ff ff       	call   1a1a <strings_not_equal>
    1d5f:	83 c4 10             	add    $0x10,%esp
    1d62:	85 c0                	test   %eax,%eax
    1d64:	75 d1                	jne    1d37 <phase_defused+0x67>
    1d66:	83 ec 0c             	sub    $0xc,%esp
    1d69:	8d 83 f4 d2 ff ff    	lea    -0x2d0c(%ebx),%eax
    1d6f:	50                   	push   %eax
    1d70:	e8 7b f3 ff ff       	call   10f0 <puts@plt>
    1d75:	8d 83 1c d3 ff ff    	lea    -0x2ce4(%ebx),%eax
    1d7b:	89 04 24             	mov    %eax,(%esp)
    1d7e:	e8 6d f3 ff ff       	call   10f0 <puts@plt>
    1d83:	e8 68 fb ff ff       	call   18f0 <secret_phase>
    1d88:	83 c4 10             	add    $0x10,%esp
    1d8b:	eb aa                	jmp    1d37 <phase_defused+0x67>
    1d8d:	e8 ae 0b 00 00       	call   2940 <__stack_chk_fail_local>

00001d92 <sigalrm_handler>:
    1d92:	53                   	push   %ebx
    1d93:	83 ec 08             	sub    $0x8,%esp
    1d96:	e8 a5 f4 ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    1d9b:	81 c3 c9 41 00 00    	add    $0x41c9,%ebx
    1da1:	6a 00                	push   $0x0
    1da3:	8d 83 84 d4 ff ff    	lea    -0x2b7c(%ebx),%eax
    1da9:	50                   	push   %eax
    1daa:	6a 01                	push   $0x1
    1dac:	8b 83 80 00 00 00    	mov    0x80(%ebx),%eax
    1db2:	ff 30                	pushl  (%eax)
    1db4:	e8 d7 f3 ff ff       	call   1190 <__fprintf_chk@plt>
    1db9:	c7 04 24 01 00 00 00 	movl   $0x1,(%esp)
    1dc0:	e8 4b f3 ff ff       	call   1110 <exit@plt>

00001dc5 <rio_readlineb>:
    1dc5:	55                   	push   %ebp
    1dc6:	57                   	push   %edi
    1dc7:	56                   	push   %esi
    1dc8:	53                   	push   %ebx
    1dc9:	83 ec 1c             	sub    $0x1c,%esp
    1dcc:	e8 6f f4 ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    1dd1:	81 c3 93 41 00 00    	add    $0x4193,%ebx
    1dd7:	89 d5                	mov    %edx,%ebp
    1dd9:	83 f9 01             	cmp    $0x1,%ecx
    1ddc:	0f 86 87 00 00 00    	jbe    1e69 <rio_readlineb+0xa4>
    1de2:	89 c6                	mov    %eax,%esi
    1de4:	8d 44 0a ff          	lea    -0x1(%edx,%ecx,1),%eax
    1de8:	89 44 24 0c          	mov    %eax,0xc(%esp)
    1dec:	c7 44 24 08 01 00 00 	movl   $0x1,0x8(%esp)
    1df3:	00 
    1df4:	8d 7e 0c             	lea    0xc(%esi),%edi
    1df7:	eb 51                	jmp    1e4a <rio_readlineb+0x85>
    1df9:	e8 62 f3 ff ff       	call   1160 <__errno_location@plt>
    1dfe:	83 38 04             	cmpl   $0x4,(%eax)
    1e01:	75 50                	jne    1e53 <rio_readlineb+0x8e>
    1e03:	83 ec 04             	sub    $0x4,%esp
    1e06:	68 00 20 00 00       	push   $0x2000
    1e0b:	57                   	push   %edi
    1e0c:	ff 36                	pushl  (%esi)
    1e0e:	e8 4d f2 ff ff       	call   1060 <read@plt>
    1e13:	89 46 04             	mov    %eax,0x4(%esi)
    1e16:	83 c4 10             	add    $0x10,%esp
    1e19:	85 c0                	test   %eax,%eax
    1e1b:	78 dc                	js     1df9 <rio_readlineb+0x34>
    1e1d:	74 39                	je     1e58 <rio_readlineb+0x93>
    1e1f:	89 7e 08             	mov    %edi,0x8(%esi)
    1e22:	8b 56 08             	mov    0x8(%esi),%edx
    1e25:	0f b6 0a             	movzbl (%edx),%ecx
    1e28:	83 c2 01             	add    $0x1,%edx
    1e2b:	89 56 08             	mov    %edx,0x8(%esi)
    1e2e:	83 e8 01             	sub    $0x1,%eax
    1e31:	89 46 04             	mov    %eax,0x4(%esi)
    1e34:	83 c5 01             	add    $0x1,%ebp
    1e37:	88 4d ff             	mov    %cl,-0x1(%ebp)
    1e3a:	80 f9 0a             	cmp    $0xa,%cl
    1e3d:	74 38                	je     1e77 <rio_readlineb+0xb2>
    1e3f:	83 44 24 08 01       	addl   $0x1,0x8(%esp)
    1e44:	3b 6c 24 0c          	cmp    0xc(%esp),%ebp
    1e48:	74 29                	je     1e73 <rio_readlineb+0xae>
    1e4a:	8b 46 04             	mov    0x4(%esi),%eax
    1e4d:	85 c0                	test   %eax,%eax
    1e4f:	7e b2                	jle    1e03 <rio_readlineb+0x3e>
    1e51:	eb cf                	jmp    1e22 <rio_readlineb+0x5d>
    1e53:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    1e58:	85 c0                	test   %eax,%eax
    1e5a:	75 2b                	jne    1e87 <rio_readlineb+0xc2>
    1e5c:	83 7c 24 08 01       	cmpl   $0x1,0x8(%esp)
    1e61:	75 14                	jne    1e77 <rio_readlineb+0xb2>
    1e63:	89 44 24 08          	mov    %eax,0x8(%esp)
    1e67:	eb 12                	jmp    1e7b <rio_readlineb+0xb6>
    1e69:	c7 44 24 08 01 00 00 	movl   $0x1,0x8(%esp)
    1e70:	00 
    1e71:	eb 04                	jmp    1e77 <rio_readlineb+0xb2>
    1e73:	8b 6c 24 0c          	mov    0xc(%esp),%ebp
    1e77:	c6 45 00 00          	movb   $0x0,0x0(%ebp)
    1e7b:	8b 44 24 08          	mov    0x8(%esp),%eax
    1e7f:	83 c4 1c             	add    $0x1c,%esp
    1e82:	5b                   	pop    %ebx
    1e83:	5e                   	pop    %esi
    1e84:	5f                   	pop    %edi
    1e85:	5d                   	pop    %ebp
    1e86:	c3                   	ret    
    1e87:	c7 44 24 08 ff ff ff 	movl   $0xffffffff,0x8(%esp)
    1e8e:	ff 
    1e8f:	eb ea                	jmp    1e7b <rio_readlineb+0xb6>

00001e91 <submitr>:
    1e91:	55                   	push   %ebp
    1e92:	57                   	push   %edi
    1e93:	56                   	push   %esi
    1e94:	53                   	push   %ebx
    1e95:	8d 84 24 00 60 ff ff 	lea    -0xa000(%esp),%eax
    1e9c:	81 ec 00 10 00 00    	sub    $0x1000,%esp
    1ea2:	83 0c 24 00          	orl    $0x0,(%esp)
    1ea6:	39 c4                	cmp    %eax,%esp
    1ea8:	75 f2                	jne    1e9c <submitr+0xb>
    1eaa:	83 ec 60             	sub    $0x60,%esp
    1ead:	e8 8e f3 ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    1eb2:	81 c3 b2 40 00 00    	add    $0x40b2,%ebx
    1eb8:	8b b4 24 74 a0 00 00 	mov    0xa074(%esp),%esi
    1ebf:	8b 84 24 7c a0 00 00 	mov    0xa07c(%esp),%eax
    1ec6:	89 44 24 08          	mov    %eax,0x8(%esp)
    1eca:	8b 84 24 80 a0 00 00 	mov    0xa080(%esp),%eax
    1ed1:	89 44 24 0c          	mov    %eax,0xc(%esp)
    1ed5:	8b 84 24 84 a0 00 00 	mov    0xa084(%esp),%eax
    1edc:	89 44 24 10          	mov    %eax,0x10(%esp)
    1ee0:	8b 84 24 88 a0 00 00 	mov    0xa088(%esp),%eax
    1ee7:	89 44 24 04          	mov    %eax,0x4(%esp)
    1eeb:	8b 84 24 8c a0 00 00 	mov    0xa08c(%esp),%eax
    1ef2:	89 44 24 14          	mov    %eax,0x14(%esp)
    1ef6:	65 a1 14 00 00 00    	mov    %gs:0x14,%eax
    1efc:	89 84 24 50 a0 00 00 	mov    %eax,0xa050(%esp)
    1f03:	31 c0                	xor    %eax,%eax
    1f05:	c7 44 24 30 00 00 00 	movl   $0x0,0x30(%esp)
    1f0c:	00 
    1f0d:	6a 00                	push   $0x0
    1f0f:	6a 01                	push   $0x1
    1f11:	6a 02                	push   $0x2
    1f13:	e8 68 f2 ff ff       	call   1180 <socket@plt>
    1f18:	83 c4 10             	add    $0x10,%esp
    1f1b:	85 c0                	test   %eax,%eax
    1f1d:	0f 88 2b 01 00 00    	js     204e <submitr+0x1bd>
    1f23:	89 c5                	mov    %eax,%ebp
    1f25:	83 ec 0c             	sub    $0xc,%esp
    1f28:	56                   	push   %esi
    1f29:	e8 72 f2 ff ff       	call   11a0 <gethostbyname@plt>
    1f2e:	83 c4 10             	add    $0x10,%esp
    1f31:	85 c0                	test   %eax,%eax
    1f33:	0f 84 67 01 00 00    	je     20a0 <submitr+0x20f>
    1f39:	8d 74 24 30          	lea    0x30(%esp),%esi
    1f3d:	c7 44 24 30 00 00 00 	movl   $0x0,0x30(%esp)
    1f44:	00 
    1f45:	c7 44 24 34 00 00 00 	movl   $0x0,0x34(%esp)
    1f4c:	00 
    1f4d:	c7 44 24 38 00 00 00 	movl   $0x0,0x38(%esp)
    1f54:	00 
    1f55:	c7 44 24 3c 00 00 00 	movl   $0x0,0x3c(%esp)
    1f5c:	00 
    1f5d:	66 c7 44 24 30 02 00 	movw   $0x2,0x30(%esp)
    1f64:	6a 0c                	push   $0xc
    1f66:	ff 70 0c             	pushl  0xc(%eax)
    1f69:	8b 40 10             	mov    0x10(%eax),%eax
    1f6c:	ff 30                	pushl  (%eax)
    1f6e:	8d 44 24 40          	lea    0x40(%esp),%eax
    1f72:	50                   	push   %eax
    1f73:	e8 88 f1 ff ff       	call   1100 <__memmove_chk@plt>
    1f78:	0f b7 84 24 84 a0 00 	movzwl 0xa084(%esp),%eax
    1f7f:	00 
    1f80:	66 c1 c0 08          	rol    $0x8,%ax
    1f84:	66 89 44 24 42       	mov    %ax,0x42(%esp)
    1f89:	83 c4 0c             	add    $0xc,%esp
    1f8c:	6a 10                	push   $0x10
    1f8e:	56                   	push   %esi
    1f8f:	55                   	push   %ebp
    1f90:	e8 2b f2 ff ff       	call   11c0 <connect@plt>
    1f95:	83 c4 10             	add    $0x10,%esp
    1f98:	85 c0                	test   %eax,%eax
    1f9a:	0f 88 70 01 00 00    	js     2110 <submitr+0x27f>
    1fa0:	83 ec 0c             	sub    $0xc,%esp
    1fa3:	ff 74 24 0c          	pushl  0xc(%esp)
    1fa7:	e8 74 f1 ff ff       	call   1120 <strlen@plt>
    1fac:	83 c4 04             	add    $0x4,%esp
    1faf:	89 c6                	mov    %eax,%esi
    1fb1:	ff 74 24 10          	pushl  0x10(%esp)
    1fb5:	e8 66 f1 ff ff       	call   1120 <strlen@plt>
    1fba:	83 c4 04             	add    $0x4,%esp
    1fbd:	89 44 24 20          	mov    %eax,0x20(%esp)
    1fc1:	ff 74 24 14          	pushl  0x14(%esp)
    1fc5:	e8 56 f1 ff ff       	call   1120 <strlen@plt>
    1fca:	83 c4 04             	add    $0x4,%esp
    1fcd:	89 c7                	mov    %eax,%edi
    1fcf:	ff 74 24 18          	pushl  0x18(%esp)
    1fd3:	e8 48 f1 ff ff       	call   1120 <strlen@plt>
    1fd8:	83 c4 10             	add    $0x10,%esp
    1fdb:	89 c2                	mov    %eax,%edx
    1fdd:	8b 44 24 14          	mov    0x14(%esp),%eax
    1fe1:	8d 84 38 80 00 00 00 	lea    0x80(%eax,%edi,1),%eax
    1fe8:	01 d0                	add    %edx,%eax
    1fea:	8d 14 76             	lea    (%esi,%esi,2),%edx
    1fed:	01 d0                	add    %edx,%eax
    1fef:	3d 00 20 00 00       	cmp    $0x2000,%eax
    1ff4:	0f 87 78 01 00 00    	ja     2172 <submitr+0x2e1>
    1ffa:	8d 94 24 4c 40 00 00 	lea    0x404c(%esp),%edx
    2001:	b9 00 08 00 00       	mov    $0x800,%ecx
    2006:	b8 00 00 00 00       	mov    $0x0,%eax
    200b:	89 d7                	mov    %edx,%edi
    200d:	f3 ab                	rep stos %eax,%es:(%edi)
    200f:	83 ec 0c             	sub    $0xc,%esp
    2012:	8b 74 24 0c          	mov    0xc(%esp),%esi
    2016:	56                   	push   %esi
    2017:	e8 04 f1 ff ff       	call   1120 <strlen@plt>
    201c:	83 c4 10             	add    $0x10,%esp
    201f:	85 c0                	test   %eax,%eax
    2021:	0f 84 6c 02 00 00    	je     2293 <submitr+0x402>
    2027:	8d bc 24 4c 40 00 00 	lea    0x404c(%esp),%edi
    202e:	8d 8b 8e d5 ff ff    	lea    -0x2a72(%ebx),%ecx
    2034:	89 4c 24 18          	mov    %ecx,0x18(%esp)
    2038:	8d 8c 24 4c 80 00 00 	lea    0x804c(%esp),%ecx
    203f:	89 4c 24 1c          	mov    %ecx,0x1c(%esp)
    2043:	89 6c 24 14          	mov    %ebp,0x14(%esp)
    2047:	89 c5                	mov    %eax,%ebp
    2049:	e9 b8 01 00 00       	jmp    2206 <submitr+0x375>
    204e:	8b 44 24 10          	mov    0x10(%esp),%eax
    2052:	c7 00 45 72 72 6f    	movl   $0x6f727245,(%eax)
    2058:	c7 40 04 72 3a 20 43 	movl   $0x43203a72,0x4(%eax)
    205f:	c7 40 08 6c 69 65 6e 	movl   $0x6e65696c,0x8(%eax)
    2066:	c7 40 0c 74 20 75 6e 	movl   $0x6e752074,0xc(%eax)
    206d:	c7 40 10 61 62 6c 65 	movl   $0x656c6261,0x10(%eax)
    2074:	c7 40 14 20 74 6f 20 	movl   $0x206f7420,0x14(%eax)
    207b:	c7 40 18 63 72 65 61 	movl   $0x61657263,0x18(%eax)
    2082:	c7 40 1c 74 65 20 73 	movl   $0x73206574,0x1c(%eax)
    2089:	c7 40 20 6f 63 6b 65 	movl   $0x656b636f,0x20(%eax)
    2090:	66 c7 40 24 74 00    	movw   $0x74,0x24(%eax)
    2096:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    209b:	e9 80 04 00 00       	jmp    2520 <submitr+0x68f>
    20a0:	8b 44 24 10          	mov    0x10(%esp),%eax
    20a4:	c7 00 45 72 72 6f    	movl   $0x6f727245,(%eax)
    20aa:	c7 40 04 72 3a 20 44 	movl   $0x44203a72,0x4(%eax)
    20b1:	c7 40 08 4e 53 20 69 	movl   $0x6920534e,0x8(%eax)
    20b8:	c7 40 0c 73 20 75 6e 	movl   $0x6e752073,0xc(%eax)
    20bf:	c7 40 10 61 62 6c 65 	movl   $0x656c6261,0x10(%eax)
    20c6:	c7 40 14 20 74 6f 20 	movl   $0x206f7420,0x14(%eax)
    20cd:	c7 40 18 72 65 73 6f 	movl   $0x6f736572,0x18(%eax)
    20d4:	c7 40 1c 6c 76 65 20 	movl   $0x2065766c,0x1c(%eax)
    20db:	c7 40 20 73 65 72 76 	movl   $0x76726573,0x20(%eax)
    20e2:	c7 40 24 65 72 20 61 	movl   $0x61207265,0x24(%eax)
    20e9:	c7 40 28 64 64 72 65 	movl   $0x65726464,0x28(%eax)
    20f0:	66 c7 40 2c 73 73    	movw   $0x7373,0x2c(%eax)
    20f6:	c6 40 2e 00          	movb   $0x0,0x2e(%eax)
    20fa:	83 ec 0c             	sub    $0xc,%esp
    20fd:	55                   	push   %ebp
    20fe:	e8 cd f0 ff ff       	call   11d0 <close@plt>
    2103:	83 c4 10             	add    $0x10,%esp
    2106:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    210b:	e9 10 04 00 00       	jmp    2520 <submitr+0x68f>
    2110:	8b 44 24 10          	mov    0x10(%esp),%eax
    2114:	c7 00 45 72 72 6f    	movl   $0x6f727245,(%eax)
    211a:	c7 40 04 72 3a 20 55 	movl   $0x55203a72,0x4(%eax)
    2121:	c7 40 08 6e 61 62 6c 	movl   $0x6c62616e,0x8(%eax)
    2128:	c7 40 0c 65 20 74 6f 	movl   $0x6f742065,0xc(%eax)
    212f:	c7 40 10 20 63 6f 6e 	movl   $0x6e6f6320,0x10(%eax)
    2136:	c7 40 14 6e 65 63 74 	movl   $0x7463656e,0x14(%eax)
    213d:	c7 40 18 20 74 6f 20 	movl   $0x206f7420,0x18(%eax)
    2144:	c7 40 1c 74 68 65 20 	movl   $0x20656874,0x1c(%eax)
    214b:	c7 40 20 73 65 72 76 	movl   $0x76726573,0x20(%eax)
    2152:	66 c7 40 24 65 72    	movw   $0x7265,0x24(%eax)
    2158:	c6 40 26 00          	movb   $0x0,0x26(%eax)
    215c:	83 ec 0c             	sub    $0xc,%esp
    215f:	55                   	push   %ebp
    2160:	e8 6b f0 ff ff       	call   11d0 <close@plt>
    2165:	83 c4 10             	add    $0x10,%esp
    2168:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    216d:	e9 ae 03 00 00       	jmp    2520 <submitr+0x68f>
    2172:	8b 44 24 10          	mov    0x10(%esp),%eax
    2176:	c7 00 45 72 72 6f    	movl   $0x6f727245,(%eax)
    217c:	c7 40 04 72 3a 20 52 	movl   $0x52203a72,0x4(%eax)
    2183:	c7 40 08 65 73 75 6c 	movl   $0x6c757365,0x8(%eax)
    218a:	c7 40 0c 74 20 73 74 	movl   $0x74732074,0xc(%eax)
    2191:	c7 40 10 72 69 6e 67 	movl   $0x676e6972,0x10(%eax)
    2198:	c7 40 14 20 74 6f 6f 	movl   $0x6f6f7420,0x14(%eax)
    219f:	c7 40 18 20 6c 61 72 	movl   $0x72616c20,0x18(%eax)
    21a6:	c7 40 1c 67 65 2e 20 	movl   $0x202e6567,0x1c(%eax)
    21ad:	c7 40 20 49 6e 63 72 	movl   $0x72636e49,0x20(%eax)
    21b4:	c7 40 24 65 61 73 65 	movl   $0x65736165,0x24(%eax)
    21bb:	c7 40 28 20 53 55 42 	movl   $0x42555320,0x28(%eax)
    21c2:	c7 40 2c 4d 49 54 52 	movl   $0x5254494d,0x2c(%eax)
    21c9:	c7 40 30 5f 4d 41 58 	movl   $0x58414d5f,0x30(%eax)
    21d0:	c7 40 34 42 55 46 00 	movl   $0x465542,0x34(%eax)
    21d7:	83 ec 0c             	sub    $0xc,%esp
    21da:	55                   	push   %ebp
    21db:	e8 f0 ef ff ff       	call   11d0 <close@plt>
    21e0:	83 c4 10             	add    $0x10,%esp
    21e3:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    21e8:	e9 33 03 00 00       	jmp    2520 <submitr+0x68f>
    21ed:	3c 5f                	cmp    $0x5f,%al
    21ef:	75 7e                	jne    226f <submitr+0x3de>
    21f1:	88 07                	mov    %al,(%edi)
    21f3:	8d 7f 01             	lea    0x1(%edi),%edi
    21f6:	83 c6 01             	add    $0x1,%esi
    21f9:	8b 04 24             	mov    (%esp),%eax
    21fc:	01 e8                	add    %ebp,%eax
    21fe:	39 c6                	cmp    %eax,%esi
    2200:	0f 84 89 00 00 00    	je     228f <submitr+0x3fe>
    2206:	0f b6 06             	movzbl (%esi),%eax
    2209:	8d 50 d6             	lea    -0x2a(%eax),%edx
    220c:	80 fa 0f             	cmp    $0xf,%dl
    220f:	77 dc                	ja     21ed <submitr+0x35c>
    2211:	b9 d9 ff 00 00       	mov    $0xffd9,%ecx
    2216:	0f a3 d1             	bt     %edx,%ecx
    2219:	72 d6                	jb     21f1 <submitr+0x360>
    221b:	3c 5f                	cmp    $0x5f,%al
    221d:	74 d2                	je     21f1 <submitr+0x360>
    221f:	8d 50 e0             	lea    -0x20(%eax),%edx
    2222:	80 fa 5f             	cmp    $0x5f,%dl
    2225:	76 08                	jbe    222f <submitr+0x39e>
    2227:	3c 09                	cmp    $0x9,%al
    2229:	0f 85 f1 03 00 00    	jne    2620 <submitr+0x78f>
    222f:	83 ec 0c             	sub    $0xc,%esp
    2232:	0f b6 c0             	movzbl %al,%eax
    2235:	50                   	push   %eax
    2236:	ff 74 24 28          	pushl  0x28(%esp)
    223a:	6a 08                	push   $0x8
    223c:	6a 01                	push   $0x1
    223e:	ff 74 24 38          	pushl  0x38(%esp)
    2242:	e8 a9 ef ff ff       	call   11f0 <__sprintf_chk@plt>
    2247:	0f b6 84 24 6c 80 00 	movzbl 0x806c(%esp),%eax
    224e:	00 
    224f:	88 07                	mov    %al,(%edi)
    2251:	0f b6 84 24 6d 80 00 	movzbl 0x806d(%esp),%eax
    2258:	00 
    2259:	88 47 01             	mov    %al,0x1(%edi)
    225c:	0f b6 84 24 6e 80 00 	movzbl 0x806e(%esp),%eax
    2263:	00 
    2264:	88 47 02             	mov    %al,0x2(%edi)
    2267:	83 c4 20             	add    $0x20,%esp
    226a:	8d 7f 03             	lea    0x3(%edi),%edi
    226d:	eb 87                	jmp    21f6 <submitr+0x365>
    226f:	89 c2                	mov    %eax,%edx
    2271:	83 e2 df             	and    $0xffffffdf,%edx
    2274:	83 ea 41             	sub    $0x41,%edx
    2277:	80 fa 19             	cmp    $0x19,%dl
    227a:	0f 86 71 ff ff ff    	jbe    21f1 <submitr+0x360>
    2280:	3c 20                	cmp    $0x20,%al
    2282:	75 9b                	jne    221f <submitr+0x38e>
    2284:	c6 07 2b             	movb   $0x2b,(%edi)
    2287:	8d 7f 01             	lea    0x1(%edi),%edi
    228a:	e9 67 ff ff ff       	jmp    21f6 <submitr+0x365>
    228f:	8b 6c 24 14          	mov    0x14(%esp),%ebp
    2293:	8d 84 24 4c 40 00 00 	lea    0x404c(%esp),%eax
    229a:	50                   	push   %eax
    229b:	ff 74 24 10          	pushl  0x10(%esp)
    229f:	ff 74 24 10          	pushl  0x10(%esp)
    22a3:	ff 74 24 10          	pushl  0x10(%esp)
    22a7:	8d 83 1c d5 ff ff    	lea    -0x2ae4(%ebx),%eax
    22ad:	50                   	push   %eax
    22ae:	68 00 20 00 00       	push   $0x2000
    22b3:	6a 01                	push   $0x1
    22b5:	8d b4 24 68 20 00 00 	lea    0x2068(%esp),%esi
    22bc:	56                   	push   %esi
    22bd:	e8 2e ef ff ff       	call   11f0 <__sprintf_chk@plt>
    22c2:	83 c4 14             	add    $0x14,%esp
    22c5:	56                   	push   %esi
    22c6:	e8 55 ee ff ff       	call   1120 <strlen@plt>
    22cb:	83 c4 10             	add    $0x10,%esp
    22ce:	89 c6                	mov    %eax,%esi
    22d0:	8d bc 24 4c 20 00 00 	lea    0x204c(%esp),%edi
    22d7:	85 c0                	test   %eax,%eax
    22d9:	0f 85 20 01 00 00    	jne    23ff <submitr+0x56e>
    22df:	89 6c 24 40          	mov    %ebp,0x40(%esp)
    22e3:	c7 44 24 44 00 00 00 	movl   $0x0,0x44(%esp)
    22ea:	00 
    22eb:	8d 44 24 40          	lea    0x40(%esp),%eax
    22ef:	8d 54 24 4c          	lea    0x4c(%esp),%edx
    22f3:	89 54 24 48          	mov    %edx,0x48(%esp)
    22f7:	8d 94 24 4c 20 00 00 	lea    0x204c(%esp),%edx
    22fe:	b9 00 20 00 00       	mov    $0x2000,%ecx
    2303:	e8 bd fa ff ff       	call   1dc5 <rio_readlineb>
    2308:	85 c0                	test   %eax,%eax
    230a:	0f 8e 16 01 00 00    	jle    2426 <submitr+0x595>
    2310:	83 ec 0c             	sub    $0xc,%esp
    2313:	8d 84 24 58 80 00 00 	lea    0x8058(%esp),%eax
    231a:	50                   	push   %eax
    231b:	8d 44 24 3c          	lea    0x3c(%esp),%eax
    231f:	50                   	push   %eax
    2320:	8d 84 24 60 60 00 00 	lea    0x6060(%esp),%eax
    2327:	50                   	push   %eax
    2328:	8d 83 95 d5 ff ff    	lea    -0x2a6b(%ebx),%eax
    232e:	50                   	push   %eax
    232f:	8d 84 24 68 20 00 00 	lea    0x2068(%esp),%eax
    2336:	50                   	push   %eax
    2337:	e8 04 ee ff ff       	call   1140 <__isoc99_sscanf@plt>
    233c:	8b 44 24 4c          	mov    0x4c(%esp),%eax
    2340:	83 c4 20             	add    $0x20,%esp
    2343:	3d c8 00 00 00       	cmp    $0xc8,%eax
    2348:	0f 85 52 01 00 00    	jne    24a0 <submitr+0x60f>
    234e:	8d bb a6 d5 ff ff    	lea    -0x2a5a(%ebx),%edi
    2354:	8d b4 24 4c 20 00 00 	lea    0x204c(%esp),%esi
    235b:	83 ec 08             	sub    $0x8,%esp
    235e:	57                   	push   %edi
    235f:	56                   	push   %esi
    2360:	e8 db ec ff ff       	call   1040 <strcmp@plt>
    2365:	83 c4 10             	add    $0x10,%esp
    2368:	85 c0                	test   %eax,%eax
    236a:	0f 84 63 01 00 00    	je     24d3 <submitr+0x642>
    2370:	8d 44 24 40          	lea    0x40(%esp),%eax
    2374:	b9 00 20 00 00       	mov    $0x2000,%ecx
    2379:	89 f2                	mov    %esi,%edx
    237b:	e8 45 fa ff ff       	call   1dc5 <rio_readlineb>
    2380:	85 c0                	test   %eax,%eax
    2382:	7f d7                	jg     235b <submitr+0x4ca>
    2384:	8b 44 24 10          	mov    0x10(%esp),%eax
    2388:	c7 00 45 72 72 6f    	movl   $0x6f727245,(%eax)
    238e:	c7 40 04 72 3a 20 43 	movl   $0x43203a72,0x4(%eax)
    2395:	c7 40 08 6c 69 65 6e 	movl   $0x6e65696c,0x8(%eax)
    239c:	c7 40 0c 74 20 75 6e 	movl   $0x6e752074,0xc(%eax)
    23a3:	c7 40 10 61 62 6c 65 	movl   $0x656c6261,0x10(%eax)
    23aa:	c7 40 14 20 74 6f 20 	movl   $0x206f7420,0x14(%eax)
    23b1:	c7 40 18 72 65 61 64 	movl   $0x64616572,0x18(%eax)
    23b8:	c7 40 1c 20 68 65 61 	movl   $0x61656820,0x1c(%eax)
    23bf:	c7 40 20 64 65 72 73 	movl   $0x73726564,0x20(%eax)
    23c6:	c7 40 24 20 66 72 6f 	movl   $0x6f726620,0x24(%eax)
    23cd:	c7 40 28 6d 20 73 65 	movl   $0x6573206d,0x28(%eax)
    23d4:	c7 40 2c 72 76 65 72 	movl   $0x72657672,0x2c(%eax)
    23db:	c6 40 30 00          	movb   $0x0,0x30(%eax)
    23df:	83 ec 0c             	sub    $0xc,%esp
    23e2:	55                   	push   %ebp
    23e3:	e8 e8 ed ff ff       	call   11d0 <close@plt>
    23e8:	83 c4 10             	add    $0x10,%esp
    23eb:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    23f0:	e9 2b 01 00 00       	jmp    2520 <submitr+0x68f>
    23f5:	01 c7                	add    %eax,%edi
    23f7:	29 c6                	sub    %eax,%esi
    23f9:	0f 84 e0 fe ff ff    	je     22df <submitr+0x44e>
    23ff:	83 ec 04             	sub    $0x4,%esp
    2402:	56                   	push   %esi
    2403:	57                   	push   %edi
    2404:	55                   	push   %ebp
    2405:	e8 26 ed ff ff       	call   1130 <write@plt>
    240a:	83 c4 10             	add    $0x10,%esp
    240d:	85 c0                	test   %eax,%eax
    240f:	7f e4                	jg     23f5 <submitr+0x564>
    2411:	e8 4a ed ff ff       	call   1160 <__errno_location@plt>
    2416:	83 38 04             	cmpl   $0x4,(%eax)
    2419:	0f 85 9b 01 00 00    	jne    25ba <submitr+0x729>
    241f:	b8 00 00 00 00       	mov    $0x0,%eax
    2424:	eb cf                	jmp    23f5 <submitr+0x564>
    2426:	8b 44 24 10          	mov    0x10(%esp),%eax
    242a:	c7 00 45 72 72 6f    	movl   $0x6f727245,(%eax)
    2430:	c7 40 04 72 3a 20 43 	movl   $0x43203a72,0x4(%eax)
    2437:	c7 40 08 6c 69 65 6e 	movl   $0x6e65696c,0x8(%eax)
    243e:	c7 40 0c 74 20 75 6e 	movl   $0x6e752074,0xc(%eax)
    2445:	c7 40 10 61 62 6c 65 	movl   $0x656c6261,0x10(%eax)
    244c:	c7 40 14 20 74 6f 20 	movl   $0x206f7420,0x14(%eax)
    2453:	c7 40 18 72 65 61 64 	movl   $0x64616572,0x18(%eax)
    245a:	c7 40 1c 20 66 69 72 	movl   $0x72696620,0x1c(%eax)
    2461:	c7 40 20 73 74 20 68 	movl   $0x68207473,0x20(%eax)
    2468:	c7 40 24 65 61 64 65 	movl   $0x65646165,0x24(%eax)
    246f:	c7 40 28 72 20 66 72 	movl   $0x72662072,0x28(%eax)
    2476:	c7 40 2c 6f 6d 20 73 	movl   $0x73206d6f,0x2c(%eax)
    247d:	c7 40 30 65 72 76 65 	movl   $0x65767265,0x30(%eax)
    2484:	66 c7 40 34 72 00    	movw   $0x72,0x34(%eax)
    248a:	83 ec 0c             	sub    $0xc,%esp
    248d:	55                   	push   %ebp
    248e:	e8 3d ed ff ff       	call   11d0 <close@plt>
    2493:	83 c4 10             	add    $0x10,%esp
    2496:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    249b:	e9 80 00 00 00       	jmp    2520 <submitr+0x68f>
    24a0:	83 ec 08             	sub    $0x8,%esp
    24a3:	8d 94 24 54 80 00 00 	lea    0x8054(%esp),%edx
    24aa:	52                   	push   %edx
    24ab:	50                   	push   %eax
    24ac:	8d 83 a8 d4 ff ff    	lea    -0x2b58(%ebx),%eax
    24b2:	50                   	push   %eax
    24b3:	6a ff                	push   $0xffffffff
    24b5:	6a 01                	push   $0x1
    24b7:	ff 74 24 2c          	pushl  0x2c(%esp)
    24bb:	e8 30 ed ff ff       	call   11f0 <__sprintf_chk@plt>
    24c0:	83 c4 14             	add    $0x14,%esp
    24c3:	55                   	push   %ebp
    24c4:	e8 07 ed ff ff       	call   11d0 <close@plt>
    24c9:	83 c4 10             	add    $0x10,%esp
    24cc:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    24d1:	eb 4d                	jmp    2520 <submitr+0x68f>
    24d3:	8d 94 24 4c 20 00 00 	lea    0x204c(%esp),%edx
    24da:	8d 44 24 40          	lea    0x40(%esp),%eax
    24de:	b9 00 20 00 00       	mov    $0x2000,%ecx
    24e3:	e8 dd f8 ff ff       	call   1dc5 <rio_readlineb>
    24e8:	85 c0                	test   %eax,%eax
    24ea:	7e 53                	jle    253f <submitr+0x6ae>
    24ec:	83 ec 08             	sub    $0x8,%esp
    24ef:	8d 84 24 54 20 00 00 	lea    0x2054(%esp),%eax
    24f6:	50                   	push   %eax
    24f7:	8b 7c 24 1c          	mov    0x1c(%esp),%edi
    24fb:	57                   	push   %edi
    24fc:	e8 cf eb ff ff       	call   10d0 <strcpy@plt>
    2501:	89 2c 24             	mov    %ebp,(%esp)
    2504:	e8 c7 ec ff ff       	call   11d0 <close@plt>
    2509:	83 c4 08             	add    $0x8,%esp
    250c:	8d 83 a9 d5 ff ff    	lea    -0x2a57(%ebx),%eax
    2512:	50                   	push   %eax
    2513:	57                   	push   %edi
    2514:	e8 27 eb ff ff       	call   1040 <strcmp@plt>
    2519:	83 c4 10             	add    $0x10,%esp
    251c:	f7 d8                	neg    %eax
    251e:	19 c0                	sbb    %eax,%eax
    2520:	8b 94 24 4c a0 00 00 	mov    0xa04c(%esp),%edx
    2527:	65 2b 15 14 00 00 00 	sub    %gs:0x14,%edx
    252e:	0f 85 37 01 00 00    	jne    266b <submitr+0x7da>
    2534:	81 c4 5c a0 00 00    	add    $0xa05c,%esp
    253a:	5b                   	pop    %ebx
    253b:	5e                   	pop    %esi
    253c:	5f                   	pop    %edi
    253d:	5d                   	pop    %ebp
    253e:	c3                   	ret    
    253f:	8b 44 24 10          	mov    0x10(%esp),%eax
    2543:	c7 00 45 72 72 6f    	movl   $0x6f727245,(%eax)
    2549:	c7 40 04 72 3a 20 43 	movl   $0x43203a72,0x4(%eax)
    2550:	c7 40 08 6c 69 65 6e 	movl   $0x6e65696c,0x8(%eax)
    2557:	c7 40 0c 74 20 75 6e 	movl   $0x6e752074,0xc(%eax)
    255e:	c7 40 10 61 62 6c 65 	movl   $0x656c6261,0x10(%eax)
    2565:	c7 40 14 20 74 6f 20 	movl   $0x206f7420,0x14(%eax)
    256c:	c7 40 18 72 65 61 64 	movl   $0x64616572,0x18(%eax)
    2573:	c7 40 1c 20 73 74 61 	movl   $0x61747320,0x1c(%eax)
    257a:	c7 40 20 74 75 73 20 	movl   $0x20737574,0x20(%eax)
    2581:	c7 40 24 6d 65 73 73 	movl   $0x7373656d,0x24(%eax)
    2588:	c7 40 28 61 67 65 20 	movl   $0x20656761,0x28(%eax)
    258f:	c7 40 2c 66 72 6f 6d 	movl   $0x6d6f7266,0x2c(%eax)
    2596:	c7 40 30 20 73 65 72 	movl   $0x72657320,0x30(%eax)
    259d:	c7 40 34 76 65 72 00 	movl   $0x726576,0x34(%eax)
    25a4:	83 ec 0c             	sub    $0xc,%esp
    25a7:	55                   	push   %ebp
    25a8:	e8 23 ec ff ff       	call   11d0 <close@plt>
    25ad:	83 c4 10             	add    $0x10,%esp
    25b0:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    25b5:	e9 66 ff ff ff       	jmp    2520 <submitr+0x68f>
    25ba:	8b 44 24 10          	mov    0x10(%esp),%eax
    25be:	c7 00 45 72 72 6f    	movl   $0x6f727245,(%eax)
    25c4:	c7 40 04 72 3a 20 43 	movl   $0x43203a72,0x4(%eax)
    25cb:	c7 40 08 6c 69 65 6e 	movl   $0x6e65696c,0x8(%eax)
    25d2:	c7 40 0c 74 20 75 6e 	movl   $0x6e752074,0xc(%eax)
    25d9:	c7 40 10 61 62 6c 65 	movl   $0x656c6261,0x10(%eax)
    25e0:	c7 40 14 20 74 6f 20 	movl   $0x206f7420,0x14(%eax)
    25e7:	c7 40 18 77 72 69 74 	movl   $0x74697277,0x18(%eax)
    25ee:	c7 40 1c 65 20 74 6f 	movl   $0x6f742065,0x1c(%eax)
    25f5:	c7 40 20 20 74 68 65 	movl   $0x65687420,0x20(%eax)
    25fc:	c7 40 24 20 73 65 72 	movl   $0x72657320,0x24(%eax)
    2603:	c7 40 28 76 65 72 00 	movl   $0x726576,0x28(%eax)
    260a:	83 ec 0c             	sub    $0xc,%esp
    260d:	55                   	push   %ebp
    260e:	e8 bd eb ff ff       	call   11d0 <close@plt>
    2613:	83 c4 10             	add    $0x10,%esp
    2616:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    261b:	e9 00 ff ff ff       	jmp    2520 <submitr+0x68f>
    2620:	8b 6c 24 14          	mov    0x14(%esp),%ebp
    2624:	8b 83 d8 d4 ff ff    	mov    -0x2b28(%ebx),%eax
    262a:	8b 4c 24 10          	mov    0x10(%esp),%ecx
    262e:	89 01                	mov    %eax,(%ecx)
    2630:	8b 83 17 d5 ff ff    	mov    -0x2ae9(%ebx),%eax
    2636:	89 41 3f             	mov    %eax,0x3f(%ecx)
    2639:	89 c8                	mov    %ecx,%eax
    263b:	8d 79 04             	lea    0x4(%ecx),%edi
    263e:	83 e7 fc             	and    $0xfffffffc,%edi
    2641:	29 f8                	sub    %edi,%eax
    2643:	8d b3 d8 d4 ff ff    	lea    -0x2b28(%ebx),%esi
    2649:	29 c6                	sub    %eax,%esi
    264b:	83 c0 43             	add    $0x43,%eax
    264e:	c1 e8 02             	shr    $0x2,%eax
    2651:	89 c1                	mov    %eax,%ecx
    2653:	f3 a5                	rep movsl %ds:(%esi),%es:(%edi)
    2655:	83 ec 0c             	sub    $0xc,%esp
    2658:	55                   	push   %ebp
    2659:	e8 72 eb ff ff       	call   11d0 <close@plt>
    265e:	83 c4 10             	add    $0x10,%esp
    2661:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    2666:	e9 b5 fe ff ff       	jmp    2520 <submitr+0x68f>
    266b:	e8 d0 02 00 00       	call   2940 <__stack_chk_fail_local>

00002670 <init_timeout>:
    2670:	56                   	push   %esi
    2671:	53                   	push   %ebx
    2672:	83 ec 04             	sub    $0x4,%esp
    2675:	e8 c6 eb ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    267a:	81 c3 ea 38 00 00    	add    $0x38ea,%ebx
    2680:	8b 74 24 10          	mov    0x10(%esp),%esi
    2684:	85 f6                	test   %esi,%esi
    2686:	75 06                	jne    268e <init_timeout+0x1e>
    2688:	83 c4 04             	add    $0x4,%esp
    268b:	5b                   	pop    %ebx
    268c:	5e                   	pop    %esi
    268d:	c3                   	ret    
    268e:	83 ec 08             	sub    $0x8,%esp
    2691:	8d 83 2e be ff ff    	lea    -0x41d2(%ebx),%eax
    2697:	50                   	push   %eax
    2698:	6a 0e                	push   $0xe
    269a:	e8 f1 e9 ff ff       	call   1090 <signal@plt>
    269f:	85 f6                	test   %esi,%esi
    26a1:	b8 00 00 00 00       	mov    $0x0,%eax
    26a6:	0f 48 f0             	cmovs  %eax,%esi
    26a9:	89 34 24             	mov    %esi,(%esp)
    26ac:	e8 ff e9 ff ff       	call   10b0 <alarm@plt>
    26b1:	83 c4 10             	add    $0x10,%esp
    26b4:	eb d2                	jmp    2688 <init_timeout+0x18>

000026b6 <init_driver>:
    26b6:	55                   	push   %ebp
    26b7:	57                   	push   %edi
    26b8:	56                   	push   %esi
    26b9:	53                   	push   %ebx
    26ba:	83 ec 34             	sub    $0x34,%esp
    26bd:	e8 7e eb ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    26c2:	81 c3 a2 38 00 00    	add    $0x38a2,%ebx
    26c8:	8b 7c 24 48          	mov    0x48(%esp),%edi
    26cc:	65 a1 14 00 00 00    	mov    %gs:0x14,%eax
    26d2:	89 44 24 24          	mov    %eax,0x24(%esp)
    26d6:	31 c0                	xor    %eax,%eax
    26d8:	6a 01                	push   $0x1
    26da:	6a 0d                	push   $0xd
    26dc:	e8 af e9 ff ff       	call   1090 <signal@plt>
    26e1:	83 c4 08             	add    $0x8,%esp
    26e4:	6a 01                	push   $0x1
    26e6:	6a 1d                	push   $0x1d
    26e8:	e8 a3 e9 ff ff       	call   1090 <signal@plt>
    26ed:	83 c4 08             	add    $0x8,%esp
    26f0:	6a 01                	push   $0x1
    26f2:	6a 1d                	push   $0x1d
    26f4:	e8 97 e9 ff ff       	call   1090 <signal@plt>
    26f9:	83 c4 0c             	add    $0xc,%esp
    26fc:	6a 00                	push   $0x0
    26fe:	6a 01                	push   $0x1
    2700:	6a 02                	push   $0x2
    2702:	e8 79 ea ff ff       	call   1180 <socket@plt>
    2707:	83 c4 10             	add    $0x10,%esp
    270a:	85 c0                	test   %eax,%eax
    270c:	0f 88 ac 00 00 00    	js     27be <init_driver+0x108>
    2712:	89 c6                	mov    %eax,%esi
    2714:	83 ec 0c             	sub    $0xc,%esp
    2717:	8d 83 ac d5 ff ff    	lea    -0x2a54(%ebx),%eax
    271d:	50                   	push   %eax
    271e:	e8 7d ea ff ff       	call   11a0 <gethostbyname@plt>
    2723:	83 c4 10             	add    $0x10,%esp
    2726:	85 c0                	test   %eax,%eax
    2728:	0f 84 db 00 00 00    	je     2809 <init_driver+0x153>
    272e:	8d 6c 24 0c          	lea    0xc(%esp),%ebp
    2732:	c7 44 24 0c 00 00 00 	movl   $0x0,0xc(%esp)
    2739:	00 
    273a:	c7 44 24 10 00 00 00 	movl   $0x0,0x10(%esp)
    2741:	00 
    2742:	c7 44 24 14 00 00 00 	movl   $0x0,0x14(%esp)
    2749:	00 
    274a:	c7 44 24 18 00 00 00 	movl   $0x0,0x18(%esp)
    2751:	00 
    2752:	66 c7 44 24 0c 02 00 	movw   $0x2,0xc(%esp)
    2759:	6a 0c                	push   $0xc
    275b:	ff 70 0c             	pushl  0xc(%eax)
    275e:	8b 40 10             	mov    0x10(%eax),%eax
    2761:	ff 30                	pushl  (%eax)
    2763:	8d 44 24 1c          	lea    0x1c(%esp),%eax
    2767:	50                   	push   %eax
    2768:	e8 93 e9 ff ff       	call   1100 <__memmove_chk@plt>
    276d:	66 c7 44 24 1e 3b 6e 	movw   $0x6e3b,0x1e(%esp)
    2774:	83 c4 0c             	add    $0xc,%esp
    2777:	6a 10                	push   $0x10
    2779:	55                   	push   %ebp
    277a:	56                   	push   %esi
    277b:	e8 40 ea ff ff       	call   11c0 <connect@plt>
    2780:	83 c4 10             	add    $0x10,%esp
    2783:	85 c0                	test   %eax,%eax
    2785:	0f 88 ea 00 00 00    	js     2875 <init_driver+0x1bf>
    278b:	83 ec 0c             	sub    $0xc,%esp
    278e:	56                   	push   %esi
    278f:	e8 3c ea ff ff       	call   11d0 <close@plt>
    2794:	66 c7 07 4f 4b       	movw   $0x4b4f,(%edi)
    2799:	c6 47 02 00          	movb   $0x0,0x2(%edi)
    279d:	83 c4 10             	add    $0x10,%esp
    27a0:	b8 00 00 00 00       	mov    $0x0,%eax
    27a5:	8b 54 24 1c          	mov    0x1c(%esp),%edx
    27a9:	65 2b 15 14 00 00 00 	sub    %gs:0x14,%edx
    27b0:	0f 85 f0 00 00 00    	jne    28a6 <init_driver+0x1f0>
    27b6:	83 c4 2c             	add    $0x2c,%esp
    27b9:	5b                   	pop    %ebx
    27ba:	5e                   	pop    %esi
    27bb:	5f                   	pop    %edi
    27bc:	5d                   	pop    %ebp
    27bd:	c3                   	ret    
    27be:	c7 07 45 72 72 6f    	movl   $0x6f727245,(%edi)
    27c4:	c7 47 04 72 3a 20 43 	movl   $0x43203a72,0x4(%edi)
    27cb:	c7 47 08 6c 69 65 6e 	movl   $0x6e65696c,0x8(%edi)
    27d2:	c7 47 0c 74 20 75 6e 	movl   $0x6e752074,0xc(%edi)
    27d9:	c7 47 10 61 62 6c 65 	movl   $0x656c6261,0x10(%edi)
    27e0:	c7 47 14 20 74 6f 20 	movl   $0x206f7420,0x14(%edi)
    27e7:	c7 47 18 63 72 65 61 	movl   $0x61657263,0x18(%edi)
    27ee:	c7 47 1c 74 65 20 73 	movl   $0x73206574,0x1c(%edi)
    27f5:	c7 47 20 6f 63 6b 65 	movl   $0x656b636f,0x20(%edi)
    27fc:	66 c7 47 24 74 00    	movw   $0x74,0x24(%edi)
    2802:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    2807:	eb 9c                	jmp    27a5 <init_driver+0xef>
    2809:	c7 07 45 72 72 6f    	movl   $0x6f727245,(%edi)
    280f:	c7 47 04 72 3a 20 44 	movl   $0x44203a72,0x4(%edi)
    2816:	c7 47 08 4e 53 20 69 	movl   $0x6920534e,0x8(%edi)
    281d:	c7 47 0c 73 20 75 6e 	movl   $0x6e752073,0xc(%edi)
    2824:	c7 47 10 61 62 6c 65 	movl   $0x656c6261,0x10(%edi)
    282b:	c7 47 14 20 74 6f 20 	movl   $0x206f7420,0x14(%edi)
    2832:	c7 47 18 72 65 73 6f 	movl   $0x6f736572,0x18(%edi)
    2839:	c7 47 1c 6c 76 65 20 	movl   $0x2065766c,0x1c(%edi)
    2840:	c7 47 20 73 65 72 76 	movl   $0x76726573,0x20(%edi)
    2847:	c7 47 24 65 72 20 61 	movl   $0x61207265,0x24(%edi)
    284e:	c7 47 28 64 64 72 65 	movl   $0x65726464,0x28(%edi)
    2855:	66 c7 47 2c 73 73    	movw   $0x7373,0x2c(%edi)
    285b:	c6 47 2e 00          	movb   $0x0,0x2e(%edi)
    285f:	83 ec 0c             	sub    $0xc,%esp
    2862:	56                   	push   %esi
    2863:	e8 68 e9 ff ff       	call   11d0 <close@plt>
    2868:	83 c4 10             	add    $0x10,%esp
    286b:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    2870:	e9 30 ff ff ff       	jmp    27a5 <init_driver+0xef>
    2875:	83 ec 0c             	sub    $0xc,%esp
    2878:	8d 83 ac d5 ff ff    	lea    -0x2a54(%ebx),%eax
    287e:	50                   	push   %eax
    287f:	8d 83 68 d5 ff ff    	lea    -0x2a98(%ebx),%eax
    2885:	50                   	push   %eax
    2886:	6a ff                	push   $0xffffffff
    2888:	6a 01                	push   $0x1
    288a:	57                   	push   %edi
    288b:	e8 60 e9 ff ff       	call   11f0 <__sprintf_chk@plt>
    2890:	83 c4 14             	add    $0x14,%esp
    2893:	56                   	push   %esi
    2894:	e8 37 e9 ff ff       	call   11d0 <close@plt>
    2899:	83 c4 10             	add    $0x10,%esp
    289c:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    28a1:	e9 ff fe ff ff       	jmp    27a5 <init_driver+0xef>
    28a6:	e8 95 00 00 00       	call   2940 <__stack_chk_fail_local>

000028ab <driver_post>:
    28ab:	56                   	push   %esi
    28ac:	53                   	push   %ebx
    28ad:	83 ec 04             	sub    $0x4,%esp
    28b0:	e8 8b e9 ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    28b5:	81 c3 af 36 00 00    	add    $0x36af,%ebx
    28bb:	8b 54 24 10          	mov    0x10(%esp),%edx
    28bf:	8b 44 24 18          	mov    0x18(%esp),%eax
    28c3:	8b 74 24 1c          	mov    0x1c(%esp),%esi
    28c7:	85 c0                	test   %eax,%eax
    28c9:	75 18                	jne    28e3 <driver_post+0x38>
    28cb:	85 d2                	test   %edx,%edx
    28cd:	74 05                	je     28d4 <driver_post+0x29>
    28cf:	80 3a 00             	cmpb   $0x0,(%edx)
    28d2:	75 37                	jne    290b <driver_post+0x60>
    28d4:	66 c7 06 4f 4b       	movw   $0x4b4f,(%esi)
    28d9:	c6 46 02 00          	movb   $0x0,0x2(%esi)
    28dd:	83 c4 04             	add    $0x4,%esp
    28e0:	5b                   	pop    %ebx
    28e1:	5e                   	pop    %esi
    28e2:	c3                   	ret    
    28e3:	83 ec 04             	sub    $0x4,%esp
    28e6:	ff 74 24 18          	pushl  0x18(%esp)
    28ea:	8d 83 ba d5 ff ff    	lea    -0x2a46(%ebx),%eax
    28f0:	50                   	push   %eax
    28f1:	6a 01                	push   $0x1
    28f3:	e8 78 e8 ff ff       	call   1170 <__printf_chk@plt>
    28f8:	66 c7 06 4f 4b       	movw   $0x4b4f,(%esi)
    28fd:	c6 46 02 00          	movb   $0x0,0x2(%esi)
    2901:	83 c4 10             	add    $0x10,%esp
    2904:	b8 00 00 00 00       	mov    $0x0,%eax
    2909:	eb d2                	jmp    28dd <driver_post+0x32>
    290b:	83 ec 04             	sub    $0x4,%esp
    290e:	56                   	push   %esi
    290f:	ff 74 24 1c          	pushl  0x1c(%esp)
    2913:	8d 83 d1 d5 ff ff    	lea    -0x2a2f(%ebx),%eax
    2919:	50                   	push   %eax
    291a:	52                   	push   %edx
    291b:	8d 83 d9 d5 ff ff    	lea    -0x2a27(%ebx),%eax
    2921:	50                   	push   %eax
    2922:	68 6e 3b 00 00       	push   $0x3b6e
    2927:	8d 83 ac d5 ff ff    	lea    -0x2a54(%ebx),%eax
    292d:	50                   	push   %eax
    292e:	e8 5e f5 ff ff       	call   1e91 <submitr>
    2933:	83 c4 20             	add    $0x20,%esp
    2936:	eb a5                	jmp    28dd <driver_post+0x32>
    2938:	66 90                	xchg   %ax,%ax
    293a:	66 90                	xchg   %ax,%ax
    293c:	66 90                	xchg   %ax,%ax
    293e:	66 90                	xchg   %ax,%ax

00002940 <__stack_chk_fail_local>:
    2940:	f3 0f 1e fb          	endbr32 
    2944:	53                   	push   %ebx
    2945:	e8 f6 e8 ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    294a:	81 c3 1a 36 00 00    	add    $0x361a,%ebx
    2950:	83 ec 08             	sub    $0x8,%esp
    2953:	e8 68 e7 ff ff       	call   10c0 <__stack_chk_fail@plt>

Disassembly of section .fini:

00002958 <_fini>:
    2958:	f3 0f 1e fb          	endbr32 
    295c:	53                   	push   %ebx
    295d:	83 ec 08             	sub    $0x8,%esp
    2960:	e8 db e8 ff ff       	call   1240 <__x86.get_pc_thunk.bx>
    2965:	81 c3 ff 35 00 00    	add    $0x35ff,%ebx
    296b:	83 c4 08             	add    $0x8,%esp
    296e:	5b                   	pop    %ebx
    296f:	c3                   	ret    
