# 第五章．实验3：进程管理

### 目录

- [5.1 实验3的基础知识](#fundamental)
  - [5.1.1 多任务环境下进程的封装](#subsec_process_structure)
  - [5.1.2 进程的换入与换出](#subsec_switch)
  - [5.1.3 就绪进程的管理与调度](#subsec_management)
- [5.2 lab3_1 进程创建（fork）](#lab3_1_naive_fork)
  - [给定应用](#lab3_1_app)
  - [实验内容](#lab3_1_content)
  - [实验指导](#lab3_1_guide)
- [5.3 lab3_2 进程yield](#lab3_2_yield)
  - [给定应用](#lab3_2_app)
  - [实验内容](#lab3_2_content)
  - [实验指导](#lab3_2_guide)
- [5.4 lab3_3 循环轮转调度](#lab3_3_rrsched)
  - [给定应用](#lab3_3_app)
  - [实验内容](#lab3_3_content)
  - [实验指导](#lab3_3_guide)
- [5.5 lab3_challenge1 进程等待和数据段复制（难度：&#9733;&#9733;&#9734;&#9734;&#9734;）](#lab3_challenge1_wait)
  - [给定应用](#lab3_challenge1_app)
  - [实验内容](#lab3_challenge1_content)
  - [实验指导](#lab3_challenge1_guide)
- [5.6 lab3_challenge2 实现信号量（难度：&#9733;&#9733;&#9733;&#9734;&#9734;）](#lab3_challenge2_semaphore)
  - [给定应用](#lab3_challenge2_app)
  - [实验内容](#lab3_challenge2_content)
  - [实验指导](#lab3_challenge2_guide)
- [5.7 lab3_challenge3 写时复制（Copy On Write）（难度：&#9733;&#9733;&#9733;&#9734;&#9734;）](#lab3_challenge3_cow)
  - [给定应用](#lab3_challenge3_app)
  - [实验内容](#lab3_challenge3_content)
  - [实验指导](#lab3_challenge3_guide)

`<a name="fundamental"></a>`

## 5.1 实验3的基础知识

完成了实验1和实验2的读者，应该对PKE实验中的“进程”不会太陌生。因为实际上，我们从最开始的lab1_1开始就有了进程结构（struct process），只是在之前的实验中，进程结构中最重要的成员是trapframe和kstack，它们分别用来记录系统进入S模式前的进程上下文以及作为进入S模式后的操作系统栈。在实验3，我们将进入多任务环境，完成PKE实验环境下的进程创建、换入换出，以及进程调度相关实验。

`<a name="subsec_process_structure"></a>`

### 5.1.1 多任务环境下进程的封装

实验3跟之前的两个实验最大的不同，在于在实验3的3个基本实验中，PKE操作系统将需要支持多个进程的执行。为了对多任务环境进行支撑，PKE操作系统定义了一个“进程池”（见kernel/process.c文件）：

```C
 29 // process pool. added @lab3_1
 30 process procs[NPROC];
```

实际上，这个进程池就是一个包含NPROC（=32，见kernel/process.h文件）个process结构的数组。

接下来，PKE操作系统对进程的结构进行了扩充（见kernel/process.h文件）：

```C
73   // points to a page that contains mapped_regions. below are added @lab3_1
74   mapped_region *mapped_info;
75   // next free mapped region in mapped_info
76   int total_mapped_region;
77 
78   // heap management
79   process_heap_manager user_heap;
80 
81   // process id
82   uint64 pid;
83   // process status
84   int status;
85   // parent process
86   struct process_t *parent;
87   // next queue element
88   struct process_t *queue_next;
```

- 前两项mapped_info和total_mapped_region用于对进程的虚拟地址空间（中的代码段、堆栈段等）进行跟踪，这些虚拟地址空间在进程创建（fork）时，将发挥重要作用。同时，这也是lab3_1的内容。PKE将进程可能拥有的段分为以下几个类型：

```C
36 enum segment_type {
37   STACK_SEGMENT = 0,   // runtime stack segment
38   CONTEXT_SEGMENT, // trapframe segment
39   SYSTEM_SEGMENT,  // system segment
40   HEAP_SEGMENT,    // runtime heap segment
41   CODE_SEGMENT,    // ELF segment
42   DATA_SEGMENT,    // ELF segment
43 };
```

其中STACK_SEGMENT为进程自身的栈段，CONTEXT_SEGMENT为保存进程上下文的trapframe所对应的段，SYSTEM_SEGMENT为进程的系统段，如所映射的异常处理段，HEAP_SEGMENT为进程的堆段，CODE_SEGMENT表示该段是从可执行ELF文件中加载的代码段，DATA_SEGMENT为从ELF文件中加载的数据段。

* user_heap是用来管理进程堆段的数据结构，在实验3以及之后的实验中，PKE堆的实现发生了改变。由于多进程的存在，进程的堆不能再简单地实现为一个全局的数组。现在，每个进程有了一个专属的堆段：HEAP_SEGMENT，并新增了user_heap成员队进程的堆区进行管理。该结构体类型定义如下（位于kernel/process.h）：

```c
52 typedef struct process_heap_manager {
53   // points to the last free page in our simple heap.
54   uint64 heap_top;
55   // points to the bottom of our simple heap.
56   uint64 heap_bottom;
57 
58   // the address of free pages in the heap
59   uint64 free_pages_address[MAX_HEAP_PAGES];
60   // the number of free pages in the heap
61   uint32 free_pages_count;
62 }process_heap_manager;
```

该结构维护了当前进程堆的堆顶（heap_top）和堆底（heap_bottom），以及一个用来回收空闲块的数组（free_pages_address）。user_heap的初始化过程见后文alloc_process函数。另外，读者可以阅读位于kernel/syscall.c下的新的sys_user_allocate_page和sys_user_free_page系统调用，便于理解进程的堆区是如何维护的。

sys_user_allocate_page的函数定义如下：

```c
45 uint64 sys_user_allocate_page() {
46   void* pa = alloc_page();
47   uint64 va;
48   // if there are previously reclaimed pages, use them first (this does not change the
49   // size of the heap)
50   if (current->user_heap.free_pages_count > 0) {
51     va =  current->user_heap.free_pages_address[--current->user_heap.free_pages_count];
52     assert(va < current->user_heap.heap_top);
53   } else {
54     // otherwise, allocate a new page (this increases the size of the heap by one page)
55     va = current->user_heap.heap_top;
56     current->user_heap.heap_top += PGSIZE;
57 
58     current->mapped_info[HEAP_SEGMENT].npages++;
59   }
60   user_vm_map((pagetable_t)current->pagetable, va, PGSIZE, (uint64)pa,
61          prot_to_type(PROT_WRITE | PROT_READ, 1));
62 
63   return va;
64 }
```

在sys_user_allocate_page中，当用户尝试从堆上分配一块内存时，会首先在free_pages_address查看是否有被回收的块。如果有，则直接从free_pages_address中分配；如果没有，则扩展堆顶指针heap_top，分配新的页。

sys_user_free_page的函数定义如下：

```c
69 uint64 sys_user_free_page(uint64 va) {
70   user_vm_unmap((pagetable_t)current->pagetable, va, PGSIZE, 1);
71   // add the reclaimed page to the free page list
72   current->user_heap.free_pages_address[current->user_heap.free_pages_count++] = va;
73   return 0;
74 }
```

在sys_user_free_page中，当堆上的页被释放时，对应的物理页会被释放，并将相应的虚拟页地址暂存在free_pages_address数组中。

- pid是进程的ID号，具有唯一性；
- status记录了进程的状态，PKE操作系统在实验3给进程规定了以下几种状态：

```C
27 enum proc_status {
28   FREE,            // unused state
29   READY,           // ready state
30   RUNNING,         // currently running
31   BLOCKED,         // waiting for something
32   ZOMBIE,          // terminated but not reclaimed yet
33 };
```

其中，FREE为自由态，表示进程结构可用；READY为就绪态，即进程所需的资源都已准备好，可以被调度执行；RUNNING表示该进程处于正在运行的状态；BLOCKED表示进程处于阻塞状态；ZOMBIE表示进程处于“僵尸”状态，进程的资源可以被释放和回收。

- parent用于记录进程的父进程；
- queue_next用于将进程链接进各类队列（比如就绪队列）；
- tick_count用于对进程进行记账，即记录它的执行经历了多少次的timer事件，将在lab3_3中实现循环轮转调度时使用。

`<a name="subsec_switch"></a>`

### 5.1.2 进程的启动与终止

PKE实验中，创建一个进程需要先调用kernel/process.c文件中的alloc_process()函数：

```C
 89 process* alloc_process() {
 90   // locate the first usable process structure
 91   int i;
 92 
 93   for( i=0; i<NPROC; i++ )
 94     if( procs[i].status == FREE ) break;
 95 
 96   if( i>=NPROC ){
 97     panic( "cannot find any free process structure.\n" );
 98     return 0;
 99   }
100 
101   // init proc[i]'s vm space
102   procs[i].trapframe = (trapframe *)alloc_page();  //trapframe, used to save context
103   memset(procs[i].trapframe, 0, sizeof(trapframe));
104 
105   // page directory
106   procs[i].pagetable = (pagetable_t)alloc_page();
107   memset((void *)procs[i].pagetable, 0, PGSIZE);
108 
109   procs[i].kstack = (uint64)alloc_page() + PGSIZE;   //user kernel stack top
110   uint64 user_stack = (uint64)alloc_page();       //phisical address of user stack bottom
111   procs[i].trapframe->regs.sp = USER_STACK_TOP;  //virtual address of user stack top
112 
113   // allocates a page to record memory regions (segments)
114   procs[i].mapped_info = (mapped_region*)alloc_page();
115   memset( procs[i].mapped_info, 0, PGSIZE );
116 
117   // map user stack in userspace
118   user_vm_map((pagetable_t)procs[i].pagetable, USER_STACK_TOP - PGSIZE, PGSIZE,
119     user_stack, prot_to_type(PROT_WRITE | PROT_READ, 1));
120   procs[i].mapped_info[STACK_SEGMENT].va = USER_STACK_TOP - PGSIZE;
121   procs[i].mapped_info[STACK_SEGMENT].npages = 1;
122   procs[i].mapped_info[STACK_SEGMENT].seg_type = STACK_SEGMENT;
123 
124   // map trapframe in user space (direct mapping as in kernel space).
125   user_vm_map((pagetable_t)procs[i].pagetable, (uint64)procs[i].trapframe, PGSIZE,
126     (uint64)procs[i].trapframe, prot_to_type(PROT_WRITE | PROT_READ, 0));
127   procs[i].mapped_info[CONTEXT_SEGMENT].va = (uint64)procs[i].trapframe;
128   procs[i].mapped_info[CONTEXT_SEGMENT].npages = 1;
129   procs[i].mapped_info[CONTEXT_SEGMENT].seg_type = CONTEXT_SEGMENT;
130 
131   // map S-mode trap vector section in user space (direct mapping as in kernel space)
132   // we assume that the size of usertrap.S is smaller than a page.
133   user_vm_map((pagetable_t)procs[i].pagetable, (uint64)trap_sec_start, PGSIZE,
134     (uint64)trap_sec_start, prot_to_type(PROT_READ | PROT_EXEC, 0));
135   procs[i].mapped_info[SYSTEM_SEGMENT].va = (uint64)trap_sec_start;
136   procs[i].mapped_info[SYSTEM_SEGMENT].npages = 1;
137   procs[i].mapped_info[SYSTEM_SEGMENT].seg_type = SYSTEM_SEGMENT;
138 
139   sprint("in alloc_proc. user frame 0x%lx, user stack 0x%lx, user kstack 0x%lx \n",
140     procs[i].trapframe, procs[i].trapframe->regs.sp, procs[i].kstack);
141 
142   // initialize the process's heap manager
143   procs[i].user_heap.heap_top = USER_FREE_ADDRESS_START;
144   procs[i].user_heap.heap_bottom = USER_FREE_ADDRESS_START;
145   procs[i].user_heap.free_pages_count = 0;
146 
147   // map user heap in userspace
148   procs[i].mapped_info[HEAP_SEGMENT].va = USER_FREE_ADDRESS_START;
149   procs[i].mapped_info[HEAP_SEGMENT].npages = 0;  // no pages are mapped to heap yet.
150   procs[i].mapped_info[HEAP_SEGMENT].seg_type = HEAP_SEGMENT;
151 
152   procs[i].total_mapped_region = 4;
153 
154   // return after initialization.
155   return &procs[i];
156 }
```

通过以上代码，可以发现alloc_process()函数除了找到一个空的进程结构外，还为新创建的进程建立了KERN_BASE以上逻辑地址的映射（这段代码在实验3之前位于kernel/kernel.c文件的load_user_program()函数中），并将映射信息保存到了进程结构中。在本实验中还额外添加了HEAP_SEGMENT段的映射（第148--150行）。同时可以看出，对于未调用sys_user_allocate_page分配堆区页面的进程，其堆段的大小为0（第149行），不会被分配页面。

对于给定应用，PKE将通过调用load_bincode_from_host_elf()函数载入给定应用对应的ELF文件的各个段。之后被调用的elf_load()函数在载入段后，将对被载入的段进行判断，以记录它们的虚地址映射：

```c
 65 elf_status elf_load(elf_ctx *ctx) {
 66   // elf_prog_header structure is defined in kernel/elf.h
 67   elf_prog_header ph_addr;
 68   int i, off;
 69
 70   // traverse the elf program segment headers
 71   for (i = 0, off = ctx->ehdr.phoff; i < ctx->ehdr.phnum; i++, off += sizeof(ph_addr)) {
 72     // read segment headers
 73     if (elf_fpread(ctx, (void *)&ph_addr, sizeof(ph_addr), off) != sizeof(ph_addr)) return EL    _EIO;
 74
 75     if (ph_addr.type != ELF_PROG_LOAD) continue;
 76     if (ph_addr.memsz < ph_addr.filesz) return EL_ERR;
 77     if (ph_addr.vaddr + ph_addr.memsz < ph_addr.vaddr) return EL_ERR;
 78
 79     // allocate memory block before elf loading
 80     void *dest = elf_alloc_mb(ctx, ph_addr.vaddr, ph_addr.vaddr, ph_addr.memsz);
 81
 82     // actual loading
 83     if (elf_fpread(ctx, dest, ph_addr.memsz, ph_addr.off) != ph_addr.memsz)
 84       return EL_EIO;
 85
 86     // record the vm region in proc->mapped_info. added @lab3_1
 87     int j;
 88     for( j=0; j<PGSIZE/sizeof(mapped_region); j++ ) //seek the last mapped region
 89       if( (process*)(((elf_info*)(ctx->info))->p)->mapped_info[j].va == 0x0 ) break;
 90
 91     ((process*)(((elf_info*)(ctx->info))->p))->mapped_info[j].va = ph_addr.vaddr;
 92     ((process*)(((elf_info*)(ctx->info))->p))->mapped_info[j].npages = 1;
 93
 94     // SEGMENT_READABLE, SEGMENT_EXECUTABLE, SEGMENT_WRITABLE are defined in kernel/elf.h
 95     if( ph_addr.flags == (SEGMENT_READABLE|SEGMENT_EXECUTABLE) ){
 96       ((process*)(((elf_info*)(ctx->info))->p))->mapped_info[j].seg_type = CODE_SEGMENT;
 97       sprint( "CODE_SEGMENT added at mapped info offset:%d\n", j );
 98     }else if ( ph_addr.flags == (SEGMENT_READABLE|SEGMENT_WRITABLE) ){
 99       ((process*)(((elf_info*)(ctx->info))->p))->mapped_info[j].seg_type = DATA_SEGMENT;
100       sprint( "DATA_SEGMENT added at mapped info offset:%d\n", j );
101     }else
102       panic( "unknown program segment encountered, segment flag:%d.\n", ph_addr.flags );
103
104     ((process*)(((elf_info*)(ctx->info))->p))->total_mapped_region ++;
105   }
106
107   return EL_OK;
108 }
```

以上代码段中，第95--102行将对被载入的段的类型（ph_addr.flags）进行判断以确定它是代码段还是数据段。完成以上的虚地址空间到物理地址空间的映射后，将形成用户进程的虚地址空间结构（如[图4.5](chapter4_memory.md#user_vm_space)所示）。

接下来，将通过switch_to()函数将所构造的进程投入执行：

```c
38 void switch_to(process* proc) {
39   assert(proc);
40   current = proc;
41 
42   // write the smode_trap_vector (64-bit func. address) defined in kernel/strap_vector.S
43   // to the stvec privilege register, such that trap handler pointed by smode_trap_vector
44   // will be triggered when an interrupt occurs in S mode.
45   write_csr(stvec, (uint64)smode_trap_vector);
46 
47   // set up trapframe values (in process structure) that smode_trap_vector will need when
48   // the process next re-enters the kernel.
49   proc->trapframe->kernel_sp = proc->kstack;      // process's kernel stack
50   proc->trapframe->kernel_satp = read_csr(satp);  // kernel page table
51   proc->trapframe->kernel_trap = (uint64)smode_trap_handler;
52 
53   // SSTATUS_SPP and SSTATUS_SPIE are defined in kernel/riscv.h
54   // set S Previous Privilege mode (the SSTATUS_SPP bit in sstatus register) to User mode.
55   unsigned long x = read_csr(sstatus);
56   x &= ~SSTATUS_SPP;  // clear SPP to 0 for user mode
57   x |= SSTATUS_SPIE;  // enable interrupts in user mode
58 
59   // write x back to 'sstatus' register to enable interrupts, and sret destination mode.
60   write_csr(sstatus, x);
61 
62   // set S Exception Program Counter (sepc register) to the elf entry pc.
63   write_csr(sepc, proc->trapframe->epc);
64 
65   // make user page table. macro MAKE_SATP is defined in kernel/riscv.h. added @lab2_1
66   uint64 user_satp = MAKE_SATP(proc->pagetable);
67 
68   // return_to_user() is defined in kernel/strap_vector.S. switch to user mode with sret.
69   // note, return_to_user takes two parameters @ and after lab2_1.
70   return_to_user(proc->trapframe, user_satp);
71 }
```

实际上，以上函数在[实验1](chapter3_traps.md)就有所涉及，它的作用是将进程结构中的trapframe作为进程上下文恢复到RISC-V机器的通用寄存器中，并最后调用sret指令（通过return_to_user()函数）将进程投入执行。

不同于实验1和实验2，实验3的exit系统调用不能够直接将系统shutdown，因为一个进程的结束并不一定意味着系统中所有进程的完成。以下是实验3中exit系统调用的实现：

```c
 34 ssize_t sys_user_exit(uint64 code) {
 35   sprint("User exit with code:%d.\n", code);
 36   // reclaim the current process, and reschedule. added @lab3_1
 37   free_process( current );
 38   schedule();
 39   return 0;
 40 }
```

可以看到，如果某进程调用了exit()系统调用，操作系统的处理方法是调用free_process()函数，将当前进程（也就是调用者）进行“释放”，然后转进程调度。其中free_process()函数（kernel/process.c文件）的实现非常简单：

```c
161 int free_process( process* proc ) {
162   // we set the status to ZOMBIE, but cannot destruct its vm space immediately.
163   // since proc can be current process, and its user kernel stack is currently in use!
164   // but for proxy kernel, it (memory leaking) may NOT be a really serious issue,
165   // as it is different from regular OS, which needs to run 7x24.
166   proc->status = ZOMBIE;
167 
168   return 0;
169 }
```

可以看到，**free_process()函数仅是将进程设为ZOMBIE状态，而不会将进程所占用的资源全部释放**！这是因为free_process()函数的调用，说明操作系统当前是在S模式下运行，而按照PKE的设计思想，S态的运行将使用当前进程的用户系统栈（user kernel stack）。此时，如果将当前进程的内存空间进行释放，将导致操作系统本身的崩溃。所以释放进程时，PKE采用的是折衷的办法，即只将其设置为僵尸（ZOMBIE）状态，而不是立即将它所占用的资源进行释放。最后，schedule()函数的调用，将选择系统中可能存在的其他处于就绪状态的进程投入运行，它的处理逻辑我们将在下一节讨论。

`<a name="subsec_management"></a>`

### 5.1.3 就绪进程的管理与调度

PKE的操作系统设计了一个非常简单的就绪队列管理（因为实验3的基础实验并未涉及进程的阻塞，所以未设计阻塞队列），队列头在kernel/sched.c文件中定义：

```
8 process* ready_queue_head = NULL;
```

将一个进程加入就绪队列，可以调用insert_to_ready_queue()函数：

```c
 13 void insert_to_ready_queue( process* proc ) {
 14   sprint( "going to insert process %d to ready queue.\n", proc->pid );
 15   // if the queue is empty in the beginning
 16   if( ready_queue_head == NULL ){
 17     proc->status = READY;
 18     proc->queue_next = NULL;
 19     ready_queue_head = proc;
 20     return;
 21   }
 22
 23   // ready queue is not empty
 24   process *p;
 25   // browse the ready queue to see if proc is already in-queue
 26   for( p=ready_queue_head; p->queue_next!=NULL; p=p->queue_next )
 27     if( p == proc ) return;  //already in queue
 28
 29   // p points to the last element of the ready queue
 30   if( p==proc ) return;
 31   p->queue_next = proc;
 32   proc->status = READY;
 33   proc->queue_next = NULL;
 34
 35   return;
 36 }
```

该函数首先（第16--21行）处理ready_queue_head为空（初始状态）的情况，如果就绪队列不为空，则将进程加入到队尾（第26--33行）。

PKE操作系统内核通过调用schedule()函数来完成进程的选择和换入：

```c
 45 void schedule() {
 46   if ( !ready_queue_head ){
 47     // by default, if there are no ready process, and all processes are in the status of
 48     // FREE and ZOMBIE, we should shutdown the emulated RISC-V machine.
 49     int should_shutdown = 1;
 50
 51     for( int i=0; i<NPROC; i++ )
 52       if( (procs[i].status != FREE) && (procs[i].status != ZOMBIE) ){
 53         should_shutdown = 0;
 54         sprint( "ready queue empty, but process %d is not in free/zombie state:%d\n",
 55           i, procs[i].status );
 56       }
 57
 58     if( should_shutdown ){
 59       sprint( "no more ready processes, system shutdown now.\n" );
 60       shutdown( 0 );
 61     }else{
 62       panic( "Not handled: we should let system wait for unfinished processes.\n" );
 63     }
 64   }
 65
 66   current = ready_queue_head;
 67   assert( current->status == READY );
 68   ready_queue_head = ready_queue_head->queue_next;
 69
 70   current->status == RUNNING;
 71   sprint( "going to schedule process %d to run.\n", current->pid );
 72   switch_to( current );
 73 }
```

可以看到，schedule()函数首先判断就绪队列ready_queue_head是否为空，对于为空的情况（第46--64行），schedule()函数将判断系统中所有的进程是否全部都处于被释放（FREE）状态，或者僵尸（ZOMBIE）状态。如果是，则启动关（模拟RISC-V）机程序，否则应进入等待系统中进程结束的状态。但是，由于实验3的基础实验并无可能进入这样的状态，所以我们在这里调用了panic，等后续实验有可能进入这种状态后再进一步处理。

对于就绪队列非空的情况（第66--72行），处理就简单得多：只需要将就绪队列队首的进程换入执行即可。对于换入的过程，需要注意的是，要将被选中的进程从就绪队列中摘掉。

`<a name="lab3_1_naive_fork"></a>`

## 5.2 lab3_1 进程创建（fork）（[头歌实验链接](https://www.educoder.net/shixuns/lf3vgx7c/challenges)）

`<a name="lab3_1_app"></a>`

#### **给定应用**

- user/app_naive_fork.c

```c
  1 /*
  2  * The application of lab3_1.
  3  * it simply forks a child process.
  4  */
  5
  6 #include "user/user_lib.h"
  7 #include "util/types.h"
  8
  9 int main(void) {
 10   uint64 pid = fork();
 11   if (pid == 0) {
 12     printu("Child: Hello world!\n");
 13   } else {
 14     printu("Parent: Hello world! child id %ld\n", pid);
 15   }
 16
 17   exit(0);
 18 }
```

以上程序的行为非常简单：主进程调用fork()函数，后者产生一个系统调用，基于主进程这个模板创建它的子进程。

- （先提交lab2_3的答案，然后）切换到lab3_1，继承lab2_3及之前实验所做的修改，并make后的直接运行结果：

```bash
//切换到lab3_1
$ git checkout lab3_1_fork

//继承lab2_3以及之前的答案
$ git merge lab2_3_pagefault -m "continue to work on lab3_1"

//重新构造
$ make clean; make

//运行构造结果
$ spike ./obj/riscv-pke ./obj/app_naive_fork
In m_start, hartid:0
HTIF is available!
(Emulated) memory size: 2048 MB
Enter supervisor mode...
PKE kernel start 0x0000000080000000, PKE kernel end: 0x0000000080010000, PKE kernel size: 0x0000000000010000 .
free physical memory address: [0x0000000080010000, 0x0000000087ffffff]
kernel memory manager is initializing ...
KERN_BASE 0x0000000080000000
physical address of _etext is: 0x0000000080005000
kernel page table is on
Switching to user mode...
in alloc_proc. user frame 0x0000000087fbc000, user stack 0x000000007ffff000, user kstack 0x0000000087fbb000
User application is loading.
Application: ./obj/app_naive_fork
CODE_SEGMENT added at mapped info offset:3
Application program entry point (virtual address): 0x0000000000010078
going to insert process 0 to ready queue.
going to schedule process 0 to run.
User call fork.
will fork a child from parent 0.
in alloc_proc. user frame 0x0000000087faf000, user stack 0x000000007ffff000, user kstack 0x0000000087fae000
You need to implement the code segment mapping of child in lab3_1.

System is shutting down with exit code -1.
```

从以上运行结果来看，应用程序的fork动作并未将子进程给创建出来并投入运行。按照提示，我们需要在PKE操作系统内核中实现子进程到父进程代码段的映射，以最终完成fork动作。

这里，既然涉及到了父进程的代码段，我们就可以先用readelf命令查看一下给定应用程序的可执行代码对应的ELF文件结构：

```bash
$ riscv64-unknown-elf-readelf -l ./obj/app_naive_fork

Elf file type is EXEC (Executable file)
Entry point 0x10078
There is 1 program header, starting at offset 64

Program Headers:
  Type           Offset             VirtAddr           PhysAddr
                 FileSiz            MemSiz              Flags  Align
  LOAD           0x0000000000000000 0x0000000000010000 0x0000000000010000
                 0x000000000000040c 0x000000000000040c  R E    0x1000

 Section to Segment mapping:
  Segment Sections...
   00     .text .rodata
```

可以看到，app_naive_fork可执行文件只包含一个代码段（编号为00），应该来讲是最简单的可执行文件结构了（无须考虑数据段的问题）。如果要依据这样的父进程模板创建子进程，只需要将它的代码段映射（而非拷贝）到子进程的对应虚地址即可。

`<a name="lab3_1_content"></a>`

#### **实验内容**

完善操作系统内核kernel/process.c文件中的do_fork()函数，并最终获得以下预期结果：

```bash
$ spike ./obj/riscv-pke ./obj/app_naive_fork
In m_start, hartid:0
HTIF is available!
(Emulated) memory size: 2048 MB
Enter supervisor mode...
PKE kernel start 0x0000000080000000, PKE kernel end: 0x0000000080010000, PKE kernel size: 0x0000000000010000 .
free physical memory address: [0x0000000080010000, 0x0000000087ffffff]
kernel memory manager is initializing ...
KERN_BASE 0x0000000080000000
physical address of _etext is: 0x0000000080005000
kernel page table is on
Switching to user mode...
in alloc_proc. user frame 0x0000000087fbc000, user stack 0x000000007ffff000, user kstack 0x0000000087fbb000
User application is loading.
Application: ./obj/app_naive_fork
CODE_SEGMENT added at mapped info offset:3
Application program entry point (virtual address): 0x0000000000010078
going to insert process 0 to ready queue.
going to schedule process 0 to run.
User call fork.
will fork a child from parent 0.
in alloc_proc. user frame 0x0000000087faf000, user stack 0x000000007ffff000, user kstack 0x0000000087fae000
do_fork map code segment at pa:0000000087fb2000 of parent to child at va:0000000000010000.
going to insert process 1 to ready queue.
Parent: Hello world! child id 1
User exit with code:0.
going to schedule process 1 to run.
Child: Hello world!
User exit with code:0.
no more ready processes, system shutdown now.
System is shutting down with exit code 0.
```

从以上运行结果来看，子进程已经被创建，且在其后被投入运行。

`<a name="lab3_1_guide"></a>`

#### **实验指导**

读者可以回顾[lab1_1](chapter3_traps.md#syscall)中所学习到的系统调用的知识，从应用程序（user/app_naive_fork.c）开始，跟踪fork()函数的实现：

user/app_naive_fork.c --> user/user_lib.c --> kernel/strap_vector.S --> kernel/strap.c --> kernel/syscall.c

直至跟踪到kernel/process.c文件中的do_fork()函数：

```c
178 int do_fork( process* parent)
179 {
180   sprint( "will fork a child from parent %d.\n", parent->pid );
181   process* child = alloc_process();
182 
183   for( int i=0; i<parent->total_mapped_region; i++ ){
184     // browse parent's vm space, and copy its trapframe and data segments,
185     // map its code segment.
186     switch( parent->mapped_info[i].seg_type ){
187       case CONTEXT_SEGMENT:
188         *child->trapframe = *parent->trapframe;
189         break;
190       case STACK_SEGMENT:
191         memcpy( (void*)lookup_pa(child->pagetable, child->mapped_info[STACK_SEGMENT].va),
192           (void*)lookup_pa(parent->pagetable, parent->mapped_info[i].va), PGSIZE );
193         break;
194       case HEAP_SEGMENT:
195         // build a same heap for child process.
196 
197         // convert free_pages_address into a filter to skip reclaimed blocks in the heap
198         // when mapping the heap blocks
199         int free_block_filter[MAX_HEAP_PAGES];
200         memset(free_block_filter, 0, MAX_HEAP_PAGES);
201         uint64 heap_bottom = parent->user_heap.heap_bottom;
202         for (int i = 0; i < parent->user_heap.free_pages_count; i++) {
203           int index = (parent->user_heap.free_pages_address[i] - heap_bottom) / PGSIZE;
204           free_block_filter[index] = 1;
205         }
206 
207         // copy and map the heap blocks
208         for (uint64 heap_block = current->user_heap.heap_bottom;
209              heap_block < current->user_heap.heap_top; heap_block += PGSIZE) {
210           if (free_block_filter[(heap_block - heap_bottom) / PGSIZE])  // skip free blocks
211             continue;
212 
213           void* child_pa = alloc_page();
214           memcpy(child_pa, (void*)lookup_pa(parent->pagetable, heap_block), PGSIZE);
215           user_vm_map((pagetable_t)child->pagetable, heap_block, PGSIZE, (uint64)child_pa,
216                       prot_to_type(PROT_WRITE | PROT_READ, 1));
217         }
218 
219         child->mapped_info[HEAP_SEGMENT].npages = parent->mapped_info[HEAP_SEGMENT].npages;
220 
221         // copy the heap manager from parent to child
222         memcpy((void*)&child->user_heap, (void*)&parent->user_heap, sizeof(parent->user_heap));
223         break;
224       case CODE_SEGMENT:
225         // TODO (lab3_1): implment the mapping of child code segment to parent's
226         // code segment.
227         // hint: the virtual address mapping of code segment is tracked in mapped_info
228         // page of parent's process structure. use the information in mapped_info to
229         // retrieve the virtual to physical mapping of code segment.
230         // after having the mapping information, just map the corresponding virtual
231         // address region of child to the physical pages that actually store the code
232         // segment of parent process.
233         // DO NOT COPY THE PHYSICAL PAGES, JUST MAP THEM.
234         panic( "You need to implement the code segment mapping of child in lab3_1.\n" );
235 
236         // after mapping, register the vm region (do not delete codes below!)
237         child->mapped_info[child->total_mapped_region].va = parent->mapped_info[i].va;
238         child->mapped_info[child->total_mapped_region].npages =
239           parent->mapped_info[i].npages;
240         child->mapped_info[child->total_mapped_region].seg_type = CODE_SEGMENT;
241         child->total_mapped_region++;
242         break;
243     }
244   }
245 
246   child->status = READY;
247   child->trapframe->regs.a0 = 0;
248   child->parent = parent;
249   insert_to_ready_queue( child );
250 
251   return child->pid;
252 }
```

该函数使用第183--244行的循环来拷贝父进程的逻辑地址空间到其子进程。我们看到，对于trapframe段（case CONTEXT_SEGMENT）以及栈段（case STACK_SEGMENT），do_fork()函数采用了简单复制的办法来拷贝父进程的这两个段到子进程中，这样做的目的是将父进程的执行现场传递给子进程。对于堆段（case HEAP_SEGMENT），堆中存在两种状态的虚拟页面：未被释放的页面（位于当前进程的user_heap.heap_bottom与user_heap.heap_top之间，但不在user_heap.free_pages_address数组中）和已被释放的页面（位于当前进程的user_heap.free_pages_address数组中），其中已被释放的页面不需要额外处理。而对于父进程中每一页未被释放的虚拟页面，需要首先分配一页空闲物理页，再将父进程中该页数据进行拷贝（第213--214行），然后将新分配的物理页映射到子进程的相同虚页（第215行）。处理完堆中的所有页面后，还需要拷贝父进程的mapped_info[HEAP_SEGMENT]和user_heap信息到其子进程（第219--222行）。

然而，对于父进程的代码段，子进程应该如何“继承”呢？通过第225--233行的注释，我们知道对于代码段，我们不应直接复制（减少系统开销），而应通过映射的办法，将子进程中对应的逻辑地址空间映射到其父进程中装载代码段的物理页面。这里，就要回到[实验2内存管理](chapter4_memory.md#pagetablecook)部分，寻找合适的函数来实现了。注意对页面的权限设置（可读可执行）。

**实验完毕后，记得提交修改（命令行中-m后的字符串可自行确定），以便在后续实验中继承lab3_1中所做的工作**：

```bash
$ git commit -a -m "my work on lab3_1 is done."
```

#### **参考答案**

请参考本文档的[纸质书](https://www.ryjiaoyu.com/book/details/50200)。

`<a name="lab3_2_yield"></a>`

## 5.3 lab3_2 进程yield（[头歌实验链接](https://www.educoder.net/shixuns/e7q4csmt/challenges)）

`<a name="lab3_2_app"></a>`

#### **给定应用**

- user/app_yield.c

```C
  1 /*
  2  * The application of lab3_2.
  3  * parent and child processes intermittently give up their processors.
  4  */
  5
  6 #include "user/user_lib.h"
  7 #include "util/types.h"
  8
  9 int main(void) {
 10   uint64 pid = fork();
 11   uint64 rounds = 0xffff;
 12   if (pid == 0) {
 13     printu("Child: Hello world! \n");
 14     for (uint64 i = 0; i < rounds; ++i) {
 15       if (i % 10000 == 0) {
 16         printu("Child running %ld \n", i);
 17         yield();
 18       }
 19     }
 20   } else {
 21     printu("Parent: Hello world! \n");
 22     for (uint64 i = 0; i < rounds; ++i) {
 23       if (i % 10000 == 0) {
 24         printu("Parent running %ld \n", i);
 25         yield();
 26       }
 27     }
 28   }
 29
 30   exit(0);
 31   return 0;
 32 }
```

和lab3_1一样，以上的应用程序通过fork系统调用创建了一个子进程，接下来，父进程和子进程都进入了一个很长的循环。在循环中，无论是父进程还是子进程，在循环的次数是10000的整数倍时，除了打印信息外都调用了yield()函数，来释放自己的执行权（即CPU）。

- （先提交lab3_1的答案，然后）切换到lab3_2，继承lab3_1及之前实验所做的修改，并make后的直接运行结果：

```bash
//切换到lab3_2
$ git checkout lab3_2_yield

//继承lab3_1以及之前的答案
$ git merge lab3_1_fork -m "continue to work on lab3_2"

//重新构造
$ make clean; make

//运行构造结果
$ spike ./obj/riscv-pke ./obj/app_yield
In m_start, hartid:0
HTIF is available!
(Emulated) memory size: 2048 MB
Enter supervisor mode...
PKE kernel start 0x0000000080000000, PKE kernel end: 0x0000000080010000, PKE kernel size: 0x0000000000010000 .
free physical memory address: [0x0000000080010000, 0x0000000087ffffff]
kernel memory manager is initializing ...
KERN_BASE 0x0000000080000000
physical address of _etext is: 0x0000000080005000
kernel page table is on
Switching to user mode...
in alloc_proc. user frame 0x0000000087fbc000, user stack 0x000000007ffff000, user kstack 0x0000000087fbb000
User application is loading.
Application: ./obj/app_yield
CODE_SEGMENT added at mapped info offset:3
Application program entry point (virtual address): 0x000000000001017c
going to insert process 0 to ready queue.
going to schedule process 0 to run.
User call fork.
will fork a child from parent 0.
in alloc_proc. user frame 0x0000000087faf000, user stack 0x000000007ffff000, user kstack 0x0000000087fae000
do_fork map code segment at pa:0000000087fb2000 of parent to child at va:0000000000010000.
going to insert process 1 to ready queue.
Parent: Hello world!
Parent running 0
You need to implement the yield syscall in lab3_2.

System is shutting down with exit code -1.
```

从以上输出来看，还是因为PKE操作系统中的yield()功能未完善，导致应用无法正常执行下去。

`<a name="lab3_2_content"></a>`

#### **实验内容**

完善yield系统调用，实现进程执行过程中的主动释放CPU的动作。实验完成后，获得以下预期结果：

```bash
$ spike ./obj/riscv-pke ./obj/app_yield
In m_start, hartid:0
HTIF is available!
(Emulated) memory size: 2048 MB
Enter supervisor mode...
PKE kernel start 0x0000000080000000, PKE kernel end: 0x0000000080010000, PKE kernel size: 0x0000000000010000 .
free physical memory address: [0x0000000080010000, 0x0000000087ffffff]
kernel memory manager is initializing ...
KERN_BASE 0x0000000080000000
physical address of _etext is: 0x0000000080005000
kernel page table is on
Switching to user mode...
in alloc_proc. user frame 0x0000000087fbc000, user stack 0x000000007ffff000, user kstack 0x0000000087fbb000
User application is loading.
Application: ./obj/app_yield
CODE_SEGMENT added at mapped info offset:3
Application program entry point (virtual address): 0x000000000001017c
going to insert process 0 to ready queue.
going to schedule process 0 to run.
User call fork.
will fork a child from parent 0.
in alloc_proc. user frame 0x0000000087faf000, user stack 0x000000007ffff000, user kstack 0x0000000087fae000
do_fork map code segment at pa:0000000087fb2000 of parent to child at va:0000000000010000.
going to insert process 1 to ready queue.
Parent: Hello world!
Parent running 0
going to insert process 0 to ready queue.
going to schedule process 1 to run.
Child: Hello world!
Child running 0
going to insert process 1 to ready queue.
going to schedule process 0 to run.
Parent running 10000
going to insert process 0 to ready queue.
going to schedule process 1 to run.
Child running 10000
going to insert process 1 to ready queue.
going to schedule process 0 to run.
Parent running 20000
going to insert process 0 to ready queue.
going to schedule process 1 to run.
Child running 20000
going to insert process 1 to ready queue.
going to schedule process 0 to run.
Parent running 30000
going to insert process 0 to ready queue.
going to schedule process 1 to run.
Child running 30000
going to insert process 1 to ready queue.
going to schedule process 0 to run.
Parent running 40000
going to insert process 0 to ready queue.
going to schedule process 1 to run.
Child running 40000
going to insert process 1 to ready queue.
going to schedule process 0 to run.
Parent running 50000
going to insert process 0 to ready queue.
going to schedule process 1 to run.
Child running 50000
going to insert process 1 to ready queue.
going to schedule process 0 to run.
Parent running 60000
going to insert process 0 to ready queue.
going to schedule process 1 to run.
Child running 60000
going to insert process 1 to ready queue.
going to schedule process 0 to run.
User exit with code:0.
going to schedule process 1 to run.
User exit with code:0.
no more ready processes, system shutdown now.
System is shutting down with exit code 0.
```

`<a name="lab3_2_guide"></a>`

#### **实验指导**

进程释放CPU的动作应该是：

- 将当前进程置为就绪状态（READY）；
- 将当前进程加入到就绪队列的队尾；
- 转进程调度。

**实验完毕后，记得提交修改（命令行中-m后的字符串可自行确定），以便在后续实验中继承lab3_2中所做的工作**：

```bash
$ git commit -a -m "my work on lab3_2 is done."
```

#### **参考答案**

请参考本文档的[纸质书](https://www.ryjiaoyu.com/book/details/50200)。

`<a name="lab3_3_rrsched"></a>`

## 5.4 lab3_3 循环轮转调度（[头歌实验链接](https://www.educoder.net/shixuns/3cvhjfw4/challenges)）

`<a name="lab3_3_app"></a>`

#### **给定应用**

```C
  1 /*
  2  * The application of lab3_3.
  3  * parent and child processes never give up their processor during execution.
  4  */
  5
  6 #include "user/user_lib.h"
  7 #include "util/types.h"
  8
  9 int main(void) {
 10   uint64 pid = fork();
 11   uint64 rounds = 100000000;
 12   uint64 interval = 10000000;
 13   uint64 a = 0;
 14   if (pid == 0) {
 15     printu("Child: Hello world! \n");
 16     for (uint64 i = 0; i < rounds; ++i) {
 17       if (i % interval == 0) printu("Child running %ld \n", i);
 18     }
 19   } else {
 20     printu("Parent: Hello world! \n");
 21     for (uint64 i = 0; i < rounds; ++i) {
 22       if (i % interval == 0) printu("Parent running %ld \n", i);
 23     }
 24   }
 25
 26   exit(0);
 27   return 0;
 28 }
```

和lab3_2类似，lab3_3给出的应用仍然是父子两个进程，他们的执行体都是两个大循环。但与lab3_2不同的是，这两个进程在执行各自循环体时，都没有主动释放CPU的动作。显然，这样的设计会导致某个进程长期占据CPU，而另一个进程无法得到执行。

- （先提交lab3_2的答案，然后）切换到lab3_3，继承lab3_2及之前实验所做的修改，并make后的直接运行结果：

```bash
//切换到lab3_3
$ git checkout lab3_3_rrsched

//继承lab3_2以及之前的答案
$ git merge lab3_2_yield -m "continue to work on lab3_3"

//重新构造
$ make clean; make

//运行构造结果
$ spike ./obj/riscv-pke ./obj/app_two_long_loops
In m_start, hartid:0
HTIF is available!
(Emulated) memory size: 2048 MB
Enter supervisor mode...
PKE kernel start 0x0000000080000000, PKE kernel end: 0x0000000080010000, PKE kernel size: 0x0000000000010000 .
free physical memory address: [0x0000000080010000, 0x0000000087ffffff]
kernel memory manager is initializing ...
KERN_BASE 0x0000000080000000
physical address of _etext is: 0x0000000080005000
kernel page table is on
Switching to user mode...
in alloc_proc. user frame 0x0000000087fbc000, user stack 0x000000007ffff000, user kstack 0x0000000087fbb000
User application is loading.
Application: ./obj/app_two_long_loops
CODE_SEGMENT added at mapped info offset:3
Application program entry point (virtual address): 0x000000000001017c
going to insert process 0 to ready queue.
going to schedule process 0 to run.
User call fork.
will fork a child from parent 0.
in alloc_proc. user frame 0x0000000087faf000, user stack 0x000000007ffff000, user kstack 0x0000000087fae000
do_fork map code segment at pa:0000000087fb2000 of parent to child at va:0000000000010000.
going to insert process 1 to ready queue.
Parent: Hello world!
Parent running 0
Parent running 10000000
Ticks 0
You need to further implement the timer handling in lab3_3.

System is shutting down with exit code -1.
```

回顾实验1的[lab1_3](chapter3_traps.md#irq)，我们看到由于进程的执行体很长，执行过程中时钟中断被触发（输出中的“Ticks 0”）。显然，我们可以通过利用时钟中断来实现进程的循环轮转调度，避免由于一个进程的执行体过长，导致系统中其他进程无法得到调度的问题！

`<a name="lab3_3_content"></a>`

#### **实验内容**

实现kernel/strap.c文件中的rrsched()函数，获得以下预期结果：

```bash
$ spike ./obj/riscv-pke ./obj/app_two_long_loops
In m_start, hartid:0
HTIF is available!
(Emulated) memory size: 2048 MB
Enter supervisor mode...
PKE kernel start 0x0000000080000000, PKE kernel end: 0x0000000080010000, PKE kernel size: 0x0000000000010000 .
free physical memory address: [0x0000000080010000, 0x0000000087ffffff]
kernel memory manager is initializing ...
KERN_BASE 0x0000000080000000
physical address of _etext is: 0x0000000080005000
kernel page table is on
Switching to user mode...
in alloc_proc. user frame 0x0000000087fbc000, user stack 0x000000007ffff000, user kstack 0x0000000087fbb000
User application is loading.
Application: ./obj/app_two_long_loops
CODE_SEGMENT added at mapped info offset:3
Application program entry point (virtual address): 0x000000000001017c
going to insert process 0 to ready queue.
going to schedule process 0 to run.
User call fork.
will fork a child from parent 0.
in alloc_proc. user frame 0x0000000087faf000, user stack 0x000000007ffff000, user kstack 0x0000000087fae000
do_fork map code segment at pa:0000000087fb2000 of parent to child at va:0000000000010000.
going to insert process 1 to ready queue.
Parent: Hello world!
Parent running 0
Parent running 10000000
Ticks 0
Parent running 20000000
Ticks 1
going to insert process 0 to ready queue.
going to schedule process 1 to run.
Child: Hello world!
Child running 0
Child running 10000000
Ticks 2
Child running 20000000
Ticks 3
going to insert process 1 to ready queue.
going to schedule process 0 to run.
Parent running 30000000
Ticks 4
Parent running 40000000
Ticks 5
going to insert process 0 to ready queue.
going to schedule process 1 to run.
Child running 30000000
Ticks 6
Child running 40000000
Ticks 7
going to insert process 1 to ready queue.
going to schedule process 0 to run.
Parent running 50000000
Parent running 60000000
Ticks 8
Parent running 70000000
Ticks 9
going to insert process 0 to ready queue.
going to schedule process 1 to run.
Child running 50000000
Child running 60000000
Ticks 10
Child running 70000000
Ticks 11
going to insert process 1 to ready queue.
going to schedule process 0 to run.
Parent running 80000000
Ticks 12
Parent running 90000000
Ticks 13
going to insert process 0 to ready queue.
going to schedule process 1 to run.
Child running 80000000
Ticks 14
Child running 90000000
Ticks 15
going to insert process 1 to ready queue.
going to schedule process 0 to run.
User exit with code:0.
going to schedule process 1 to run.
User exit with code:0.
no more ready processes, system shutdown now.
System is shutting down with exit code 0.
```

`<a name="lab3_3_guide"></a>`

#### **实验指导**

实际上，如果单纯为了实现进程的轮转，避免单个进程长期霸占CPU的情况，只需要简单地在时钟中断被触发时做重新调度即可。然而，为了实现时间片的概念，以及控制进程在单时间片内获得的执行长度，我们在kernel/sched.h文件中定义了“时间片”的长度：

```C
  6 //length of a time slice, in number of ticks
  7 #define TIME_SLICE_LEN  2
```

可以看到时间片的长度（TIME_SLICE_LEN）为2个ticks，这就意味着我们要每隔两个ticks触发一次进程重新调度动作。

为配合调度的实现，我们在进程结构中定义了整型成员（参见[5.1.1](#subsec_process_structure)）tick_count，完善kernel/strap.c文件中的rrsched()函数，以实现循环轮转调度时，应采取的逻辑为：

- 判断当前进程的tick_count加1后是否大于等于TIME_SLICE_LEN？
- 若答案为yes，则应将当前进程的tick_count清零，并将当前进程加入就绪队列，转进程调度；
- 若答案为no，则应将当前进程的tick_count加1，并返回。

**实验完毕后，记得提交修改（命令行中-m后的字符串可自行确定），以便在后续实验中继承lab3_3中所做的工作**：

```bash
$ git commit -a -m "my work on lab3_3 is done."
```

#### **参考答案**

请参考本文档的[纸质书](https://www.ryjiaoyu.com/book/details/50200)。

`<a name="lab3_challenge1_wait"></a>`

## 5.5 lab3_challenge1 进程等待和数据段复制（难度：&#9733;&#9733;&#9734;&#9734;&#9734;，[头歌实验链接](https://www.educoder.net/shixuns/hwtam4is/challenges)）

`<a name="lab3_challenge1_app"></a>`

#### **给定应用**

- user/app_wait.c

```c
  1 /*                                                                           
  2  * This app fork a child process, and the child process fork a grandchild process.
  3  * every process waits for its own child exit then prints.                   
  4  * Three processes also write their own global variables "flag"
  5  * to different values.
  6  */
  7 
  8 #include "user/user_lib.h"
  9 #include "util/types.h"
 10 
 11 int flag;
 12 int main(void) {
 13     flag = 0;
 14     int pid = fork();
 15     if (pid == 0) {
 16         flag = 1;
 17         pid = fork();
 18         if (pid == 0) {
 19             flag = 2;
 20             printu("Grandchild process end, flag = %d.\n", flag);
 21         } else {
 22             wait(pid);
 23             printu("Child process end, flag = %d.\n", flag);
 24         }
 25     } else {
 26         wait(-1);
 27         printu("Parent process end, flag = %d.\n", flag);
 28     }
 29     exit(0);
 30     return 0;
 31 }
```

wait系统调用是进程管理中一个非常重要的系统调用，它主要有两大功能：

* 当一个进程退出之后，它所占用的资源并不一定能够立即回收，比如该进程的内核栈目前就正用来进行系统调用处理。对于这种问题，一种典型的做法是当进程退出的时候内核立即回收一部分资源并将该进程标记为僵尸进程。由父进程调用wait函数的时候再回收该进程的其他资源。
* 父进程的有些操作需要子进程运行结束后获得结果才能继续执行，这时wait函数起到进程同步的作用。

在以上程序中，父进程把flag变量赋值为0，然后fork生成一个子进程，接着通过wait函数等待子进程的退出。子进程把自己的变量flag赋值为1，然后fork生成孙子进程，接着通过wait函数等待孙子进程的退出。孙子进程给自己的变量flag赋值为2并在退出时输出信息，然后子进程退出时输出信息，最后父进程退出时输出信息。由于fork之后父子进程的数据段相互独立（同一虚拟地址对应不同的物理地址），子进程对全局变量的赋值不影响父进程全局变量的值，因此结果如下：

```bash
In m_start, hartid:0
HTIF is available!
(Emulated) memory size: 2048 MB
Enter supervisor mode...
PKE kernel start 0x0000000080000000, PKE kernel end: 0x0000000080009000, PKE kernel size: 0x0000000000009000 .
free physical memory address: [0x0000000080009000, 0x0000000087ffffff] 
kernel memory manager is initializing ...
KERN_BASE 0x0000000080000000
physical address of _etext is: 0x0000000080005000
kernel page table is on 
Switch to user mode...
in alloc_proc. user frame 0x0000000087fbc000, user stack 0x000000007ffff000, user kstack 0x0000000087fbb000 
User application is loading.
Application: obj/app_wait
CODE_SEGMENT added at mapped info offset:3
DATA_SEGMENT added at mapped info offset:4
Application program entry point (virtual address): 0x00000000000100b0
going to insert process 0 to ready queue.
going to schedule process 0 to run.
User call fork.
will fork a child from parent 0.
in alloc_proc. user frame 0x0000000087fae000, user stack 0x000000007ffff000, user kstack 0x0000000087fad000 
do_fork map code segment at pa:0000000087fb2000 of parent to child at va:0000000000010000.
going to insert process 1 to ready queue.
going to schedule process 1 to run.
User call fork.
will fork a child from parent 1.
in alloc_proc. user frame 0x0000000087fa1000, user stack 0x000000007ffff000, user kstack 0x0000000087fa0000 
do_fork map code segment at pa:0000000087fb2000 of parent to child at va:0000000000010000.
going to insert process 2 to ready queue.
going to schedule process 2 to run.
Grandchild process end, flag = 2.
User exit with code:0.
going to insert process 1 to ready queue.
going to schedule process 1 to run.
Child process end, flag = 1.
User exit with code:0.
going to insert process 0 to ready queue.
going to schedule process 0 to run.
Parent process end, flag = 0.
User exit with code:0.
no more ready processes, system shutdown now.
System is shutting down with exit code 0.
```

`<a name="lab3_challenge1_content"></a>`

#### 实验内容

本实验为挑战实验，基础代码将继承和使用lab3_3完成后的代码：

- （先提交lab3_3的答案，然后）切换到lab3_challenge1_wait、继承lab3_3中所做修改：

```bash
//切换到lab3_challenge1_wait
$ git checkout lab3_challenge1_wait

//继承lab3_3以及之前的答案
$ git merge lab3_3_rrsched -m "continue to work on lab3_challenge1"
```

注意：**不同于基础实验，挑战实验的基础代码具有更大的不完整性，可能无法直接通过构造过程。**
同样，不同于基础实验，我们在代码中也并未专门地哪些地方的代码需要填写，哪些地方的代码无须填写。这样，我们留给读者更大的“想象空间”。

- 本实验的具体要求为：
  - 通过修改PKE内核和系统调用，为用户程序提供wait函数的功能，wait函数接受一个参数pid：
    - 当pid为-1时，父进程等待任意一个子进程退出即返回子进程的pid；
    - 当pid大于0时，父进程等待进程号为pid的子进程退出即返回子进程的pid；
    - 如果pid不合法或pid大于0且pid对应的进程不是当前进程的子进程，返回-1。
  - 补充do_fork函数，实验3_1实现了代码段的复制，你需要继续实现数据段的复制并保证fork后父子进程的数据段相互独立。
- 注意：最终测试程序可能和给出的用户程序不同，但都只涉及wait函数、fork函数和全局变量读写的相关操作。

`<a name="lab3_challenge1_guide"></a>`

#### 实验指导

* 你对内核代码的修改可能包含添加系统调用、在内核中实现wait函数的功能以及对do_fork函数的完善。

**注意：完成实验内容后，请读者另外编写应用，对自己的实现进行检测。**

**另外，后续的基础实验代码并不依赖挑战实验，所以读者可自行决定是否将自己的工作提交到本地代码仓库中（当然，提交到本地仓库是个好习惯，至少能保存自己的“作品”）。**

#### **更多指导**

请参考本文档的[纸质书](https://www.ryjiaoyu.com/book/details/50200)。

`<a name="lab3_challenge2_semaphore"></a>`

## 5.6 lab3_challenge2 实现信号量（难度：&#9733;&#9733;&#9733;&#9734;&#9734;，[头歌实验链接](https://www.educoder.net/shixuns/ehb5zf9f/challenges)）

`<a name="lab3_challenge2_app"></a>`

#### **给定应用**

- user/app_semaphore.c

```c
  1 /*                                                                                                         2 * This app create two child process.
  3 * Use semaphores to control the order of
  4 * the main process and two child processes print info. 
  5 */
  6 #include "user/user_lib.h"
  7 #include "util/types.h"
  8 
  9 int main(void) {
 10     int main_sem, child_sem[2];
 11     main_sem = sem_new(1);
 12     for (int i = 0; i < 2; i++) child_sem[i] = sem_new(0);
 13     int pid = fork();
 14     if (pid == 0) {
 15         pid = fork();
 16         for (int i = 0; i < 10; i++) {
 17             sem_P(child_sem[pid == 0]);
 18             printu("Child%d print %d\n", pid == 0, i);
 19             if (pid != 0) sem_V(child_sem[1]); else sem_V(main_sem);
 20         }
 21     } else {
 22         for (int i = 0; i < 10; i++) {
 23             sem_P(main_sem);
 24             printu("Parent print %d\n", i);
 25             sem_V(child_sem[0]);
 26         }
 27     }
 28     exit(0);
 29     return 0;
 30 }
```

以上程序通过信号量的增减，控制主进程和两个子进程的输出按主进程，第一个子进程，第二个子进程，主进程，第一个子进程，第二个子进程……这样的顺序轮流输出，如上面的应用预期输出如下：

```bash
In m_start, hartid:0
HTIF is available!
(Emulated) memory size: 2048 MB
Enter supervisor mode...
PKE kernel start 0x0000000080000000, PKE kernel end: 0x0000000080009000, PKE kernel size: 0x0000000000009000 .
free physical memory address: [0x0000000080009000, 0x0000000087ffffff] 
kernel memory manager is initializing ...
KERN_BASE 0x0000000080000000
physical address of _etext is: 0x0000000080005000
kernel page table is on 
Switch to user mode...
in alloc_proc. user frame 0x0000000087fbc000, user stack 0x000000007ffff000, user kstack 0x0000000087fbb000 
User application is loading.
Application: obj/app_semaphore
CODE_SEGMENT added at mapped info offset:3
DATA_SEGMENT added at mapped info offset:4
Application program entry point (virtual address): 0x00000000000100b0
going to insert process 0 to ready queue.
going to schedule process 0 to run.
User call fork.
will fork a child from parent 0.
in alloc_proc. user frame 0x0000000087fae000, user stack 0x000000007ffff000, user kstack 0x0000000087fad000 
do_fork map code segment at pa:0000000087fb2000 of parent to child at va:0000000000010000.
going to insert process 1 to ready queue.
Parent print 0
going to schedule process 1 to run.
User call fork.
will fork a child from parent 1.
in alloc_proc. user frame 0x0000000087fa2000, user stack 0x000000007ffff000, user kstack 0x0000000087fa1000 
do_fork map code segment at pa:0000000087fb2000 of parent to child at va:0000000000010000.
going to insert process 2 to ready queue.
Child0 print 0
going to schedule process 2 to run.
Child1 print 0
going to insert process 0 to ready queue.
going to schedule process 0 to run.
Parent print 1
going to insert process 1 to ready queue.
going to schedule process 1 to run.
Child0 print 1
going to insert process 2 to ready queue.
going to schedule process 2 to run.
Child1 print 1
going to insert process 0 to ready queue.
going to schedule process 0 to run.
Parent print 2
going to insert process 1 to ready queue.
going to schedule process 1 to run.
Child0 print 2
going to insert process 2 to ready queue.
going to schedule process 2 to run.
Child1 print 2
going to insert process 0 to ready queue.
going to schedule process 0 to run.
Parent print 3
going to insert process 1 to ready queue.
going to schedule process 1 to run.
Child0 print 3
going to insert process 2 to ready queue.
going to schedule process 2 to run.
Child1 print 3
going to insert process 0 to ready queue.
going to schedule process 0 to run.
Parent print 4
going to insert process 1 to ready queue.
going to schedule process 1 to run.
Child0 print 4
going to insert process 2 to ready queue.
going to schedule process 2 to run.
Child1 print 4
going to insert process 0 to ready queue.
going to schedule process 0 to run.
Parent print 5
going to insert process 1 to ready queue.
going to schedule process 1 to run.
Child0 print 5
going to insert process 2 to ready queue.
going to schedule process 2 to run.
Child1 print 5
going to insert process 0 to ready queue.
going to schedule process 0 to run.
Parent print 6
going to insert process 1 to ready queue.
going to schedule process 1 to run.
Child0 print 6
going to insert process 2 to ready queue.
going to schedule process 2 to run.
Child1 print 6
going to insert process 0 to ready queue.
going to schedule process 0 to run.
Parent print 7
going to insert process 1 to ready queue.
going to schedule process 1 to run.
Child0 print 7
going to insert process 2 to ready queue.
going to schedule process 2 to run.
Child1 print 7
going to insert process 0 to ready queue.
going to schedule process 0 to run.
Parent print 8
going to insert process 1 to ready queue.
going to schedule process 1 to run.
Child0 print 8
going to insert process 2 to ready queue.
going to schedule process 2 to run.
Child1 print 8
going to insert process 0 to ready queue.
going to schedule process 0 to run.
Parent print 9
going to insert process 1 to ready queue.
User exit with code:0.
going to schedule process 1 to run.
Child0 print 9
going to insert process 2 to ready queue.
User exit with code:0.
going to schedule process 2 to run.
Child1 print 9
User exit with code:0.
no more ready processes, system shutdown now.
System is shutting down with exit code 0.
```

`<a name="lab3_challenge2_content"></a>`

#### 实验内容

本实验为挑战实验，基础代码将继承和使用lab3_3完成后的代码：

- （先提交lab3_3的答案，然后）切换到lab3_challenge2_semaphore、继承lab3_3中所做修改：

```bash
//切换到lab3_challenge2_semaphore
$ git checkout lab3_challenge2_semaphore

//继承lab3_3以及之前的答案
$ git merge lab3_3_rrsched -m "continue to work on lab3_challenge1"
```

**注意：不同于基础实验，挑战实验的基础代码具有更大的不完整性，可能无法直接通过构造过程。**
同样，不同于基础实验，我们在代码中也并未专门地哪些地方的代码需要填写，哪些地方的代码无须填写。这样，我们留给读者更大的“想象空间”。

- 本实验的具体要求为：通过修改PKE内核和系统调用，为用户程序提供信号量功能。
- 注意：最终测试程序**可能和给出的用户程序不同**，但都只涉及信号量的相关操作。
- 提示：信号灯的结构**不能开太大**，否则会导致kernel_size出现问题

`<a name="lab3_challenge2_guide"></a>`

#### 实验指导

* 你对内核代码的修改可能包含以下内容：
  * 添加系统调用，使得用户对信号量的操作可以在内核态处理
  * 在内核中实现信号量的分配、释放和PV操作，当P操作处于等待状态时能够触发进程调度

**注意：完成实验内容后，请读者另外编写应用，对自己的实现进行检测。**

**另外，后续的基础实验代码并不依赖挑战实验，所以读者可自行决定是否将自己的工作提交到本地代码仓库中（当然，提交到本地仓库是个好习惯，至少能保存自己的“作品”）。**

#### **更多指导**

请参考本文档的[纸质书](https://www.ryjiaoyu.com/book/details/50200)。

`<a name="lab3_challenge3_cow"></a>`

## 5.7 lab3_challenge3 写时复制（Copy On Write）（难度：&#9733;&#9733;&#9733;&#9734;&#9734;，[头歌实验链接](https://www.educoder.net/shixuns/lfhrecys/challenges)）

`<a name="lab3_challenge3_app"></a>`

#### 给定应用

- user/app_cow.c

```c
/*
 * This app fork a child process to read and write the heap data from parent process.
 * Because implemented copy on write, when child process only read the heap data,
 * the physical address is the same as the parent process.
 * But after writing, child process heap will have different physical address.            
 */

#include "user/user_lib.h"
#include "util/types.h"

int main(void) {
  int *heap_data = naive_malloc();
  printu("the physical address of parent process heap is: ");
  printpa(heap_data);
  int pid = fork();
  if (pid == 0) {
    printu("the physical address of child process heap before copy on write is: ");
    printpa(heap_data);
    heap_data[0] = 0;
    printu("the physical address of child process heap after copy on write is: ");
    printpa(heap_data);
  }
  exit(0);
  return 0;
}
```

该应用执行如下操作：

1. 在父进程的堆上申请一片区域，并输出其物理地址。
2. 进行fork操作，输出子进程在对堆数据写入前后对应的物理地址。

`<a name="lab3_challenge3_content"></a>`

#### 实验内容

本实验为挑战实验，基础代码将继承和使用lab3_3完成后的代码：

- 先提交lab3_3的答案，然后）切换到lab3_challenge3_cow、继承lab3_3（注意，不是继承lab3_challenge1_wait和lab3_challenge2_semaphore！PKE的挑战实验之间无继承关联）中所做修改，并make后的直接运行结果：

```bash
//切换到lab3_challenge3_cow
$ git checkout lab3_challenge3_cow

//继承lab3_3以及之前的答案
$ git merge lab3_3_rrsched -m "continue to work on lab3_challenge3"

$ spike obj/riscv-pke obj/app_cow
In m_start, hartid:0
HTIF is available!
(Emulated) memory size: 2048 MB
Enter supervisor mode...
PKE kernel start 0x0000000080000000, PKE kernel end: 0x000000008000b000, PKE kernel size: 0x000000000000b000 .
free physical memory address: [0x000000008000b000, 0x0000000087ffffff] 
kernel memory manager is initializing ...
KERN_BASE 0x0000000080000000
physical address of _etext is: 0x0000000080005000
kernel page table is on 
Switch to user mode...
in alloc_proc. user frame 0x0000000087fbc000, user stack 0x000000007ffff000, user kstack 0x0000000087fbb000 
User application is loading.
Application: obj/app_two_long_loops
CODE_SEGMENT added at mapped info offset:4
Application program entry point (virtual address): 0x0000000000010078
going to insert process 0 to ready queue.
going to schedule process 0 to run.
the physical address of parent process heap is: 0000000087faf000
User call fork.
will fork a child from parent 0.
in alloc_proc. user frame 0x0000000087fad000, user stack 0x000000007ffff000, user kstack 0x0000000087fac000 
do_fork map code segment at pa:0000000087fb2000 of parent to child at va:0000000000010000.
going to insert process 1 to ready queue.
User exit with code:0.
going to schedule process 1 to run.
the physical address of child process heap before copy on write is: 0000000087fa3000
the physical address of child process heap after copy on write is: 0000000087fa3000
User exit with code:0.
no more ready processes, system shutdown now.
System is shutting down with exit code 0.
```

可以看到，在fork之后，子进程的堆数据无论是否进行写入，其物理地址与父进程均不同。这是因为我们fork时直接为子进程分配了新的物理页面导致的。在实现了写时复制的fork中，是不会直接给子进程申请新的物理页面，而是当子进程执行写入操作时，才通过异常机制来分配新的物理页面，也就是说，子进程的堆数据物理地址在未执行写入前，和父进程应该是一样的。

注意：**不同于基础实验，挑战实验的基础代码具有更大的不完整性，可能无法直接通过构造过程。** 同样，不同于基础实验，我们在代码中也并未专门地哪些地方的代码需要填写，哪些地方的代码无须填写。这样，我们留给读者更大的“想象空间”。

- 本实验要求你通过修改内核代码，使得在fork时不直接为新的堆空间分配内存，而是等写入之后才分配，即实现fork的写时复制(COW)机制。
- 你的程序输出应该如下：

```
$ spike obj/riscv-pke obj/app_cow
In m_start, hartid:0
HTIF is available!
(Emulated) memory size: 2048 MB
Enter supervisor mode...
PKE kernel start 0x0000000080000000, PKE kernel end: 0x000000008004b000, PKE kernel size: 0x000000000004b000 .
free physical memory address: [0x000000008004b000, 0x0000000087ffffff] 
kernel memory manager is initializing ...
KERN_BASE 0x0000000080000000
physical address of _etext is: 0x0000000080005000
kernel page table is on 
Switch to user mode...
in alloc_proc. user frame 0x0000000087fbc000, user stack 0x000000007ffff000, user kstack 0x0000000087fbb000 
User application is loading.
Application: obj/app_cow
CODE_SEGMENT added at mapped info offset:4
Application program entry point (virtual address): 0x0000000000010078
going to insert process 0 to ready queue.
going to schedule process 0 to run.
the physical address of parent process heap is: 0000000087faf000
User call fork.
will fork a child from parent 0.
in alloc_proc. user frame 0x0000000087fad000, user stack 0x000000007ffff000, user kstack 0x0000000087fac000 
do_fork map code segment at pa:0000000087fb2000 of parent to child at va:0000000000010000.
going to insert process 1 to ready queue.
User exit with code:0.
going to schedule process 1 to run.
the physical address of child process heap before copy on write is: 0000000087faf000
handle_page_fault: 0000000000400000
the physical address of child process heap after copy on write is: 0000000087fa0000
User exit with code:0.
no more ready processes, system shutdown now.
System is shutting down with exit code 0.
```

可以看到，输出里多出了处理异常和cow验证输出的语句。

`<a name="lab3_challenge3_guide"></a>`

#### 实验指导

写时复制是一种非常常见的优化手段。在Shell中，我们经常fork完一个进程之后会立即调用exec来将子进程替换为新的进程。如果我们在fork时将父进程的所有地址空间全部复制给子进程，而子进程在exec后做的第一件事情是丢弃这个地址空间，也就是我们做了一些无用的复制操作，产生了额外的开销。所以我们可以用“先映射但不实际复制”的方式来优化上述提到的开销，只有子进程真正需要写入映射到父进程的地址空间时，我们才真正的执行“申请空间-复制”这一操作，这就是写时复制（Copy on Write，COW）技术。

- 为完成该挑战，你需要对pke中进程空间有更详细的了解，判断哪些地址范围可能会被访问。
- 你可以参考Linux或其他主流操作系统中写时复制机制的实现。
- 你对内核代码的修改可能包含以下内容：
  - 在堆段的复制发生时，实现”先映射但不实际复制“。
  - 对页表项的标志位进行修改。当子进程拥有映射到父进程的地址的页表项时，页表项应该具有哪些权限？当COW发生后，页表项又该具有哪些权限？
  - 内核如何分辨现在是一个copy-on-write fork的场景？**提示：PTE的标志位中有两位RSW是没有使用的。**
  - 如果有父进程fork了多个子进程，意味着该父进程的物理页会被多个子进程”共享“。举个例子，当父进程退出时我们需要更加的小心，因为我们要判断是否能立即释放相应的物理页。如果有子进程还在使用这些物理页，而内核又释放了这些物理页，将会出现问题。该“共享”页在什么条件下才能释放？

**注意：本挑战的难点在于对页表项的控制以及对页面释放的判断，读者应思考两个问题如何解决，完成设计并验证后，读者可以继续尝试在数据段也加入写时复制机制。**

**另外，后续的基础实验代码并不依赖挑战实验，所以读者可自行决定是否将自己的工作提交到本地代码仓库中（当然，提交到本地仓库是个好习惯，至少能保存自己的“作品”）。**

#### **更多指导**

请参考本文档的[纸质书](https://www.ryjiaoyu.com/book/details/50200)。
