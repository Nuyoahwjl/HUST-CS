        .text
        .align  1
        .globl  bubblesort
        .type   bubblesort, @function

bubblesort:
        mv      t0, a0          # t0 = arr 数组首地址
        mv      t1, a1          # t1 = n 数组长度

        li      t2, 0           # t2 = i = 0，外层循环变量

.L_outer:
        addi    t3, t1, -1      # t3 = n - 1
        bge     t2, t3, .L_done # if i >= n - 1，排序结束

        li      t4, 0           # t4 = j = 0，内层循环变量
        sub     t5, t3, t2      # t5 = n - 1 - i，内层循环上界

.L_inner:
        bge     t4, t5, .L_next_outer   # if j >= n - 1 - i，结束内层循环

        slli    t6, t4, 2       # t6 = j * 4，因为 int 占 4 字节
        add     t6, t0, t6      # t6 = &arr[j]

        lw      a2, 0(t6)       # a2 = arr[j]
        lw      a3, 4(t6)       # a3 = arr[j + 1]

        ble     a2, a3, .L_no_swap      # if arr[j] <= arr[j+1]，不交换

        sw      a3, 0(t6)       # arr[j] = arr[j + 1]
        sw      a2, 4(t6)       # arr[j + 1] = 原来的 arr[j]

.L_no_swap:
        addi    t4, t4, 1       # j++
        j       .L_inner

.L_next_outer:
        addi    t2, t2, 1       # i++
        j       .L_outer

.L_done:
        li      a0, 0           # 返回值必须为 0
        ret

        .size   bubblesort, .-bubblesort
