#include <stdio.h>
#include <stdlib.h>

#define MAX_N 1000

// 比较函数，用于qsort排序
int cmp(const void *a, const void *b) {
    return *(int *)a - *(int *)b;
}

int main() {
    int n;
    scanf("%d", &n);
    int arr[MAX_N], sorted[MAX_N];
    int visited[MAX_N] = {0}; // 标记是否已经访问过
    int i, global_min = 10001;

    for (i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
        sorted[i] = arr[i];
        if (arr[i] < global_min) {
            global_min = arr[i];
        }
    }

    // 对原数组排序
    qsort(sorted, n, sizeof(int), cmp);

    int total_cost = 0;

    for (i = 0; i < n; i++) {
        if (visited[i] || arr[i] == sorted[i]) {
            // 已经处理过或者位置正确，跳过
            continue;
        }

        int cycle_sum = 0;       // 当前环的总和
        int cycle_size = 0;      // 当前环的大小
        int min_in_cycle = 10001; // 当前环的最小值
        int x = i;

        // 处理一个置换环
        while (!visited[x]) {
            visited[x] = 1;
            int val = arr[x];
            cycle_sum += val;
            if (val < min_in_cycle) {
                min_in_cycle = val;
            }
            cycle_size++;
            // 找到 val 在目标数组中的索引
            for (int j = 0; j < n; j++) {
                if (sorted[j] == val) {
                    x = j;
                    break;
                }
            }
        }

        // 计算两种方式的代价
        int cost_direct = cycle_sum + (cycle_size - 2) * min_in_cycle;
        int cost_with_global_min = cycle_sum + min_in_cycle + (cycle_size + 1) * global_min;

        // 累加最小代价
        total_cost += cost_direct < cost_with_global_min ? cost_direct : cost_with_global_min;
    }

    printf("%d\n", total_cost);
    return 0;
}
