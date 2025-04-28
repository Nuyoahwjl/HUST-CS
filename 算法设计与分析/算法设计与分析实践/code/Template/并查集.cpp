#include <stdio.h>

#define MAX_N 10001

int parent[MAX_N];  // 存储每个元素的父节点
int rank[MAX_N];    // 存储每个集合的秩（树的高度）

// 查找 x 的根节点，并进行路径压缩
int find(int x) {
    if (parent[x] != x) {
        parent[x] = find(parent[x]);
    }
    return parent[x];
}

// 将两个元素 x 和 y 所在的集合合并
void union_sets(int x, int y) {
    int rootX = find(x);
    int rootY = find(y);

    if (rootX != rootY) {
        // 按秩合并：将秩小的树挂在秩大的树下
        if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        } else if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        } else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }
    }
}

int main() {
    int N, M;
    scanf("%d %d", &N, &M);

    // 初始化，每个元素的父节点指向自身，秩为 0
    for (int i = 1; i <= N; i++) {
        parent[i] = i;
        rank[i] = 0;
    }

    for (int i = 0; i < M; i++) {
        int Zi, Xi, Yi;
        scanf("%d %d %d", &Zi, &Xi, &Yi);

        if (Zi == 1) {
            union_sets(Xi, Yi);
        } else if (Zi == 2) {
            if (find(Xi) == find(Yi)) {
                printf("Y\n");
            } else {
                printf("N\n");
            }
        }
    }

    return 0;
}