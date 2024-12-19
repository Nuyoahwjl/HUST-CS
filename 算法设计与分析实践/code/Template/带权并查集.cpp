#include <iostream>
#include <vector>
#include <cstdio>

using namespace std;

#define MAX_N 30005

// parent[i] 表示 i 的父节点
// size[i] 表示 i 所在集合的大小
// distance[i] 表示 i 到其父节点的距离
int parent[MAX_N], size[MAX_N], dist[MAX_N];

// 查找操作，带路径压缩
int find(int x) {
    if (parent[x] != x) {
        int original_parent = parent[x];
        parent[x] = find(parent[x]);
        dist[x] += dist[original_parent];  // 更新距离
    }
    return parent[x];
}

// 合并操作
void union_sets(int x, int y) {
    int rootX = find(x);
    int rootY = find(y);

    if (rootX != rootY) {
        // 按秩合并，保持树的平衡
        parent[rootX] = rootY;
        dist[rootX] = size[rootY];  // 更新 rootX 到 rootY 的距离
        size[rootY] += size[rootX];     // 更新 rootY 的集合大小
        size[rootX] = 0;               
    }
}

int main() {
    int T;
    cin >> T;

    // 初始化并查集
    for (int i = 1; i <= MAX_N; ++i) {
        parent[i] = i;
        size[i] = 1;
        dist[i] = 0;
    }

    // 处理每一条指令
    while (T--) {
        char op;
        int i, j;
        cin >> op >> i >> j;

        if (op == 'M') {
            // 合并指令 M i j
            union_sets(i, j);
        } else if (op == 'C') {
            // 查询指令 C i j
            if (find(i) == find(j)) {
                // 如果在同一列，输出战舰之间的数目
                cout << abs(dist[i] - dist[j]) - 1 << endl;
            } else {
                // 如果不在同一列，输出 -1
                cout << -1 << endl;
            }
        }
    }

    return 0;
}
