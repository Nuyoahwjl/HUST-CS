#include <iostream>
#include <vector>
#include <set>
using namespace std;

const int MAXN = 200005;

int parent[MAXN], rank_[MAXN];
bool active[MAXN];          // 记录星球是否处于活动状态
int componentCount;         // 当前连通块的个数

// 并查集 - 查找根节点（带路径压缩）
int find(int x) {
    if (parent[x] != x) parent[x] = find(parent[x]);
    return parent[x];
}

// 并查集 - 合并两个集合
void unionSets(int x, int y) {
    int rootX = find(x);
    int rootY = find(y);
    if (rootX != rootY) {
        // 按秩合并
        if (rank_[rootX] > rank_[rootY]) {
            parent[rootY] = rootX;
        } else if (rank_[rootX] < rank_[rootY]) {
            parent[rootX] = rootY;
        } else {
            parent[rootY] = rootX;
            rank_[rootX]++;
        }
        componentCount--;  // 合并成功，连通块数减少
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    cin >> n >> m;

    vector<pair<int, int>> edges;  // 存储边
    for (int i = 0; i < m; i++) {
        int x, y;
        cin >> x >> y;
        edges.push_back({x, y});
    }

    int k;
    cin >> k;

    vector<int> attacks(k);  // 按顺序存储攻击目标
    set<int> attackedSet;    // 攻击星球的集合
    for (int i = 0; i < k; i++) {
        cin >> attacks[i];
        attackedSet.insert(attacks[i]);
    }

    // 初始化并查集
    for (int i = 0; i < n; i++) {
        parent[i] = i;
        rank_[i] = 0;
        active[i] = true;  // 初始时所有星球都处于活动状态
    }

    // 初始时，将攻击目标的星球标记为非活动
    for (int i : attacks) {
        active[i] = false;
    }

    componentCount = 0;
    // 构建初始连通图，只考虑非攻击目标的星球
    for (const auto &edge : edges) {
        int x = edge.first;
        int y = edge.second;
        if (active[x] && active[y]) {
            unionSets(x, y);
        }
    }

    // 统计初始连通块数
    vector<int> size(n, 0);  // 用于去重，避免重复计算连通块
    for (int i = 0; i < n; i++) {
        if (active[i]) {
            int root = find(i);
            if (!size[root]) {
                componentCount++;
                size[root] = 1;
            }
        }
    }

    vector<int> result;  // 存储每次攻击后的连通块数量
    result.push_back(componentCount);

    // 逆序恢复攻击目标的星球
    for (int i = k - 1; i >= 0; i--) {
        int curr = attacks[i];
        active[curr] = true;
        componentCount++;  // 恢复一个星球时，连通块数+1
        // 恢复一个星球时，连通块数+1
        for (const auto &edge : edges) {
            int x = edge.first;
            int y = edge.second;
            if ((x == curr && active[y]) || (y == curr && active[x])) {
            unionSets(x, y);
            }
        }
        // 遍历所有边，若边的另一端点存在，则重新连通
        for (const auto &edge : edges) {
            int x = edge.first;
            int y = edge.second;
            if ((x == curr && active[y]) || (y == curr && active[x])) {
            unionSets(x, y);
            }
        }
        result.push_back(componentCount);
    }

    // 输出结果
    for (int i = result.size() - 1; i >= 0; i--) {
        cout << result[i] << endl;
    }

    return 0;
}
