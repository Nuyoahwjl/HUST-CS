#include <iostream>
#include <vector>
using namespace std;

// 树状数组类
class FenwickTree {
private:
    int n;                 // 元素个数
    vector<long long> BIT; // 树状数组

    // 获取最低位 1
    int lowbit(int x) {
        return x & -x;
    }

public:
    // 初始化
    FenwickTree(int size) : n(size), BIT(size + 1, 0) {}

    // 单点更新，将 idx 位置的值增加 delta
    void update(int idx, long long delta) {
        while (idx <= n) {
            BIT[idx] += delta;
            idx += lowbit(idx);
        }
    }

    // 查询前缀和 [1, idx]
    long long query(int idx) {
        long long sum = 0;
        while (idx > 0) {
            sum += BIT[idx];
            idx -= lowbit(idx);
        }
        return sum;
    }

    // 查询区间和 [l, r]
    long long rangeQuery(int l, int r) {
        return query(r) - query(l - 1);
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    cin >> n >> m;

    vector<int> arr(n + 1); // 输入数组（从 1 开始）
    FenwickTree fenwick(n);

    // 读取初始数组并构建树状数组
    for (int i = 1; i <= n; ++i) {
        cin >> arr[i];
        fenwick.update(i, arr[i]); // 初始化树状数组
    }

    // 处理操作
    while (m--) {
        int op, x, y;
        cin >> op >> x >> y;
        if (op == 1) {
            // 单点更新：将第 x 个数加上 y
            fenwick.update(x, y);
        } else if (op == 2) {
            // 区间查询：查询 [x, y] 的和
            cout << fenwick.rangeQuery(x, y) << "\n";
        }
    }

    return 0;
}