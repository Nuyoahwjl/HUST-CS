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
    // long long rangeQuery(int l, int r) {
    //     return query(r) - query(l - 1);
    // }
};

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    cin >> n >> m;

    FenwickTree fenwick(n);
    int temp1,temp2;

    cin >> temp1 ;
    fenwick.update(1, temp1); // 初始化树状数组(差分数组)
    for (int i = 2; i <= n; ++i) {
        cin >> temp2;
        fenwick.update(i, temp2-temp1); // 初始化树状数组
        temp1 = temp2;
    }

    for(int i=1;i<=m;++i)
    {
        int op, x, y ,k;
        cin >> op;
        if(op==1)
        {
            cin >> x >> y >> k;
            fenwick.update(x, k);
            fenwick.update(y+1, -k);
        }
        else
        {   
            cin >> x ;
            cout << fenwick.query(x) << endl;
        }
    }
    return 0;
}