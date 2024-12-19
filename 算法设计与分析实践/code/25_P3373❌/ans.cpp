#include <iostream>
#include <vector>
using namespace std;

class SegmentTree {
public:
    SegmentTree(int n, int mod) : n(n), mod(mod) {
        tree.resize(4 * n, 0);
        lazy_add.resize(4 * n, 0);
        lazy_mul.resize(4 * n, 1);
    }

    // 构建线段树
    void build(const vector<int>& arr, int v, int tl, int tr) {
        if (tl == tr) {
            tree[v] = arr[tl] % mod;
        } else {
            int tm = (tl + tr) / 2;
            build(arr, 2 * v, tl, tm);
            build(arr, 2 * v + 1, tm + 1, tr);
            tree[v] = (tree[2 * v] + tree[2 * v + 1]) % mod;
        }
    }

    // 应用懒标记
    void push(int v, int tl, int tr) {
        if (lazy_mul[v] != 1 || lazy_add[v] != 0) {
            // 处理乘法
            tree[v] = (tree[v] * lazy_mul[v]) % mod;
            // 处理加法
            tree[v] = (tree[v] + (tr - tl + 1) * lazy_add[v]) % mod;
            
            if (tl != tr) {
                // 更新左右子节点的懒标记
                lazy_mul[2 * v] = (lazy_mul[2 * v] * lazy_mul[v]) % mod;
                lazy_add[2 * v] = (lazy_add[2 * v] * lazy_mul[v] + lazy_add[v]) % mod;
                
                lazy_mul[2 * v + 1] = (lazy_mul[2 * v + 1] * lazy_mul[v]) % mod;
                lazy_add[2 * v + 1] = (lazy_add[2 * v + 1] * lazy_mul[v] + lazy_add[v]) % mod;
            }
            
            // 清除当前节点的懒标记
            lazy_mul[v] = 1;
            lazy_add[v] = 0;
        }
    }

    // 区间更新（加法）
    void update_add(int v, int tl, int tr, int l, int r, int add_val) {
        push(v, tl, tr);
        if (l > r) return;
        if (l == tl && r == tr) {
            lazy_add[v] = (lazy_add[v] + add_val) % mod;
            push(v, tl, tr);
        } else {
            int tm = (tl + tr) / 2;
            update_add(2 * v, tl, tm, l, min(r, tm), add_val);
            update_add(2 * v + 1, tm + 1, tr, max(l, tm + 1), r, add_val);
            tree[v] = (tree[2 * v] + tree[2 * v + 1]) % mod;
        }
    }

    // 区间更新（乘法）
    void update_mul(int v, int tl, int tr, int l, int r, int mul_val) {
        push(v, tl, tr);
        if (l > r) return;
        if (l == tl && r == tr) {
            lazy_mul[v] = (lazy_mul[v] * mul_val) % mod;
            push(v, tl, tr);
        } else {
            int tm = (tl + tr) / 2;
            update_mul(2 * v, tl, tm, l, min(r, tm), mul_val);
            update_mul(2 * v + 1, tm + 1, tr, max(l, tm + 1), r, mul_val);
            tree[v] = (tree[2 * v] + tree[2 * v + 1]) % mod;
        }
    }

    // 区间查询
    int query(int v, int tl, int tr, int l, int r) {
        push(v, tl, tr);
        if (l > r) return 0;
        if (l == tl && r == tr) {
            return tree[v];
        } else {
            int tm = (tl + tr) / 2;
            return (query(2 * v, tl, tm, l, min(r, tm)) + query(2 * v + 1, tm + 1, tr, max(l, tm + 1), r)) % mod;
        }
    }

private:
    int n, mod;
    vector<int> tree, lazy_add, lazy_mul;
};

int main() {
    int n, q, m;
    cin >> n >> q >> m;
    
    vector<int> arr(n);
    for (int i = 0; i < n; ++i) {
        cin >> arr[i];
    }

    SegmentTree seg_tree(n, m);
    seg_tree.build(arr, 1, 0, n - 1);

    for (int i = 0; i < q; ++i) {
        int op;
        cin >> op;
        if (op == 1) {
            int x, y, k;
            cin >> x >> y >> k;
            seg_tree.update_mul(1, 0, n - 1, x - 1, y - 1, k);
        } else if (op == 2) {
            int x, y, k;
            cin >> x >> y >> k;
            seg_tree.update_add(1, 0, n - 1, x - 1, y - 1, k);
        } else if (op == 3) {
            int x, y;
            cin >> x >> y;
            cout << seg_tree.query(1, 0, n - 1, x - 1, y - 1) << endl;
        }
    }

    return 0;
}













// #include <iostream>
// #include <vector>
// #include <algorithm>

// using namespace std;

// class SegmentTree 
// {
// private:
//     int n, mod;
//     vector<int> a;
//     vector<int> tree, lazy_add, lazy_mul;

// public:
//     SegmentTree(int size, int mod_value) : n(size), mod(mod_value), a(size + 1), tree(4 * size + 1), lazy_add(4 * size + 1, 0), lazy_mul(4 * size + 1, 1) 
//     {
//         for (int i = 1; i <= n; ++i) 
//         {
//             cin >> a[i];
//         }
//         build(1, 1, n);
//     }

//     void build(int id, int l, int r) 
//     {
//         if (l == r) 
//         {
//             tree[id] = a[l];
//             return;
//         }
//         int mid = (l + r) / 2;
//         build(id * 2, l, mid);
//         build(id * 2 + 1, mid + 1, r);
//         tree[id] = (tree[id * 2] + tree[id * 2 + 1]) % mod;
//     }

//     void push_down(int id, int l, int r)
//     {
//         if (lazy_mul[id] != 1 || lazy_add[id] != 0) 
//         {
//             int mid = (l + r) / 2;
//             // Apply the lazy multiplication
//             tree[id * 2] = (tree[id * 2] * lazy_mul[id]) % mod;
//             tree[id * 2 + 1] = (tree[id * 2 + 1] * lazy_mul[id]) % mod;

//             // Apply the lazy addition
//             tree[id * 2] = (tree[id * 2] + lazy_add[id] * (mid - l + 1)) % mod;
//             tree[id * 2 + 1] = (tree[id * 2 + 1] + lazy_add[id] * (r - mid)) % mod;

//             // Propagate the laziness to the children
//             if (l != r) 
//             {
//                 lazy_mul[id * 2] = (lazy_mul[id * 2] * lazy_mul[id]) % mod;
//                 lazy_mul[id * 2 + 1] = (lazy_mul[id * 2 + 1] * lazy_mul[id]) % mod;

//                 lazy_add[id * 2] = (lazy_add[id * 2] + lazy_add[id]) % mod;
//                 lazy_add[id * 2 + 1] = (lazy_add[id * 2 + 1] + lazy_add[id]) % mod;
//             }

//             // Reset the lazy values for the current node
//             lazy_add[id] = 0;
//             lazy_mul[id] = 1;
//         }
//     }

//     void update_add(int id, int l, int r, int x, int y, int k) 
//     {
//         push_down(id, l, r);
//         if (x <= l && r <= y) 
//         {
//             tree[id] = (tree[id] + k * (r - l + 1)) % mod;
//             if (l != r) 
//             {
//                 lazy_add[id] = (lazy_add[id] + k) % mod;
//             }
//             return;
//         }
//         int mid = (l + r) / 2;
//         if (x <= mid)
//             update_add(id * 2, l, mid, x, y, k);
//         if (y > mid)
//             update_add(id * 2 + 1, mid + 1, r, x, y, k);
//         tree[id] = (tree[id * 2] + tree[id * 2 + 1]) % mod;
//     }

//     void update_imul(int id, int l, int r, int x, int y, int k) 
//     {
//         push_down(id, l, r);
//         if (x <= l && r <= y) 
//         {
//             tree[id] = (tree[id] * k) % mod;
//             if (l != r) 
//             {
//                 lazy_mul[id] = (lazy_mul[id] * k) % mod;
//             }
//             return;
//         }
//         int mid = (l + r) / 2;
//         if (x <= mid)
//             update_imul(id * 2, l, mid, x, y, k);
//         if (y > mid)
//             update_imul(id * 2 + 1, mid + 1, r, x, y, k);
//         tree[id] = (tree[id * 2] + tree[id * 2 + 1]) % mod;
//     }

//     int query(int id, int l, int r, int x, int y) 
//     {
//         push_down(id, l, r);
//         if (x <= l && r <= y)
//             return tree[id];
//         int mid = (l + r) / 2;
//         int res = 0;
//         if (x <= mid)
//             res = (res + query(id * 2, l, mid, x, y)) % mod;
//         if (y > mid)
//             res = (res + query(id * 2 + 1, mid + 1, r, x, y)) % mod;
//         return res;
//     }
// };

// int main()
// {
//     int n, m, mod;
//     cin >> n >> m >> mod;
//     SegmentTree segmentTree(n, mod);
//     for (int i = 0; i < m; i++) 
//     {
//         int op, x, y, k;
//         cin >> op;
//         if (op == 1) 
//         {
//             cin >> x >> y >> k;
//             segmentTree.update_imul(1, 1, n, x, y, k);
//         } 
//         else if (op == 2) 
//         {
//             cin >> x >> y >> k;
//             segmentTree.update_add(1, 1, n, x, y, k);
//         } 
//         else 
//         {
//             cin >> x >> y;
//             cout << segmentTree.query(1, 1, n, x, y) << endl;
//         }
//     }
//     return 0;
// }












// #include <iostream>
// #include <vector>
// #include <algorithm>

// using namespace std;

// class SegmentTree 
// {
// private:
//     int n; // 元素个数
//     int mod; // 模数
//     vector<int> a;
//     vector<int> tree;

// public:
//     SegmentTree(int size,int mod) : n(size), mod(mod), a(size + 1), tree(4 * size + 1)
//     {
//         for (int i = 1; i <= n; ++i) 
//         {
//             cin >> a[i];
//         }
//         build(1, 1, n);
//     }

//     void build(int id, int l, int r) 
//     {
//         if (l == r) 
//         {
//             tree[id] = a[l];
//             return;
//         }
//         int mid = (l + r) / 2;
//         build(id * 2, l, mid);
//         build(id * 2 + 1, mid + 1, r);
//         tree[id] = (tree[id * 2] + tree[id * 2 + 1]) % mod;
//     }

//     int query(int id, int l, int r, int x, int y) 
//     {
//         if (x <= l && r <= y)
//             return tree[id];
//         int mid = (l + r) / 2;
//         int res = 0;
//         if (x <= mid)
//             res = (res + query(id * 2, l, mid, x, y)) % mod;
//         if (y > mid)
//             res = (res + query(id * 2 + 1, mid + 1, r, x, y)) % mod;
//         return res;
//     }

//     void update_add(int id, int l, int r, int x, int y, int k) 
//     {
//         if (x <= l && r <= y) 
//         {
//             tree[id] = (tree[id] + k * (r - l + 1)) % mod;
//             return;
//         }
//         int mid = (l + r) / 2;
//         if (x <= mid)
//             update_add(id * 2, l, mid, x, y, k);
//         if (y > mid)
//             update_add(id * 2 + 1, mid + 1, r, x, y, k);
//         tree[id] = (tree[id * 2] + tree[id * 2 + 1]) % mod;
//     }

//     void update_imul(int id, int l, int r, int x, int y, int k) 
//     {
//         if (x <= l && r <= y) 
//         {
//             tree[id] = (tree[id] * k) % mod;
//             return;
//         }
//         int mid = (l + r) / 2;
//         if (x <= mid)
//             update_imul(id * 2, l, mid, x, y, k);
//         if (y > mid)
//             update_imul(id * 2 + 1, mid + 1, r, x, y, k);
//         tree[id] = (tree[id * 2] + tree[id * 2 + 1]) % mod;
//     }
// };

// int main()
// {
//     int n,m,mod;
//     cin>>n>>m>>mod;
//     SegmentTree segmentTree(n,mod);
//     for(int i=0;i<m;i++)
//     {
//         int op, x, y, k;
//         cin >> op;
//         if(op==1)
//         {
//             cin >> x >> y >> k;
//             segmentTree.update_imul(1, 1, n, x, y, k);
//         }
//         else if(op==2)
//         {
//             cin >> x >> y >> k;
//             segmentTree.update_add(1, 1, n, x, y, k);
//         }
//         else
//         {
//             cin >> x >> y;
//             cout << segmentTree.query(1, 1, n, x, y) <<endl;
//         }
//     }
//     return 0;
// }
