// #include <iostream>
// #include <vector>
// #include <string>
// #include <queue>

// using namespace std;

// int main()
// {
//     string s;
//     cin >> s;
//     int n = s.size();
//     priority_queue<string, vector<string>, greater<string>> minHeap;
//     for (int i=n-1; i>=0; i--)
//     {
//         string temp =  s.substr(i);
//         minHeap.push(temp);
//     }
//     while (minHeap.size() > 0)
//     {
//         string temp = minHeap.top();
//         minHeap.pop();
//         cout << n-temp.size()+1 << ' ';
//     }
//     return 0;
// }

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

// 后缀排序（基于倍增法 + 基数排序）
vector<int> buildSuffixArray(const string &s) {
    int n = s.size();
    vector<int> sa(n), rank(n), tmp(n), cnt(max(256, n), 0);

    // 初始化：基于第一关键字（即单个字符的ASCII值）排序
    for (int i = 0; i < n; ++i) {
        rank[i] = s[i]; // 初始rank直接用ASCII值
        cnt[rank[i]]++;
    }
    for (int i = 1; i < 256; ++i) cnt[i] += cnt[i - 1];
    for (int i = n - 1; i >= 0; --i) sa[--cnt[rank[i]]] = i;

    // 倍增排序
    for (int k = 1; k < n; k *= 2) {
        // 基于第二关键字排序，sa 按照 (rank[i], rank[i + k]) 排序
        int p = 0;
        for (int i = n - k; i < n; ++i) tmp[p++] = i; // 超出范围的后缀优先
        for (int i = 0; i < n; ++i) if (sa[i] >= k) tmp[p++] = sa[i] - k;

        // 基于第一关键字排序
        fill(cnt.begin(), cnt.begin() + max(256, n), 0);
        for (int i = 0; i < n; ++i) cnt[rank[tmp[i]]]++;
        for (int i = 1; i < max(256, n); ++i) cnt[i] += cnt[i - 1];
        for (int i = n - 1; i >= 0; --i) sa[--cnt[rank[tmp[i]]]] = tmp[i];

        // 更新rank
        tmp[sa[0]] = 0;
        p = 0;
        for (int i = 1; i < n; ++i) {
            if (rank[sa[i]] != rank[sa[i - 1]] || 
                (sa[i] + k < n ? rank[sa[i] + k] : -1) != (sa[i - 1] + k < n ? rank[sa[i - 1] + k] : -1)) {
                p++;
            }
            tmp[sa[i]] = p;
        }
        rank = tmp;

        if (p == n - 1) break; // 已经排好序
    }

    return sa;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    cin >> s;

    vector<int> suffixArray = buildSuffixArray(s);

    // 输出后缀数组的第一个字符在原字符串中的位置（1-based）
    for (size_t i = 0; i < suffixArray.size(); ++i) {
        cout << suffixArray[i] + 1 << " ";
    }

    return 0;
}
