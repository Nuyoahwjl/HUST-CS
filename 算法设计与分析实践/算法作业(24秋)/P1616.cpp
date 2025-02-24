// 完全背包问题

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
    int t, m;
    cin >> t >> m; // 输入总时间t和物品数量m

    vector<long long> dp(t + 1, 0); // 初始化dp数组，长度为t+1，初始值为0

    for (int i = 0; i < m; ++i) {
        int a, b;
        cin >> a >> b; // 输入每个物品的时间a和价值b
        for (int j = a; j <= t; ++j) {
            dp[j] = max(dp[j], dp[j - a] + b); // 更新dp数组
        }
    }

    cout << dp[t] << endl; // 输出最大价值

    return 0;
}