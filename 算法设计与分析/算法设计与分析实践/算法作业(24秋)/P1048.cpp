#include <bits/stdc++.h>
using namespace std;

int main() {
    int t, m;
    // 输入总时间 t 和物品数量 m
    cin >> t >> m;
    vector<int> times(m + 1), value(m + 1);
    // 输入每个物品的时间和价值
    for (int i = 1; i <= m; i++) {
        cin >> times[i] >> value[i];
    }

    // 定义 dp 数组，dp[i][j] 表示前 i 个物品在时间 j 内的最大价值
    vector<vector<int>> dp(m + 1, vector<int>(t + 1, 0));

    // 动态规划求解
    for (int i = 1; i <= m; i++) {
        for (int j = 0; j <= t; j++) {
            if (j >= times[i]) {
                // 如果当前时间 j 大于等于第 i 个物品所需时间
                // 选择放入第 i 个物品或不放入，取最大值
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - times[i]] + value[i]);
            } else {
                // 当前时间 j 小于第 i 个物品所需时间，不放入第 i 个物品
                dp[i][j] = dp[i - 1][j];
            }
        }
    }

    // 输出在总时间 t 内可以获得的最大价值
    cout << dp[m][t] << endl;
    return 0;
}