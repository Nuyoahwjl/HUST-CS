#include <bits/stdc++.h>
using namespace std;

int main() {
    int m, s, t;
    cin >> m >> s >> t;

    int dp[300005] = {0}; // dp[i] 表示第 i 秒时守望者能走的最远距离
    bool escaped = false; // 是否成功逃离
    int escapeTime = 0;   // 逃离所需的最短时间

    // 第一遍循环：优先使用闪烁法术
    for (int i = 1; i <= t; i++) {
        if (m >= 10) {
            dp[i] = dp[i - 1] + 60; // 使用闪烁法术
            m -= 10;
        } else {
            dp[i] = dp[i - 1]; // 无法使用闪烁法术，选择休息
            m += 4;
        }
    }

    // 第二遍循环：考虑跑步的情况
    for (int i = 1; i <= t; i++) {
        dp[i] = max(dp[i], dp[i - 1] + 17); // 比较闪烁和跑步的距离
        if (dp[i] >= s) {
            escaped = true;
            escapeTime = i;
            break;
        }
    }

    // 输出结果
    if (escaped) {
        cout << "Yes\n" << escapeTime << endl;
    } else {
        cout << "No\n" << dp[t] << endl;
    }

    return 0;
}