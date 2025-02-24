#include <bits/stdc++.h>
using namespace std;

const int MAX = 25; // 定义最大网格大小
long long n, m, a, b;
long long dp[MAX][MAX]; // dp[i][j] 表示从 (1,1) 到 (i,j) 的路径数
bool blocked[MAX][MAX]; // 标记不可通过的格子

// 标记马的控制点
void markBlocked(int x, int y) {
    if (x >= 1 && x <= n && y >= 1 && y <= m) {
        blocked[x][y] = true;
    }
}

void markHorseControl(int x, int y) {
    markBlocked(x, y);
    markBlocked(x - 1, y - 2);
    markBlocked(x - 2, y - 1);
    markBlocked(x - 2, y + 1);
    markBlocked(x - 1, y + 2);
    markBlocked(x + 1, y - 2);
    markBlocked(x + 2, y - 1);
    markBlocked(x + 2, y + 1);
    markBlocked(x + 1, y + 2);
}

int main() {
    scanf("%lld %lld %lld %lld", &n, &m, &a, &b);
    a++; b++; n++; m++; // 转换为1-based坐标

    // 初始化 blocked 数组
    memset(blocked, 0, sizeof(blocked));
    markHorseControl(a, b);

    // 初始化 dp 数组
    memset(dp, 0, sizeof(dp));
    dp[1][1] = 1;

    // 动态规划计算路径数
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            if (i == 1 && j == 1) continue; // 起点已经初始化
            if (!blocked[i][j]) {
                dp[i][j] = (i > 1 ? dp[i - 1][j] : 0) + (j > 1 ? dp[i][j - 1] : 0);
            }
        }
    }

    printf("%lld\n", dp[n][m]);
    return 0;
}