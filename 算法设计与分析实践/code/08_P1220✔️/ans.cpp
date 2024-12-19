#include <cstdio>
#include <cstring>
#include <algorithm>
#include <iostream>
using namespace std;

const int INF = 0x3f3f3f3f; // 定义一个大数表示无穷大，用于初始化 DP 数组
const int MAXN = 60;        // 最大路灯数量

int n, c;                   // 路灯总数 n，老张初始位置的路灯编号 c
int dis[MAXN];              // 路灯的位置信息
int p[MAXN];                // 路灯功率的前缀和数组
int dp[MAXN][MAXN][2];      // 动态规划数组

int main() {
    // 输入路灯的数量 n 和老张的初始位置 c
    cin >> n >> c;

    int power;
    for (int i = 1; i <= n; i++) {
        // 输入每盏路灯的位置和功率
        cin >> dis[i] >> power;
        p[i] = p[i - 1] + power; // 计算路灯功率的前缀和
    }

    // 初始化 DP 数组为无穷大
    memset(dp, INF, sizeof(dp));
    
    // 初始状态：老张从第 c 个路灯出发，消耗为 0
    dp[c][c][0] = dp[c][c][1] = 0;

    // 枚举区间长度 len，从 1 到 n-1
    for (int len = 1; len < n; len++) {
        // 枚举区间左端点 l，右端点 r = l + len - 1
        for (int l = 1; l <= n - len + 1; l++) {
            int r = l + len - 1;

            // 更新区间 [l-1, r]，最后一个关掉的是 l-1
            dp[l - 1][r][0] = min(dp[l - 1][r][0],
                                  min(
                                      dp[l][r][0] + (dis[l] - dis[l - 1]) * (p[l - 1] + p[n] - p[r]),
                                      dp[l][r][1] + (dis[r] - dis[l - 1]) * (p[l - 1] + p[n] - p[r])
                                  ));

            // 更新区间 [l, r+1]，最后一个关掉的是 r+1
            dp[l][r + 1][1] = min(dp[l][r + 1][1],
                                  min(
                                      dp[l][r][1] + (dis[r + 1] - dis[r]) * (p[l - 1] + p[n] - p[r]),
                                      dp[l][r][0] + (dis[r + 1] - dis[l]) * (p[l - 1] + p[n] - p[r])
                                  ));
        }
    }

    // 输出答案：从区间 [1, n] 中选择最小功耗
    cout << min(dp[1][n][0], dp[1][n][1]);

    return 0;
}


















// #include <iostream>
// #include <vector>
// #include <algorithm>
// using namespace std;

// int n, c;
// int d[60], w[60];
// int sum = 0;

// int opt(int n, int current_sum, vector<bool>& flag);
// int findleft(int n, const vector<bool>& flag);
// int findright(int n, const vector<bool>& flag);

// int main() {
//     cin >> n >> c;
//     if (n <= 0 || n > 59 || c < 1 || c > n) {
//         cerr << "Invalid input" << endl;
//         return 1;
//     }

//     for (int i = 1; i <= n; i++) {
//         cin >> d[i] >> w[i];
//         sum += d[i];
//     }

//     vector<bool> flag(60, false);
//     flag[c] = true;

//     cout << opt(c, sum - w[c], flag) << endl;
//     return 0;
// }

// int opt(int n, int current_sum, vector<bool>& flag) {
//     int left = findleft(n, flag);
//     int right = findright(n, flag);

//     if (left == 0 && right == 0) {
//         return 0;
//     }

//     if (left == 0) {
//         flag[right] = true;
//         int result = current_sum * (right - n) + opt(right, current_sum - w[right], flag);
//         flag[right] = false;
//         return result;
//     }

//     if (right == 0) {
//         flag[left] = true;
//         int result = current_sum * (n - left) + opt(left, current_sum - w[left], flag);
//         flag[left] = false;
//         return result;
//     }

//     flag[left] = true;
//     int res_left = current_sum * (n - left) + opt(left, current_sum - w[left], flag);
//     flag[left] = false;

//     flag[right] = true;
//     int res_right = current_sum * (right - n) + opt(right, current_sum - w[right], flag);
//     flag[right] = false;

//     return min(res_left, res_right);
// }

// int findleft(int n, const vector<bool>& flag) {
//     for (int i = n - 1; i >= 1; i--) {
//         if (!flag[i]) {
//             return i;
//         }
//     }
//     return 0;
// }

// int findright(int n, const vector<bool>& flag) {
//     for (int i = n + 1; i <= 59; i++) {
//         if (!flag[i]) {
//             return i;
//         }
//     }
//     return 0;
// }
