#include <iostream>
#include <vector>
#include <cstring>
using namespace std;

const int MOD = 1e9 + 7;

int dp[20][200][2]; // dp[pos][sum][tight]

// 数位 DP，计算 1 到 num 的数字和
int digitSumDP(const string &num, int pos, int sum, bool tight) {
    if (pos == static_cast<int>(num.size())) return sum; // 数位结束，返回累加的和

    if (dp[pos][sum][tight] != -1) return dp[pos][sum][tight];

    int limit = tight ? num[pos] - '0' : 9;
    int res = 0;

    for (int digit = 0; digit <= limit; digit++) {
        res = (res + digitSumDP(num, pos + 1, sum + digit, tight && (digit == limit))) % MOD;
    }

    return dp[pos][sum][tight] = res;
}

// 包装函数，将数字转化为字符串并计算数字和
int digitSum(long long x) {
    string num = to_string(x);
    memset(dp, -1, sizeof(dp));
    return digitSumDP(num, 0, 0, 1);
}

int main() {
    int T;
    cin >> T;

    while (T--) {
        long long L, R;
        cin >> L >> R;

        // 计算区间和
        int sumR = digitSum(R);
        int sumL = digitSum(L - 1);

        int result = (sumR - sumL + MOD) % MOD;
        cout << result << endl;
    }

    return 0;
}
