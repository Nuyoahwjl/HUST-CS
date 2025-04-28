#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

inline int read() {
    int x = 0, f = 1;
    char ch = getchar();
    while (ch < '0' || ch > '9') {
        if (ch == '-') f = -1;
        ch = getchar();
    }
    while (ch >= '0' && ch <= '9') {
        x = x * 10 + ch - 48;
        ch = getchar();
    }
    return x * f;
}

int main() {
    int N = read(), M = read();
    vector<int> arr(N + 1);  // 1-indexed array
    for (int i = 1; i <= N; i++) {
        arr[i] = read();
    }

    // Step 1: Build the Sparse Table (ST Table)
    int max_log = log2(N) + 1;
    vector<vector<int>> st(N + 1, vector<int>(max_log));
    
    // Initialize the sparse table for intervals of length 1 (2^0)
    for (int i = 1; i <= N; i++) {
        st[i][0] = arr[i];
    }

    // Fill the sparse table for larger intervals
    for (int j = 1; (1 << j) <= N; j++) {
        for (int i = 1; i + (1 << j) - 1 <= N; i++) {
            st[i][j] = max(st[i][j - 1], st[i + (1 << (j - 1))][j - 1]);
        }
    }

    // Step 2: Answer the queries
    while (M--) {
        int l = read(), r = read();
        int len = r - l + 1;
        int k = log2(len);  // largest power of 2 less than or equal to len
        // Query the maximum in range [l, r]
        cout << max(st[l][k], st[r - (1 << k) + 1][k]) << endl;
    }

    return 0;
}
