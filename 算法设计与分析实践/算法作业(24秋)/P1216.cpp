#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main()
{
    int r;
    cin >> r;
    vector<vector<int>> a(r + 1, vector<int>(r + 1, 0)); // 使用vector代替变长数组

    for (int i = 1; i <= r; i++)
    {
        for (int j = 1; j <= i; j++)
        {
            cin >> a[i][j];
        }
    }

    for (int i = r - 1; i >= 1; i--)
    {
        for (int j = 1; j <= i; j++)
        {
            a[i][j] += max(a[i + 1][j], a[i + 1][j + 1]);
        }
    }

    cout << a[1][1] << endl;
    return 0;
}