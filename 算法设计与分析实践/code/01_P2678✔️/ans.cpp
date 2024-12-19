#include <iostream>
#include <cstdio>
#include <algorithm>
#define OK 1
#define NO 0

using namespace std;

int L, N, M, d[50001];
bool check(int dist);

int main()
{
    cin >> L >> N >> M;
    for(int i = 1; i <= N; i++)
        cin >> d[i];
    sort(d + 1, d + N + 1);
    int low = 1, high = L, mid;
    while(low <= high)
    {
        mid = (low + high) / 2;
        if(check(mid) == OK)
            low = mid + 1;
        else
            high = mid - 1;
    }
    cout << high << endl;  // 输出结果应该是high
}

bool check(int dist)  // 判断能否最多移走M块石头使得每块石头之间的距离都大于等于dist
{
    int cnt = 0, last = 0;
    for(int i = 1; i <= N; i++)
    {
        if(d[i] - last < dist)
            cnt++;
        else
            last = d[i];
    }
    if(L - last < dist) // 判断最后一块石头到终点的距离
        cnt++;
    return cnt <= M ? OK : NO;
}




