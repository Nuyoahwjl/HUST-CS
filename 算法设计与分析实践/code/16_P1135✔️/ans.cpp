#include <iostream>
#include <queue>
#include <climits>
#include <utility>
#include <vector>

using namespace std;

int n, a, b;
int v[201];
int res = INT_MAX;
queue<pair<int, int>> q;
vector<bool> visited(201, false); // 访问数组

int main()
{
    cin >> n >> a >> b;
    for (int i = 1; i <= n; i++)
        cin >> v[i];
    q.push(make_pair(a, 0));
    visited[a] = true; // 标记起始点为已访问
    while (!q.empty())
    {
        pair<int, int> p = q.front();
        q.pop();
        if (p.first == b)
        {
            res = min(res, p.second);
            continue;
        }
        if (p.first + v[p.first] <= n && !visited[p.first + v[p.first]])
        {
            q.push(make_pair(p.first + v[p.first], p.second + 1));
            visited[p.first + v[p.first]] = true; // 标记为已访问
        }
        if (p.first - v[p.first] > 0 && !visited[p.first - v[p.first]])
        {
            q.push(make_pair(p.first - v[p.first], p.second + 1));
            visited[p.first - v[p.first]] = true; // 标记为已访问
        }
    }
    cout << (res == INT_MAX ? -1 : res);
    return 0;
}