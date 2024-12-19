#include <iostream>
#include <queue>
#include <climits>
#include <utility>
#include <vector>

using namespace std;

int m;
vector<vector<int>> a(310, vector<int>(310, INT_MAX));
int dx[4] = {0, 0, 1, -1};
int dy[4] = {1, -1, 0, 0};
queue<pair<pair<int, int>, int>> q;
int res = INT_MAX;
vector<vector<bool>> visited(310, vector<bool>(310, false)); // 访问数组

//横纵坐标不能小于0，但可以大于300
int main()
{
    cin >> m;
    for (int i = 0; i < m; i++)
    {
        int x, y, t;
        cin >> x >> y >> t;
        a[x][y] = min(a[x][y], t);
        for (int j = 0; j < 4; j++)
        {
            int nx = x + dx[j];
            int ny = y + dy[j];
            if (nx >= 0 && ny >= 0)
                a[nx][ny] = min(a[nx][ny], t);
        }
    }
    q.push(make_pair(make_pair(0, 0), 0));
    visited[0][0] = true; // 标记起始点为已访问
    while (!q.empty())
    {
        pair<pair<int, int>, int> p = q.front();
        q.pop();
        if (a[p.first.first][p.first.second] == INT_MAX)
        {
            res = p.second;
            break;
        }
        for (int i = 0; i < 4; i++)
        {
            int nx = p.first.first + dx[i];
            int ny = p.first.second + dy[i];
            if (nx >= 0 && ny >= 0 && !visited[nx][ny] && p.second + 1 < a[nx][ny])
            {
                q.push(make_pair(make_pair(nx, ny), p.second + 1));
                visited[nx][ny] = true; // 标记为已访问
            }
        }
    }
    cout << (res == INT_MAX ? -1 : res);
    return 0;
}