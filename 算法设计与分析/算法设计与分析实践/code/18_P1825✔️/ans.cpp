#include <iostream>
#include <queue>
#include <climits>
#include <utility>
#include <vector>

using namespace std;

int n, m;
vector<vector<char>> a;
pair<int, int> in;
pair<int, int> out;
int dx[4] = {0, 0, 1, -1};
int dy[4] = {1, -1, 0, 0};
queue<pair<pair<int, int>, int>> q;
vector<vector<bool>> visited;
// bool flag[26] = {false};

// pair<int, int> find(int x, int y)
// {
//     for (int i = 1; i <= n; i++)
//         for (int j = 1; j <= m; j++)
//             if (a[i][j] == a[x][y] && (i != x || j != y))
//             {
//                 flag[a[i][j] - 'A'] = !flag[a[i][j] - 'A'];
//                 return make_pair(i, j);
//             }
//     return make_pair(-1, -1);
// }
void goto_another_teleport(int &x, int &y)
{
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++)
            if (a[i][j] == a[x][y] && (i != x || j != y))
            {
                x=i;
                y=j;
                return;
            }
}

int main()
{
    cin >> n >> m;

    a.resize(n + 1, vector<char>(m + 1, 0));
    visited.resize(n + 1, vector<bool>(m + 1, false));

    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++)
        {
            cin >> a[i][j];
            if (a[i][j] == '@')
            {
                in = make_pair(i, j);
                visited[i][j] = true;
            }
            if (a[i][j] == '=')
                out = make_pair(i, j);
        }

    q.push(make_pair(in, 0));
    while (!q.empty())
    {
        pair<pair<int, int>, int> p = q.front();
        q.pop();
        if (p.first == out)
        {
            cout << p.second;
            return 0;
        }
        if (a[p.first.first][p.first.second] >= 'A' && a[p.first.first][p.first.second] <= 'Z')
        {
        //     if (!flag[a[p.first.first][p.first.second] - 'A'])
        //     {
        //         pair<int, int> teleport = find(p.first.first, p.first.second);
        //         q.push(make_pair(teleport, p.second));  
        //         continue;
        //     }
        //     flag[a[p.first.first][p.first.second] - 'A'] = !flag[a[p.first.first][p.first.second] - 'A'];
            goto_another_teleport(p.first.first, p.first.second);
        }
        for (int i = 0; i < 4; i++)
        {
            int nx = p.first.first + dx[i];
            int ny = p.first.second + dy[i];
            if (nx > 0 && nx <= n && ny > 0 && ny <= m && a[nx][ny] != '#' && !visited[nx][ny])
            {
                q.push(make_pair(make_pair(nx, ny), p.second + 1));
                visited[nx][ny] = true; // 标记为已访问
            }
        }
    }
    cout << -1; // 如果没有找到路径，输出-1
    return 0;
}


