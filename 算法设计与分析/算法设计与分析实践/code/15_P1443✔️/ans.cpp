#include <iostream>
#include <iomanip>
#include <queue>
#include <utility>

using namespace std;

int n,m,x,y;
queue<pair<int,int>> q;
vector<vector<int>> res(401,vector<int>(401,-1));
int dx[8]={1,1,-1,-1,2,2,-2,-2};
int dy[8]={2,-2,2,-2,1,-1,1,-1};

int main()
{
    cin>>n>>m>>x>>y;
    q.push(make_pair(x,y));
    res[x][y]=0;
    while(!q.empty())
    {
        pair<int,int> p=q.front();
        q.pop();
        for(int i=0;i<8;i++)
        {
            int nx=p.first+dx[i];
            int ny=p.second+dy[i];
            if(nx>0&&nx<=n&&ny>0&&ny<=m&&res[nx][ny]==-1)
            {
                res[nx][ny]=res[p.first][p.second]+1;
                q.push(make_pair(nx,ny));
            }
        }
    }
    for(int i=1;i<=n;i++)
    {
        for(int j=1;j<=m;j++)
            cout<<left<<setw(5)<<res[i][j];
        cout<<endl;
    }
    return 0;
}