#include<bits/stdc++.h>

using namespace std;

//不带记忆化搜索，会超时
// int r,c;
// int a[101][101];
// int LongestPath(int x,int y);
// int main()
// {
//     cin>>r>>c;
//     int ans=0;
//     for(int i=1;i<=r;i++)
//         for(int j=1;j<=c;j++)
//             cin>>a[i][j];
//     for(int i=1;i<=r;i++)
//         for(int j=1;j<=c;j++)
//             ans=max(ans,LongestPath(i,j));
//     cout<<ans;
//     return 0;
// }
// int LongestPath(int x,int y)
// {
//     int dx[4]={0,0,1,-1};
//     int dy[4]={1,-1,0,0};
//     int max_length=0;
//     for(int i=0;i<4;i++)
//     {
//         int nx=x+dx[i];
//         int ny=y+dy[i];
//         if(nx>0&&nx<=r&&ny>0&&ny<=c&&a[nx][ny]<a[x][y])
//             max_length=max(max_length,LongestPath(nx,ny));
//     }
//     return max_length+1;
// }

//带记忆化搜索
int r,c;
int a[101][101];
int LongestPath(int x,int y);
int res[101][101]={0};
int main()
{
    cin>>r>>c;
    int ans=0;
    for(int i=1;i<=r;i++)
        for(int j=1;j<=c;j++)
            cin>>a[i][j];
    for(int i=1;i<=r;i++)
        for(int j=1;j<=c;j++)
            ans=max(ans,LongestPath(i,j));
    cout<<ans;
    return 0;
}
int LongestPath(int x,int y)
{
    if(res[x][y]!=0)
        return res[x][y];
    int dx[4]={0,0,1,-1};
    int dy[4]={1,-1,0,0};
    int max_length=0;
    for(int i=0;i<4;i++)
    {
        int nx=x+dx[i];
        int ny=y+dy[i];
        if(nx>0&&nx<=r&&ny>0&&ny<=c&&a[nx][ny]<a[x][y])
            max_length=max(max_length,LongestPath(nx,ny));
    }
    res[x][y]=max_length+1;
    return res[x][y];
}