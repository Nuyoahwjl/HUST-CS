#include<cstdio>
#include<cstring>
#include<cstdlib>
#include<iostream>
using namespace std;
int a[10][10];
bool h[10][10],l[10][10],g[10][10];//行，列，第几个格子
void print()//输出函数 
{
    for(int i=1;i<=9;i++)
    {
        for(int j=1;j<=9;j++)
            cout<<a[i][j]<<' ';
        cout<<endl;
    }
    exit(0);
}
void dfs(int x,int y)//深搜 
{
    if(a[x][y]!=0)//9*9中不为零的数直接跳过 
    {
        if(x==9&&y==9) 
            print();//搜索结束后输出 
        if(y==9) //行到顶端后搜索列 
            dfs(x+1,1); 
        else //搜索行 
            dfs(x,y+1);
    }
    if(a[x][y]==0)//等于零时 
    {
        for(int i=1;i<=9;i++)
        { 
            if(!h[x][i]&&!l[y][i]&&!g[(x-1)/3*3+(y-1)/3+1][i])
            {
                a[x][y]=i;
                h[x][i]=true;//
                l[y][i]=true;// 
                g[(x-1)/3*3+(y-1)/3+1][i]=true; 
                if(x==9&&y==9) //同a[x][y]!=0时                    
                    print();
                if(y==9) dfs(x+1,1); else dfs(x,y+1);
                a[x][y]=0;//当前格退出 
                h[x][i]=false;
                l[y][i]=false;
                g[(x-1)/3*3+(y-1)/3+1][i]=false;
            }
        } 
    }
}
int main()
{
    memset(h,false,sizeof(h));
    memset(l,false,sizeof(l));
    memset(g,false,sizeof(g));
    for(int i=1;i<=9;i++)
    {
        for(int j=1;j<=9;j++)
        {
            scanf("%d",&a[i][j]);
            if(a[i][j]>0)
            {
                h[i][a[i][j]]=true;//表示这一行有该数 
                l[j][a[i][j]]=true;//表示这一列有该数
                g[(i-1)/3*3+(j-1)/3+1][a[i][j]]=true;//表示这个格子有该数
            }
        }
    } 
    dfs(1,1);
    return 0;
}