#include <iostream>
#include <cstdio>

using namespace std;

int parent[150005];

// 查找 x 的根节点，并进行路径压缩
int find(int x)
{
    if(parent[x]!=x)
        parent[x]=find(parent[x]);
    return parent[x];
}

// 将两个元素 x 和 y 所在的集合合并
void union_sets(int x, int y)
{
    int fx=find(x);
    int fy=find(y);
    if(fx!=fy)
        parent[fx]=fy;
}

int main()
{
    int n,k;
    cin>>n>>k;
    // x为自己，x+n为x的天敌，x+2*n为x的猎物
    // x+n吃x，x吃x+2*n，x+2*n吃x+n
    for(int i=1;i<=3*n;i++)
        parent[i]=i;
    int ans=0;
    for(int i=0;i<k;i++)
    {
        int d,x,y;
        cin>>d>>x>>y;
        if(x>n||y>n)
        {
            ans++;
            continue;
        }
        if(d==1)
        {
            // y是x的天敌或者猎物
            if(find(x+n)==find(y)||find(x+2*n)==find(y))
            {
                ans++;
                continue;
            }
            else
            {
                union_sets(x,y);
                union_sets(x+n,y+n);
                union_sets(x+2*n,y+2*n);
            }
        }
        else if(d==2)
        {
            if(x==y)
            {
                ans++;
                continue;
            }
            //x和y是同类或者y是x的猎物
            if(find(x)==find(y)||find(x)==find(y+2*n))
            {
                ans++;
                continue;
            }
            else
            {
                union_sets(x,y+n);
                union_sets(x+n,y+2*n);
                union_sets(x+2*n,y);
            }
        }
    }
    cout<<ans;
    return 0;
}
