#include <iostream>
#include <vector>
#include <algorithm> 

using namespace std;

vector<int> a;
vector<int> t;

void build(int id,int l,int r)
{
    if(l==r)
    {
        t[id]=a[l];
        return;
    }
    int mid=(l+r)/2;
    build(id*2,l,mid);
    build(id*2+1,mid+1,r);
    t[id]=min(t[id*2],t[id*2+1]);
}

int query(int id,int l,int r,int x,int y)
{
    if(x<=l&&r<=y)
        return t[id];
    int mid=(l+r)/2;
    int res=0x7fffffff;
    if(x<=mid)
        res=min(res,query(id*2,l,mid,x,y));
    if(y>mid)
        res=min(res,query(id*2+1,mid+1,r,x,y));
    return res;
}

int main()
{
    int n,m;
    cin>>n>>m;
    a.resize(n+1);
    t.resize(4*n+1);
    for(int i=1;i<=n;i++)
        cin>>a[i];
    build(1,1,n);
    for(int i=0;i<m;i++)
    {
        int x,y;
        cin>>x>>y;
        cout<<query(1,1,n,x,y)<<" ";
    }
    return 0;
}