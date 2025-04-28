#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

struct node
{
    int b,e;
    int t;
};
vector <bool> range;
vector <node> a;
int num;

bool compare(node a,node b)
{
    if(a.e==b.e)
        return a.b>b.b;
    return a.e<b.e;
}

int main()
{
    int n,m;
    cin>>n>>m;
    range.resize(n+1,false);
    a.resize(m);
    for(int i=0;i<m;i++)
        cin>>a[i].b>>a[i].e>>a[i].t;
    sort(a.begin(),a.end(),compare);
    for(int i=0;i<m;i++)
    {
        int k=0;
        for(int j=a[i].b;j<=a[i].e;j++)
            if(range[j]==true)
                k++;
        if(k<a[i].t)
        {
            int temp=a[i].t-k;
            for(int j=a[i].e;j>=a[i].b;j--)
                if(range[j]==false)
                {
                    range[j]=true;
                    num++;
                    temp--;
                    if(temp==0)
                        break;
                }
        }
    }
    cout<<num;
    return 0;
}