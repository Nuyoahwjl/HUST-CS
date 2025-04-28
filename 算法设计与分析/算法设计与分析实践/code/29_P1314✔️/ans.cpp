#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

long long n,m,s;
long long ans=1e15; 
vector<int> w; //重量
vector<int> v; //价值
vector<long long> w_sum; //前缀和数组(>=W的重量个数)
vector<long long> v_sum; //前缀和数组(>=W的价值总和)
struct node
{
    int l,r;
};
vector<node> range; //区间

int main()
{
    cin>>n>>m>>s;
    w.resize(n+1);
    v.resize(n+1);
    w_sum.resize(n+1);
    v_sum.resize(n+1);
    range.resize(m+1);
    int left=1,right; //二分左右边界
    for(int i=1;i<=n;i++)
    {
        cin>>w[i]>>v[i];
        right=max(right,w[i]); //更新右边界
    }
    for(int i=1;i<=m;i++)
        cin>>range[i].l>>range[i].r;
    while(left<=right) //二分
    {
        int mid=(left+right)/2;
        for(int i=1;i<=n;i++)
        {
            if(w[i]>=mid) //更新前缀和数组
            {
                w_sum[i]=w_sum[i-1]+1;
                v_sum[i]=v_sum[i-1]+v[i];
            }
            else
            {
                w_sum[i]=w_sum[i-1];
                v_sum[i]=v_sum[i-1];
            }
        }
        long long sum=0;
        for(int i=1;i<=m;i++) //计算y
            sum+=(w_sum[range[i].r]-w_sum[range[i].l-1])*(v_sum[range[i].r]-v_sum[range[i].l-1]);
        if(sum<=s) 
            right=mid-1;
        else
            left=mid+1;
        ans=min(ans,abs(sum-s));
    }
    cout<<ans;
    return 0;
}