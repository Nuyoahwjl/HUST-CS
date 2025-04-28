#include <iostream>
#include <cstdio>
#include <algorithm>
#include <queue>

using namespace std;

int main()
{
    int n,temp;
    // 定义最小优先队列，底层容器使用 vector，比较器使用 std::greater
    priority_queue<int, vector<int>, greater<int>> minHeap;
    long long ans =0;
    cin>>n;
    for(int i=0;i<n;i++)
    {
        cin>>temp;
        minHeap.push(temp);
    }
    while(minHeap.size()>1)
    {
        temp=0;
        temp+=minHeap.top();
        minHeap.pop();
        temp+=minHeap.top();
        minHeap.pop();
        ans+=temp;
        minHeap.push(temp);
    }
    cout<<ans;
    return 0;
}