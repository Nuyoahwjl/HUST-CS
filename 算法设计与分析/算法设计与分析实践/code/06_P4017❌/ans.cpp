#include <iostream>
#include <cstdio>
#include <algorithm>
#include <stack>
#include <vector>

using namespace std;
int main()
{
    int n,m;
    cin>>n>>m;

    vector<vector<bool>> eat(n + 1, vector<bool>(n + 1, false));
    vector<int> indegree(n + 1, 0);
    vector<int> outdegree(n + 1, 0);

    for(int i=0;i<m;i++)
    {
        int a,b;
        cin>>a>>b;
        eat[a][b]=true;
        indegree[b]++;
        outdegree[a]++;
    }
    int left=0,right=0;
    for(int i=1;i<=n;i++)
    {
        if(indegree[i]==0)
            left=i;  
        if(outdegree[i]==0)
            right=i;  
    }

    stack<int> s;
    int num=0;
    s.push(left);
    
    while(!s.empty())
    {
        int x=s.top();
        s.pop();
        if(x==right)
        {
            num=(num+1)%80112002;
            continue;
        }
        for(int i=1;i<=n;i++)
        {
            if(eat[x][i])
            {
                s.push(i);
            }
        }
    }
    cout<<num;
    return 0;
}