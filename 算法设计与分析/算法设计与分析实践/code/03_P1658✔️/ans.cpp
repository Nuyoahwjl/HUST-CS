#include <cstdio>
#include <iostream>
#include <algorithm>

using namespace std;

int X,N;
int a[20];

int main()
{
    cin >> X >> N;
    for(int i = 1; i <= N; i++)
        cin >> a[i];
    sort(a + 1, a + N + 1);
    if(a[1]!=1) // 一定无解
    {
        cout<<-1;
        return 0;
    }
    int sum=0; //sum为当前能表示的最大面额
    int num=0;
    while(sum<X)
    {
        for(int i=N;i>=1;i--) //从大的开始
        {
            if(a[i]<=sum+1)
            {
                sum+=a[i];
                num++;
                break;
            }
        }
    }
    cout<<num;
    return 0;
}