#include <iostream>
#include <cstdio>
#include <algorithm>
#include <climits>

using namespace std;

//以下是无法记录决策过程的递归版
// int f,v;
// int value[101][101];
// int maxvalue(int f_start,int f_end,int v_start,int V_end);
// int main()
// {
//     cin>>f>>v;
//     for(int i=1;i<=f;i++)
//         for(int j=1;j<=v;j++)
//             cin>>value[i][j];
//     cout<<maxvalue(1,f,1,v);
//     return 0;
// }
// int maxvalue(int f_start,int f_end,int v_start,int v_end)
// {
//     if(f_end-f_start==v_end-v_start)
//     {
//         int sum=0;
//         for(int i=0;i<=f_end-f_start;i++)
//             sum+=value[f_start+i][v_start+i];
//         return sum;
//     }
//     if(f_end==f_start)
//     {
//         int max=0;
//         for(int i=0;i<=v_end-v_start;i++)
//             if(value[f_start][v_start+i]>max)
//                 max=value[f_start][v_start+i];
//         return max;
//     }
//     return max(maxvalue(f_start,f_end,v_start+1,v_end),maxvalue(f_start+1,f_end,v_start+1,v_end)+value[f_start][v_start]);
// }

//以下是可以记录决策过程的递归版
// int f,v;
// int value[101][101];
// bool decision[101][101];
// int maxvalue(int f_start,int f_end,int v_start,int V_end);
// int main()
// {
//     cin>>f>>v;
//     for(int i=1;i<=f;i++)
//         for(int j=1;j<=v;j++)
//             cin>>value[i][j];
//     cout<<maxvalue(1,f,1,v)<<endl;
//     for(int i=1,j=1;i<=f&&j<=v;)
//     {
//         if(decision[i][j])
//         {
//             cout<<j<<' ';
//             i++;
//             j++;
//         }
//         else
//             j++;
//     }
//     return 0;
// }
// int maxvalue(int f_start, int f_end, int v_start, int v_end) {
//     if (f_end - f_start == v_end - v_start) {
//         int sum = 0;
//         for (int i = 0; i <= f_end - f_start; i++) {
//             sum += value[f_start + i][v_start + i];
//             decision[f_start + i][v_start + i] = 1; // 记录决策
//         }
//         return sum;
//     }
//     if (f_end == f_start) {
//         int max_val = 0, best_col = -1;
//         for (int i = 0; i <= v_end - v_start; i++) {
//             if (value[f_start][v_start + i] > max_val) {
//                 max_val = value[f_start][v_start + i];
//                 best_col = v_start + i;
//             }
//         }
//         if (best_col != -1)
//             decision[f_start][best_col] = 1; // 记录决策
//         return max_val;
//     }
//     int option1 = maxvalue(f_start, f_end, v_start + 1, v_end); 
//     int option2 = maxvalue(f_start + 1, f_end, v_start + 1, v_end) + value[f_start][v_start];
//     if (option2 > option1) {
//         decision[f_start][v_start] = 1; // 记录当前选择
//         return option2;
//     }
//     return option1;
// }

//以下是非递归版
int f,v;
int value[101][101];
int dp[101][101]={INT_MIN};
int result[101];
// dp[i][j]表示前i个花朵放在前j个花瓶中的最大价值
// dp[i][j]=max(dp[i-1][j-1]+value[i][j],dp[i][j-1])
int main()
{
    cin>>f>>v;
    for(int i=1;i<=f;i++)
        for(int j=1;j<=v;j++)
            cin>>value[i][j];
    if(f==v)
    {
        int sum=0;
        for(int i=1;i<=f;i++)
            sum+=value[i][i];
        cout<<sum<<endl;
        for(int i=1;i<=f;i++)
            cout<<i<<' ';
        return 0;
    }
    // for(int k=1;k<=v-f+1;k++)
    // {
    //     int max=0;
    //     for(int i=1;i<=k;i++)
    //         if(value[1][i]>max)
    //             max=value[1][i];
    //     dp[1][k]=max;
    // }
    dp[1][1]=value[1][1];
    for(int j=2;j<=v;j++)
    {
        for(int i=1;i<=j;i++)
        {
            dp[i][j]=max(dp[i-1][j-1]+value[i][j],dp[i][j-1]);
        }
    }
    cout<<dp[f][v]<<endl;
    int i=f,j=v;
    while(i>0&&j>0)
    {
        if(dp[i][j]==dp[i][j-1])
            j--;
        else
        {
            result[i]=j;
            i--;
            j--;
        }
    }
    for(int i=1;i<=f;i++)
        cout<<result[i]<<' ';
    return 0;
}

