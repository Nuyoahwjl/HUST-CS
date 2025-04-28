#include <cstdio>
#include <iostream>
#include <cstring>

using namespace std;

char s[255];
int k;
// 删除减区间的第一个数
// int main()
// {
//     cin >> s >> k;
//     int len = strlen(s);
//     s[len]=0; // 末尾加上一个0,方便删除末尾的数
//     while(k--)
//     {
//         for(int i=0;i<len;i++)
//         {
//             if(s[i]!=-1)
//             {
//                 int j=i+1;
//                 while(s[j]==-1)
//                     j++;
//                 if(s[i]>s[j])
//                 {
//                     s[i]=-1;
//                     break;
//                 }
//             }
//         }
//     }
//     // 去除前导零,eg: 101 1 -> 1
//     int i;
//     for(i=0;i<len;i++)
//     {
//         if(s[i]>'0')
//             break;
//         s[i]=-1;
//     }
//     if(i==len)
//         cout<<0;
//     else
//         for(int i=0;i<len;i++)
//         {
//             if(s[i]!=-1)
//                 cout<<s[i];
//         }
// }

// 维护一个局部递增的栈
int main()
{
    cin >> s >> k;
    int len = strlen(s);
    char stack[255];
    int top = -1;
    for(int i=0;i<len;i++)
    {
        while(top>=0&&k&&stack[top]>s[i])
        {
            top--;
            k--;
        }
        stack[++top]=s[i];
    }
    if(k)
        top-=k;
    int i=0;
    while(i<=top&&stack[i]=='0')
        i++;
    if(i>top)
        cout<<0;
    else
        for(;i<=top;i++)
            cout<<stack[i];
    
}


