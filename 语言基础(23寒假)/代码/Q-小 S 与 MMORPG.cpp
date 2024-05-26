//Q
#include<iostream>
#include<set>
using namespace std;
int main()
{
    int n,m,k;
    set<string>a;
	cin>>n>>m>>k;
    for(int i=0;i<n;i++)
    {
        string s;
        cin>>s;
        a.insert(s);
    }
    for(int i=0;i<m;i++)
    {
        string s;
        cin>>s;
        a.erase(s);
    }
    for(int i=0;i<k;i++)
    {
        string s;
        cin>>s;
        a.insert(s);
    }
    for(auto i:a)
	cout<<i<<endl;
    return 0;
}

//#include <stdio.h>
//#include <string.h>
//int main()
//{
//	int n,m,k;
//	scanf("%d%d%d",&n,&m,&k);
//	char name[n+m][11];
//	for(int i=0;i<n+m;i++)
//		scanf("%s",name[i]);
//	char ss[n+k-m][11];
//	int a=0;
//	for(int i=0;i<n;i++)
//	{
//		int flag=1;
//		for(int j=n;j<n+m;j++)
//		{
//			if(!(strcmp(name[i],name[j])))
//			{
//				flag=0;
//				break;
//			}
//		}
//		if(flag)
//		{
//			strcpy(ss[a++],name[i]);
//		}
//	}
//	for(int i=n-m;i<n+k-m;i++)
//		scanf("%s",ss[i]);
//	for(int i=0;i<n+k-m;i++)
//	{
//		for(int j=n+k-m-1;j>i;j--)
//		{
//			if((strcmp(ss[j],ss[j-1]))<0)
//			{
//				char temp[11];
//				strcpy(temp,ss[j]);
//				strcpy(ss[j],ss[j-1]);
//				strcpy(ss[j-1],temp);
//			}
//		}
//		printf("%s\n",ss[i]);
//	}
//	return 0;
//}
