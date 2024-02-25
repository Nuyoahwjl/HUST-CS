//X
#include <iostream>
#include <vector>
#include<algorithm>
using namespace std;
const int N=25;
int a[N],flag;
int s[4];
int ans,sum;
void f(int num,int x)//select
{
	if(num>=sum/2||x==flag)
	{
		ans=min(ans,max(sum-num,num));
		return;
	}
	f(num,x+1);//ฒปัก
	f(num+a[x],x+1);//ัก 
}

int main()
{
	int result=0;
	for(int i=0;i<4;i++)
		cin>>s[i];
	for(int i=0;i<4;i++)
	{
		sum=0,ans=100000;
		flag=s[i];
		for(int k=0;k<s[i];k++){
			cin>>a[k];
			sum+=a[k];
		}
		f(0,0);
		result+=ans;
	}
	cout<<result;
	return 0;
}
