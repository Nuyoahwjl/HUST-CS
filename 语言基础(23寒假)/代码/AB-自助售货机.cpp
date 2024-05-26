//AB
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;
int main()
{
	int n,m,k;
	cin>>n>>m>>k;
//    vector<int>a(n*m+1,n);
	int h[k+1];
	for(int i=1;i<=k;i++)
		cin>>h[i];
	sort(h+1,h+k+1);
	int flag=1;
	int a=1;
	for(int i=1;i<=k;i++)
	{
		if(h[i]<a)
		{
			flag=0;
			break;
		}
		if(i%m==0) a++;
	}
	if(flag==1) cout<<"Yes";
	else cout<<"No";
	return 0;
}
