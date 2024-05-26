//Z
#include <iostream>
#include <algorithm> 
#include <set>
#include <vector>
using namespace std;
int read() 
{
	int f = 1, k = 0;
	char c = getchar();
	//非数字 
	while(c < '0' || c > '9'){
		if(c == '-') 
			f = -1;
		c = getchar(); 
	}
	//数字 
	while(c >= '0' && c <= '9'){
		k=k * 10 + c - '0';
		c = getchar(); 
	}
	return f * k;
	
}
int main()
{
	int n,Q;
	cin>>n>>Q;
	set<long>x;
	vector<long>s;
	//set不能随机访问，所以存到vector中 
	int result[Q];
	for(int i=0;i<n;i++)
	{
		long a,b,c;
		a=read();
		b=read();
		c=read();
		x.insert((c-b)/a);
	}
	for(set<long>::iterator it=x.begin();it!=x.end();it++)
	{
		s.push_back(*it);
	}
	for(int i=0;i<Q;i++)
	{
		result[i]=0;
		long L,R;
		cin>>L>>R;
		result[i]=(upper_bound(s.begin(),s.end(),R)-lower_bound(s.begin(),s.end(),L));
	}
	for(int i=0;i<Q;i++)
		cout<<result[i]<<endl;
	return 0;
} 
