//AA
#include <iostream>
#include <algorithm>
using namespace std;
int main()
{
	int N,B;
	cin>>N>>B;
	int a[N];
	for(int i=0;i<N;i++) cin>>a[i];
	sort(a,a+N);
	int sum=0;
	int num=0;
	for(int i=N-1;i>=0;i--)
	{
		num++;
		sum+=a[i];
		if(sum>=B)
		{
			printf("%d",num);
			break;
		}
	}
}
