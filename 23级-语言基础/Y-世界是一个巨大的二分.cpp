//Y
#include <iostream>    
#include <algorithm>    
#include <vector> 
using namespace std;

int main()
{
	int n,m;
	cin>>n>>m;
	vector<int>a(n);
	for(int i=0;i<n;i++)
		cin>>a[i];
	sort(a.begin(),a.end());
	int result[m];
	int j=0;
	for(int i=0;i<m;i++)
	{
		int task;
		cin>>task;
		switch(task)
		{
			case 1: int x1;
	   				cin>>x1;
					result[j++]=(upper_bound(a.begin(),a.end(),x1)-lower_bound(a.begin(),a.end(),x1));	
					break;
			case 2: int x2,y2;
					cin>>x2>>y2;
					result[j++]=(upper_bound(a.begin(),a.end(),y2)-lower_bound(a.begin(),a.end(),x2));	
					break;
			case 3: int x3,y3;
					cin>>x3>>y3;
					result[j++]=(lower_bound(a.begin(),a.end(),y3)-lower_bound(a.begin(),a.end(),x3));	
					break;
			case 4:	int x4,y4;
					cin>>x4>>y4;
					result[j++]=(upper_bound(a.begin(),a.end(),y4)-upper_bound(a.begin(),a.end(),x4));	
			        break;
			case 5: int x5,y5;
					cin>>x5>>y5;
					result[j++]=(lower_bound(a.begin(),a.end(),y5)-upper_bound(a.begin(),a.end(),x5));	
			        break;
		}
	}
	for(int i=0;i<m;i++)
	{
		cout<<(result[i]>=0?result[i]:0)<<endl;
	}
	return 0;
}
