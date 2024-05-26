//AI
//#include <stdio.h> 
//int main()
//{
//	int n,d;
//	scanf("%d%d",&n,&d);
//	int Q[n][d],K[n][d],V[n][d];
//	int W[n];
//	for(int i=0;i<n;i++)
//		for(int j=0;j<d;j++)
//			scanf("%d",&Q[i][j]);
//	for(int i=0;i<n;i++)
//		for(int j=0;j<d;j++)
//			scanf("%d",&K[i][j]);
//	for(int i=0;i<n;i++)
//		for(int j=0;j<d;j++)
//			scanf("%d",&V[i][j]);
//	for(int i=0;i<n;i++)
//		scanf("%d",&W[i]);
//	
//	int temp[n][n];
//	for(int i=0;i<n;i++)
//		for(int j=0;j<n;j++)
//			temp[i][j]=0;
////	for(int i=0;i<n;i++)
////	{
////		for(int j=0;j<n;j++)
////		{
////			temp[i][j]=0;
////			for(int p=0;p<d;p++)
////			temp[i][j]+=Q[i][p]*K[j][p];
////		}
////	}
//for(int i=0;i<n;i++)
//	{
//		for(int p=0;p<d;p++)
//		{
//			int s=Q[i][p];
//			for(int j=0;j<n;j++)
//			temp[i][j]+=s*K[j][p];
//		}
//	}
//	for(int i=0;i<n;i++)
//		for(int j=0;j<n;j++)
//			temp[i][j]*=W[i];
//	
//	int end[n][d];
//	for(int i=0;i<n;i++)
//		for(int j=0;j<d;j++)
//			end[i][j]=0;
////	for(int i=0;i<n;i++)
////	{
////		for(int j=0;j<d;j++)
////		{
////			end[i][j]=0;
////			for(int p=0;p<n;p++)
////			end[i][j]+=temp[i][p]*V[p][j];
////		}
////	}
//	for(int i=0;i<n;i++)
//	{
//		for(int p=0;p<n;p++)
//		{
//			int s=temp[i][p];
//			for(int j=0;j<d;j++)
//			end[i][j]+=s*V[p][j];
//		}
//	}
//	for(int i=0;i<n;i++)
//	{
//		for(int j=0;j<d;j++)
//		{
//			printf("%d ",end[i][j]);
//		}
//		printf("\n");
//	}	
//}


#include <iostream> 
#include <vector>
using namespace std;
int main()
{
	int n,d;
	cin>>n>>d;
	vector<vector<long long> >Q(n),K(n),V(n);
	vector<long long>W(n);
	for(int i=0;i<n;i++){
		Q[i].resize(d);
		for(int j=0;j<d;j++)
			cin>>Q[i][j];
	}	
	for(int i=0;i<n;i++){
		K[i].resize(d);
		for(int j=0;j<d;j++)
			cin>>K[i][j];
	}	
	for(int i=0;i<n;i++){
		V[i].resize(d);
		for(int j=0;j<d;j++)
			cin>>V[i][j];
	}	
	for(int i=0;i<n;i++)
		cin>>W[i];
	
	vector<vector<long long> >temp(d);
	for(int i=0;i<d;i++)
		temp[i].resize(d);
	for(int i=0;i<d;i++)
	{
		for(int j=0;j<d;j++)
		{
			for(int p=0;p<n;p++)
			temp[i][j]+=K[p][i]*V[p][j];
		}
	}
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<d;j++)
		{
			K[i][j]=0;
			for(int p=0;p<d;p++)
			K[i][j]+=Q[i][p]*temp[p][j];
		}
	}
	
	for(int i=0;i<n;i++)
		for(int j=0;j<d;j++)
			K[i][j]*=W[i];
	
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<d;j++)
		{
			if(j!=0)
				cout<<" ";
			cout<<K[i][j];
		}
		cout<<endl;
	}	
	return 0;
}
