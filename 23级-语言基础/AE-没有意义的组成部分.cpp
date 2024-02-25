//AE
//#include <cmath>
//#define N 100
//#include <iostream>
//using namespace std;
//int main()
//{
//	int q;
//	cin>>q;
//	long long result[q];
//	for(int i=0;i<q;i++)
//	{
//		int e[N],p[N];
//		long long a;
//		int k;
//		cin>>a>>k;
//		result[i]=1;
//    	int cnt=0;
//    	for(int l=2;l*l<=a;l++)
//    	{
//       		if(a%l==0)
//        	{
//            	p[++cnt]=l;
//				e[cnt]=0;
//            	while(a%l==0)
//            	{
//                	a/=l;
//                	e[cnt]++;
//            	}
//            	if(e[cnt]>=k)
//            	{
//           		result[i]*=pow(p[cnt],e[cnt]);
//				}
//        	}
//    	}
////    	if(a>1)
////    	{
////    		p[++cnt]=a;
////    		e[cnt]=1;
////		}
//
//    } 
//	for(int i=0;i<q;i++)
//		cout<<result[i]<<endl;
//	return 0;
//}

#include<iostream>
using namespace std;
int main()
{
    int q;
    cin>>q;
    long long result[q];
	for(int i=0;i<q;i++)
    {
        long long a;
        int k;
        cin>>a>>k;
        result[i]=1;
        for(int j=2;j<=a/j;j++)
        {
            if(a%j==0)
            {
                int cnt=0;
                while(a%j==0)
                {
                    cnt++;
                    a/=j;
                }
                if(cnt>=k)
                {
                    while(cnt--)
                    {
                        result[i]*=j;
                    }
                }
            }
        }
    }
    for(int i=0;i<q;i++)
		cout<<result[i]<<endl;
    return 0;
}
