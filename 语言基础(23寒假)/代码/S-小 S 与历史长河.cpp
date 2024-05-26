//S
#include <iostream>
#include <cstring>
using namespace std;
int main()
{
	string S,T;
	int Q,ls,rs,lt,rt;
	cin >> S >> T;
	cin >> Q;		
	int flag[Q];
	for(int i=0;i<Q;i++)
	{
		cin >> ls >> rs >> lt >> rt;
		string s1=S.substr(ls-1,rs-ls+1);
		string s2=T.substr(lt-1,rt-lt+1);
		if((strcmp(s1.c_str(),s2.c_str()))<0)
            flag[i]=1;
		else if((strcmp(s1.c_str(),s2.c_str()))>0)
			flag[i]=-1;
			else flag[i]=0;	
	}
	for(int i=0;i<Q;i++)
	{
		if(flag[i]==1)
			cout << "yifusuyi" << endl;
		else if(flag[i]==-1)
			cout << "erfusuer" << endl;
			else cout << "ovo" << endl;
	}
	return 0;
}

//#include <stdio.h>
//#include <string.h>
//int main()
//{
//	char S[1001],T[1001];
//	int Q,ls,rs,lt,rt;
//	scanf("%s%s",S,T);
//	scanf("%d",&Q);
//	int flag[Q];
//	for(int i=0;i<Q;i++)
//	{
//		scanf("%d %d %d %d",&ls,&rs,&lt,&rt);
//		char s1[rs-ls+2],s2[rt-lt+2];
//		for(int j=0;j<rs-ls+1;j++)
//		{
//			s1[j]=S[ls-1];
//			ls++;
//		}
//		for(int j=0;j<rt-lt+1;j++)
//		{
//			s2[j]=T[lt-1];
//			lt++;
//		}
//		s1[rs-ls+1]='\0';
//		s2[rt-lt+1]='\0';
//		if((strcmp(s1,s2))<0)
//            flag[i]=1;
//		else if((strcmp(s1,s2))>0)
//			flag[i]=-1;
//			else flag[i]=0;	
//	}
//	for(int i=0;i<Q;i++)
//	{
//		if(flag[i]==1)
//			printf("yifusuyi\n");
//		else if(flag[i]==-1)
//			printf("erfusuer\n");
//			else printf("ovo\n");
//	}
//	return 0;
//}

