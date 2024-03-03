#include<stdio.h>
#include<string.h>
struct web  /*  设计表示网址的结构  */
{
    char s1[10];
    char s2[50];
    char s3[100];
};
void sort(struct web *p,int n);

#define N 4      /*  网址表大小  */

int main()
{
    struct web T[N];
    struct web *p=T;
    char s[10];
    for(int i=0;i<N;i++){
    	scanf("%s%s%s",(p+i)->s1,(p+i)->s2,(p+i)->s3);
	}
	scanf("%s",s);
	sort(T,N);
    for(int i=0;i<N;i++){
    	printf("%s %s %s\n",(p+i)->s1,(p+i)->s2,(p+i)->s3);
	}
	int flag=0;
	for(int i=0;i<N;i++){
		if(strcmp(s,(p+i)->s1)==0){
			printf("%s",(p+i)->s3);
			flag=1;
			break;
		}
	}
	if(flag==0) printf("未找到搜寻的网址"); 
	return 0; 
}

void sort(struct web *p,int n)
{
    struct web temp;
	for(int i=1;i<N;i++){
		for(int j=0;j<N-i;j++){
			if(strcmp((p+j)->s1,(p+j+1)->s1)>0){
				temp=*(p+j);
				*(p+j)=*(p+j+1);
				*(p+j+1)=temp;
			}	
		}
	} 
}