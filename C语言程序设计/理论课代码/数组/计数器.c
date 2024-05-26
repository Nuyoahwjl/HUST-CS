#include<stdio.h>

int main()
{
	/**********  Begin  **********/
	char a[128]={0};
	char c;
	while((c=getchar())!='\n'){
		if(c>='0'&&c<='9'||c>='A'&&c<='Z') a[c]++;
		else if(c>='a'&&c<='z') a[c-32]++;
		else a[0]++;
	}
	for(int i=1;i<128;i++){
		if(a[i]!=0)
			printf("%c:%d\n",i,a[i]);
	}
	if(a[0]!=0) printf("others:%d\n",a[0]);
/**********  End  **********/
	return 0; 
}