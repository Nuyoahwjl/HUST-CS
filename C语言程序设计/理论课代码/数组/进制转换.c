#include<stdio.h>
#include<ctype.h>

void conversion(char str[]){
	/**********  Begin  **********/
    int sum=0;
    for(int i=0;str[i]!='\0'&&isxdigit(str[i]);i++){
        // if(str[i]='\0'||!(isxdigit(str[i])))
        // break;
        if(str[i]>='0'&&str[i]<='9')
        sum=sum*16+str[i]-'0';
        else if(str[i]>='a'&&str[i]<='f')
        sum=sum*16+str[i]-'a'+10;
        else
        sum=sum*16+str[i]-'A'+10;
    
    }
    printf("%d",sum);

	/**********  End  **********/
}
