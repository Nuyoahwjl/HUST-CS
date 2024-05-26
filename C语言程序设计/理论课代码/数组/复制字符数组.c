#include<stdio.h>

void strnCpy(char t[],char s[],int n){
	/**********  Begin  **********/
     for(int i=0;i<n;i++){
         t[i]=s[i];
     }
     t[n]='\0';

	/**********  End  **********/
}
