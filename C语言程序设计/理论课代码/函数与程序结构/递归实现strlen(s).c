#include<stdlib.h>
#include<stdio.h>
#include<string.h>

//请根据step4_main.c中主函数流程
//使用递归的方法补全此函数
int b=0;
int  mystrlen(char *s)
{

	/**********  Begin  **********/
    if(*s)
{
    b++;
    mystrlen(s+1);
}
  else
   return b;

    
    
    /**********  End  **********/
}