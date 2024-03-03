#include <stdio.h>

//请根据step3_main.cpp中的主函数流程，补全此函数
int digit(long n, int k)
{
	/**********  Begin  **********/
    int i=0,a;
    a=n;
    do{
        a/=10;
        i++;
    }while(a!=0);
    if(k>i)
    return -1;
    else{
        for(int j=1;j<k;j++){
            n/=10;
        }
     
     return n%10;
    }
    
    
    /**********  End  **********/
}