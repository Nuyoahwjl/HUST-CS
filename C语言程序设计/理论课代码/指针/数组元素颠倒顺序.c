#include<stdio.h>
//函数的输入为 int类型数组 和数组长度 
void reverseOrder(int a[ ],int n){
	//请在此处编辑您的代码
	/**********  Begin  **********/
    int *p=a;
    int temp;
    for(int i=0;i<n/2;i++){
        temp=*(p+i);
        *(p+i)=*(p+n-1-i);
        *(p+n-1-i)=temp;
    }
	/**********  End  **********/
}
