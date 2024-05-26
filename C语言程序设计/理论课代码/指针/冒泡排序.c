#include<stdio.h>
//冒泡升序排序 ，程序的输入为 int类型数组 和数组长度 
//请在此处编辑您的代码
/**********  Begin  **********/
void bubble_sort(int *p,int n){
     int temp;
     for(int i=1;i<n;i++){
         for(int j=0;j<n-i;j++){
             if (*(p+j)>=*(p+j+1)){
                 temp=*(p+j);
                 *(p+j)=*(p+j+1);
                 *(p+j+1)=temp;
             }
         }
     }
}
/**********  End  **********/