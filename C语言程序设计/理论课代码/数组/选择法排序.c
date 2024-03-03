#include<stdio.h>


/*****请在下面补充函数 inArray、selectSort 和 outArray 的定义 *****/
void inArray(int arr[],int n)
{
    //scanf("%d",&n);
    for(int i=0;i<n;i++)
    scanf("%d ",&arr[i]);
}
void selectSort(int arr[],int n)
{
//   int mid;
//  for(int i = 1; i < n; i++){          //n-1轮，每轮将最大数移到最右边
//   for (int j = 0; j < n-i; j++) {
//    if(arr[j] >= arr[j+1]){
//     mid = arr[j];
//     arr[j] = arr[j+1];
//     arr[j+1] = mid;
//    }
//   }
//  }
    for(int i=0;i<n;i++){
      int max=0,temp;
      for(int j=0;j<n-i;j++){
      if(arr[j]>=arr[max]) max=j;
      }
      temp = arr[max];
      arr[max] = arr[n-i-1];
      arr[n-i-1] = temp;
    }
}
void outArray(int arr[],int n)
{
    for(int i=0;i<n;i++) 
        printf("%d ",arr[i]);
}