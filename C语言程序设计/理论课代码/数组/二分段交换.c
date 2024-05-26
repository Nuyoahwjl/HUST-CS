#include<stdio.h>

void move(int arr[],int n,int k){
	/**********  Begin  **********/
    int a[n];
    int i;
    for(i=0;i<n-k;i++){
        a[i]=arr[i+k];
    }
    for(i=n-k;i<n;i++){
        a[i]=arr[i-n+k];
    }
    for(i=0;i<n;i++){
        arr[i]=a[i];
    }
 
    
    
    
	/**********  End  **********/
}