//K
#include <stdio.h>
int main()
{
	int m,n;
	scanf("%d%d",&m,&n);
	int arr[m];
	int temp,num=1;
	scanf("%d",&arr[0]);
	int i;
	int l=1;
	for(i=1;i<n;i++){
		scanf("%d",&temp);
		int flag=1;
		for(int j=0;j<l;j++){
			if(temp==arr[j]){
				flag=0;
				break;
			}
		}
		if(flag){
			 arr[l]=temp;
			 num++;
			 l++;
		}
		if(num==m) break;
	}
	if(num==m)
	{
	int h=0;
	for(int k=0;k<n-i-1;k++){
		scanf("%d",&temp);
		int _flag=1;
		for(int t=0;t<m;t++){
			if(temp==arr[t]){
				_flag=0;
				break; 
			}
		}
		if(_flag){
			arr[h]=temp;
			num++;
			h++;
			if(h==m) h=0; 
		}
	}
    }
	printf("%d",num);
	return 0;
}
