#include <stdio.h>
#include <stdlib.h>

int copy_array(int *array, int count)
{
    int i;
    /* 在堆区申请一块内存 */
    int* myarray = (int*)malloc(count * sizeof(int));     	
    if (myarray == NULL)         	
        return -1;   		
    for (i = 0; i < count; i++)         	
        myarray[i] = array[i];     	
    return count;  
}

int main()
{
    int size = (1 << 30) + 1;

    int a[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int count = copy_array(a, size);
}

