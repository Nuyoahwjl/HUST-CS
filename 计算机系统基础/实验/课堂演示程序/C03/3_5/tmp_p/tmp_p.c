#include <stdio.h> 
#include <string.h>

char* f()
{
    char temp[20];
    strcpy(temp, "hello");
    return temp;
}

int main()
{
    char* p;
    char a[20];
    int  i=0;
    p = f();
    while (*(p + i) != 0) {
        a[i] = p[i];
        i++;
    }
    a[i] = 0;
    printf("%s \n", a);
    printf("%s \n", p);
    return 0;
}
