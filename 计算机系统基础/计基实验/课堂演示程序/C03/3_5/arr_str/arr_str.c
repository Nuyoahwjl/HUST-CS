#include <stdio.h>

int main()
{
    char*   sa[5] = {"hello", "abc", "123", "yes", "no"};
    for (int i = 0; i < 5; i++)
    {
        printf("%s\n", sa[i]);
    }
    return 0;
}