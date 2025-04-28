#include <stdio.h>
int main(int argc, char* argv[])
{
    char s[10];
    s[10] = -1;
    char* p = s;
    *(p + 13) = 40;

    return 0;
}