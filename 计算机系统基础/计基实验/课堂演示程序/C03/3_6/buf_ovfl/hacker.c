#include <stdio.h>
#include
void outputs(char* str)
{
    char buffer[16];
    strcpy(buffer, str);
    printf("in outputs %s \n", buffer);
}
void hacker(void)
{
    printf("being hacked\r\n");
}
int main(int argc, char* argv[])
{
    printf("argv[1] %s\r\n", argv[1]);
    printf("argv[20] %08x\r\n", *(int*)(argv[1]+20));
    printf("hacker() %08x\r\n", hacker);
    outputs(argv[1]);
    return 0;
}
