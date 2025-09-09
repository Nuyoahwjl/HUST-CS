#include <stdio.h>

int main()
{
    int result = 0;
    int a = 12;
    int c = 0;
    int b = 10;
    switch (a)
    {
        case 15:
            c = b &0x0f;
        case 10:
            result = c + 50;
            break;
        case 12:
        case 17:
            result = b + 50;
            break;
        case 14:
            result = b;
            break;
        default:
            result = a;        
    }
    printf("%d\n", result);
	return 0;
}
