#include <stdio.h> 

union uarea
{
    char    c_data;
    short   s_data;
    int     i_data;
    long    l_data;
};

int main()
{
    union uarea u;
    char* a1 = &(u.c_data);
    short* a2 = &(u.s_data);
    int* a3 = &(u.i_data);
    long* a4 = &(u.l_data);
    return 0;
}