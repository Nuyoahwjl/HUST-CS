#include <stdio.h> 

typedef struct 
{
    int     id;
    char    name[10];
} stu_info;

stu_info get_stu()
{
    stu_info stu = {75, "Peter"};
    return stu;
}

int main()
{
    stu_info stu = get_stu();
    return 0;
}