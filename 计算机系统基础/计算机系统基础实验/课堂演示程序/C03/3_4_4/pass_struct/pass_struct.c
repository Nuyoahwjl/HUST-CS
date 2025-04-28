#include <stdio.h> 

typedef struct 
{
    int     id;
    char    name[10];
} stu_info;

void reg_stu_v(stu_info s)
{
    s.id = 101;
}

void reg_stu_a(stu_info* s)
{
    s->id = 201;
}

int main()
{
    stu_info stu = {-1, "jack"};
    reg_stu_v(stu);
    printf("%d\n", stu.id);
    reg_stu_a(&stu);
    printf("%d\n", stu.id);
    return 0;
}