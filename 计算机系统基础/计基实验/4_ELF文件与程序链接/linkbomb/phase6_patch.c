#include<stdio.h>
void f();
void (*myprint)() = f;
void f(){
    printf("The student ID you input is: U202315763\n");
}


