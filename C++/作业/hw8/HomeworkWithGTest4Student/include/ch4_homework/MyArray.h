#pragma once
#include <iostream>
#include <string>
/*
    第4章 编程题1
    一维整型数组MyArry的定义如下，请实现相应的函数成员
*/
class MyArray
{
private:
    int size=0;           // 数组大小
    int *const p=nullptr; // 指向动态分配的内存，保存数组的内容
public:
    MyArray(int size = 10);      // 构造函数，参数size指定数组大小
    MyArray(const MyArray &old); // 拷贝构造函数，要求实现深拷贝
    MyArray &operator=(const MyArray &rhs); // 重载=，要求实现深拷贝
    MyArray(MyArray &&old) noexcept; // 移动拷贝
    MyArray &operator=(MyArray &&rhs) noexcept; // 移动=
    ~MyArray(); // 析构函数，要求能防止反复释放资源
    int length(); // 返回数组大小
    int &get(int index); // 返回下标为index的元素，不考虑越界

    // 一个对象是否为空，如果size或p有一个为0，则返回true
    bool isempty() const;
    // 比较二个MyArray对象是否相等。当二个MyArray对象都不是Empty，size一样，数组的内容完全
    //一样时，这两个MyArray对象才相等
    bool equals(const MyArray &other) const; 
    //将MyArray数组内容变成字符串。要求数组元素之间用空格分开，形如这样的格式"0 1 2 3 4 5"
    //如果MyArray对象是Empty，则返回string对象的内容为""
    std::string toString();
    //返回内部p指针，仅仅用于测试，不能用于任何其他地方
    const int* const getP();
};
