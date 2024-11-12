#include "../../include/ch4_homework/MyArray.hpp"
using namespace std;

// 构造函数
MyArray::MyArray(int size) : size(size), p(new int[size]) { }

// 拷贝构造函数
MyArray::MyArray(const MyArray &old) : size(old.size),p(new int[size]) {
    for (int i = 0; i < size; i++){
        p[i] = old.p[i];
    }
}

// 重载=
MyArray &MyArray::operator=(const MyArray &rhs) {
    if (this == &rhs)
        return *this;
    if(this->p)
        delete[] this->p;
    this->size = rhs.size;
    *(const_cast<int **>(&this->p))= new int[size];
    for (int i = 0; i < size; i++)
        p[i] = rhs.p[i];
    return *this;
}

// 移动拷贝
MyArray::MyArray(MyArray &&old) noexcept : size(old.size), p(old.p) {
    old.size = 0;
    *(const_cast<int **>(&old.p)) = nullptr;
}

// 移动赋值
MyArray &MyArray::operator=(MyArray &&rhs) noexcept {
    if (this == &rhs)
        return *this;
    if(this->p)
        delete[] this->p;
    this->size = rhs.size;
    *(const_cast<int **>(&this->p)) = rhs.p;
    rhs.size = 0;
    *(const_cast<int **>(&rhs.p)) = nullptr;
    return *this;
}

// 析构函数
MyArray::~MyArray() {
    if(p)
    {
        delete[] p;
        size = 0;
        *(const_cast<int **>(&p)) = nullptr;
    }
}

// 返回数组大小
int MyArray::length() {
    return this->size;
}

// 返回下标为index的元素，不考虑越界
int &MyArray::get(int index) {
    return p[index];
}

// 一个对象是否为空，如果size或p有一个为0，则返回true
bool MyArray::isempty() const {
    return (size == 0 || p == nullptr);
}

// 比较二个MyArray对象是否相等
bool MyArray::equals(const MyArray &other) const {
    if (this->isempty() || other.isempty())
        return false;
    if (this->size != other.size)
        return false;
    for (int i = 0; i < size; i++)
    {
        if (this->p[i] != other.p[i])
            return false;
    }
    return true;
}

// 将MyArray数组内容变成字符串
std::string MyArray::toString() {
    if (this->isempty())
        return "";
    string str;
    for (int i = 0; i < size; i++)
    {
        str += to_string(p[i]);
        if (i != size - 1)
            str += " ";
    }
    return str;
}

// 返回内部p指针，仅仅用于测试，不能用于任何其他地方
const int *const MyArray::getP() {
    return p;
}