#include "../../include/ch6_ch8_ch11_homework/QUEUE.h"
using namespace std;

QUEUE::QUEUE(int m) : max(m), size(0), elems(new int[m]) {}

QUEUE::QUEUE(const QUEUE &q) : max(q.max), size(q.size), elems(new int[q.max]) {
    for (int i = 0; i < size; i++)
        elems[i] = q.elems[i];
}

QUEUE::QUEUE(QUEUE &&q) noexcept : max(q.max), size(q.size), elems(q.elems) {
    q.size = 0;
    q.max =0 ;
    *(const_cast<int **>(&q.elems)) = nullptr;
}

QUEUE::operator int() const noexcept {
    return size;
}

int QUEUE::capacity() const noexcept {
    return max;
}

QUEUE& QUEUE::operator<<(int e) {
    if (size == max)
        throw string("QUEUE is full!");
    elems[size++] = e;
    return *this;
}

QUEUE& QUEUE::operator>>(int& e) {
    if (size == 0)
        throw string("QUEUE is empty!");
    e = elems[0];
    for (int i = 0; i < size - 1; i++)
        elems[i] = elems[i + 1];
    size--;
    return *this;
} 

QUEUE& QUEUE::operator=(const QUEUE &q) {
    if (this == &q)
        return *this;
    if (elems)
        delete[] elems;
    max = q.max;
    size = q.size;
    *(const_cast<int **>(&elems)) = new int[max];
    for (int i = 0; i < size; i++)
        elems[i] = q.elems[i];
    return *this;
}

QUEUE& QUEUE::operator=(QUEUE &&q) noexcept {
    if (this == &q)
        return *this;
    if (elems)
        delete[] elems;
    max = q.max;
    size = q.size;
    *(const_cast<int **>(&elems)) = q.elems;
    q.size = 0;
    q.max = 0;
    *(const_cast<int **>(&q.elems)) = nullptr;
    return *this;
}

string QUEUE::toString() {
    string res;
    for (int i = 0; i < size; i++)
    {
        res += to_string(elems[i]);
        if(i!=size-1)
            res+=' ';
    }   
    return res;
}

QUEUE::~QUEUE() {
    if (elems)
    {
        delete[] elems;
        size = 0;
        max = 0;
        *(const_cast<int **>(&elems)) = nullptr;
    }
}