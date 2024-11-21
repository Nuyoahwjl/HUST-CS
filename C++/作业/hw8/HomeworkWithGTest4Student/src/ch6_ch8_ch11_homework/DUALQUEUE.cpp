#include "../../include/ch6_ch8_ch11_homework/DUALQUEUE.h"
using namespace std;

DUALQUEUE::DUALQUEUE(int m) : QUEUE(m), q(m) {}

DUALQUEUE::DUALQUEUE(const DUALQUEUE &s) : QUEUE(s), q(s.q) {}

DUALQUEUE::DUALQUEUE(DUALQUEUE &&s) noexcept : QUEUE(move(s)), q(move(s.q)) {}

int DUALQUEUE::capacity() const noexcept {
    return QUEUE::capacity() + q.capacity();
}

DUALQUEUE::operator int() const noexcept {
    return QUEUE::operator int() + int(q);
}

DUALQUEUE& DUALQUEUE::operator<<(int e) {
    if (int(*this) == capacity())
        throw string("DUALQUEUE is full!");
    if (int(*this) == QUEUE::capacity()) {
        while(QUEUE::operator int() > 0) {
            int temp;
            QUEUE::operator>>(temp);
            q << temp;
        }
    }
    QUEUE::operator<<(e);
    return *this;
}

DUALQUEUE& DUALQUEUE::operator>>(int &e) {
    if (int(*this) == 0)
        throw string("DUALQUEUE is empty!");
    if (int(q) == 0) {
        while(QUEUE::operator int() > 0) {
            int temp;
            QUEUE::operator>>(temp);
            q << temp;
        }
    }
    q >> e;
    return *this;
}

DUALQUEUE& DUALQUEUE::operator=(const DUALQUEUE &s) {
    if (this == &s)
        return *this;
    QUEUE::operator=(s);
    q = s.q;
    return *this;
}

DUALQUEUE& DUALQUEUE::operator=(DUALQUEUE &&s) noexcept {
    if (this == &s)
        return *this;
    QUEUE::operator=(move(s));
    q = move(s.q);
    return *this;
}

string DUALQUEUE::toString() {
    if(int(*this) == 0)
        return "";
    string res=q.toString();
    if(int(q) > 0)
        res += " ";
    res += QUEUE::toString();
    return res;
}

DUALQUEUE::~DUALQUEUE() noexcept {}