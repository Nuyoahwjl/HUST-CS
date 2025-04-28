// 二进制快速幂算法
#include <iostream>
#include <string>
#include <vector>
using namespace std;

// 将整数转换为二进制字符串
string intToBinary(long long n) {
    if (n == 0) return "0"; // 特殊情况处理
    string binary;
    while (n > 0) {
        binary = (n % 2 == 0 ? "0" : "1") + binary; // 取最低位
        n = n / 2; // 右移一位
    }
    return binary;
}

int main() {
    long long a, b, p;
    cin >> a >> b >> p;

    // 处理 b = 0 的情况
    if (b == 0) {
        cout << a << "^" << b << " mod " << p << "=" << 1 % p;
        return 0;
    }

    // 将 b 转换为二进制
    string binary = intToBinary(b);
    long long len = binary.length();

    // 初始化 temp 数组
    vector<long long> temp(len);
    temp[0] = a % p;          // a^1 mod p
    for (int i = 1; i < len; i++) {
        temp[i] = (temp[i - 1] * temp[i - 1]) % p; // a^{2^i} mod p
    }

    // 计算 a^b mod p
    long long ans = 1;
    for (int i = 0; i < len; i++) {
        if (binary[i] == '1') {
            ans = (ans * temp[len - i - 1]) % p;
        }
    }

    // 输出结果
    cout << a << "^" << b << " mod " << p << "=" << ans;
    return 0;
}


// #include <iostream>
// using namespace std;
// long long Pow(long long a,long long n,long long p);
// int main()
// {
//     long long a, b, p;
//     cin >> a >> b >> p;
//     cout << a << "^" << b << " mod " << p << "=" << Pow(a,b,p);
//     return 0;
// }
// long long Pow(long long a,long long n,long long p)
// {
//     if(n==0)    return 1;
//     if(n==1)    return a%p;
//     long long c=Pow(a,n/2,p);
//     if(n%2==0)  return (c%p)*(c%p)%p;
//     return (c%p)*(c%p)*a%p;
// }