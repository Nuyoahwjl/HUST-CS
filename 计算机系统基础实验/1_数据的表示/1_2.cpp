// #include <iostream>
// #include <limits>
// using namespace std;

// /* 返回 x 的绝对值 */
// int absVal(int x){
//     // 获取符号位：x为负数时mask为全1(-1),为正数时mask为0
//     int mask = x >> 31;
//     // x ^ mask 等价于 x 的取反或保持不变
//     // 若x<0,则|x| = ~x + 1
//     return (x ^ mask) - mask;
// }
// int absVal_standard(int x){
//     return (x < 0) ? -x : x;
// }

// /* 不使用负号实,现 -x */
// int my_negate(int x){
//     return ~x + 1;
// }

// int negate_standard(int x){
//     return -x;
// }

// /* 仅使用 ~ 和 |,实现 & */
// int bitAnd(int x, int y){
//     return ~(~x | ~y);
// }
// int bitAnd_standard(int x, int y){
//     return x & y;
// }

// /* 仅使用 ~ 和 &,实现 | */
// int bitOr(int x, int y){
//     return ~(~x & ~y);
// }
// int bitOr_standard(int x, int y){
//     return x | y;
// }

// /* 仅使用 ~ 和 &,实现 ^ */
// int bitXor(int x, int y){
//     return ~(x & y) & ~(~x & ~y);
// }
// int bitXor_standard(int x, int y){
//     return x ^ y;
// }

// /* 判断x是否为最大的正整数(7FFFFFFF)
// 只能使用 !、 ~、 &、 ^、 |、 + */
// int isTmax(int x){
//     // 计算 x + 1，并检查它是否为 0
//     int x_plus_1 = x + 1;
//     // 使用符号位判断，signX = x >> 31，如果 x 是正数，signX 为 0
//     int signX = x >> 31;
//     // 使用符号位判断，signSum = x_plus_1 >> 31，如果 x + 1 为负数，则 signSum 为 -1
//     int signSum = x_plus_1 >> 31;
//     // 如果 x + 1 为 0 且 x 是正数，返回 1；否则返回 0
//     return !(~(signX ^ signSum) & signX) & !!(x_plus_1);
// }
// int isTmax_standard(int x){
//     return x == 0x7FFFFFFF;
// }

// /*统计x的二进制表示中 1 的个数
// 只能使用,! ~ & ^ | + << >> ,运算次数不超过 40次 */
// int bitCount(int x){
//     int mask1 = 0x55 | (0x55 << 8);  // 01010101... 逐位统计
//     mask1 = mask1 | (mask1 << 16);
//     int mask2 = 0x33 | (0x33 << 8);  // 00110011... 逐位统计
//     mask2 = mask2 | (mask2 << 16);
//     int mask4 = 0x0F | (0x0F << 8);  // 00001111... 逐位统计
//     mask4 = mask4 | (mask4 << 16);
//     int mask8 = 0xFF | (0xFF << 16); // 前8位全1
//     int mask16 = 0xFF | (0xFF << 8); // 前16位全1
//     x = (x & mask1) + ((x >> 1) & mask1);  // 统计每两位内的1
//     x = (x & mask2) + ((x >> 2) & mask2);  // 统计每四位内的1
//     x = (x & mask4) + ((x >> 4) & mask4);  // 统计每八位内的1
//     x = (x & mask8) + ((x >> 8) & mask8);  // 统计每16位内的1
//     x = (x & mask16) + ((x >> 16) & mask16);  // 最终得出1的个数
//     return x;
// }
// int bitCount_standard(int x){
//     // Brian Kernighan算法
//     int cnt = 0;
//     while(x){
//         x &= x - 1; // 每次清除最低位的1
//         cnt++;
//     }
//     return cnt;
// }

// /* 产生从lowbit 到 highbit 全为1，其他位为0的数
// 只使用 ! ~ & ^ | + << >> ;运算次数不超过 16次 */
// int bitMask(int highbit, int lowbit){
//     unsigned int highMask = ~0U << lowbit;   // 从lowbit起全为1
//     unsigned int lowMask = ~0U << (highbit + 1);  // 从highbit之后全为0
//     return highMask & ~lowMask;   // 截取中间部分
// } 
// int bitMask_standard(int highbit, int lowbit){
//     int mask = 0;
//     for(int i = lowbit; i <= highbit; ++i){
//         mask |= 1 << i;
//     }
//     return mask;
// }

// /* 当x+y 会产生溢出时返回1,否则返回 0
// 仅使用 !、 ~、 &、 ^、 |、 +、 <<、 >>,运算次数不超过 20次 */
// int addOK(int x, int y){
//     // 符号位相同且结果的符号位与x的符号位不同
//     int sum = x + y;
//     int signX = (x >> 31)&1;
//     int signY = (y >> 31)&1;
//     int signSum = (sum >> 31)&1;
//     return !(~(signX ^ signY)) & (signX ^ signSum);
// }
// int addOK_standard(int x, int y){
//     return (x > 0 && y > 0 && x + y < 0) || (x < 0 && y < 0 && x + y > 0);
// }

// /* 将x的第n个字节与第m个字节交换,返回交换后的结果
// n、m的取值在 0~3之间 
// 仅使用 !、 ~、 &、 ^、 |、 +、 <<、 >>,运算次数不超过 25次 */
// int byteSwap(int x, int n, int m){
//     int n_shift = n << 3;  // 将字节编号转化为位编号
//     int m_shift = m << 3;
//     int n_byte = (x >> n_shift) & 0xFF;  // 提取n字节
//     int m_byte = (x >> m_shift) & 0xFF;  // 提取m字节
//     int mask = (0xFF << n_shift) | (0xFF << m_shift);  // 构造掩码去除两个字节
//     x = x & ~mask;  // 清空n和m字节
//     return x | (n_byte << m_shift) | (m_byte << n_shift);  // 交换字节后重新赋值
// }

// void printMenu(){
//     cout << "|============|" << endl;
//     cout << "| 1. absVal  |" << endl;
//     cout << "| 2. negate  |" << endl;
//     cout << "| 3. bitAnd  |" << endl;
//     cout << "| 4. bitOr   |" << endl;
//     cout << "| 5. bitXor  |" << endl;
//     cout << "| 6. isTmax  |" << endl;
//     cout << "| 7. bitCount|" << endl;
//     cout << "| 8. bitMask |" << endl;
//     cout << "| 9. addOK   |" << endl;
//     cout << "| 10.byteSwap|" << endl;
//     cout << "| 0. Exit    |" << endl;
//     cout << "|============|" << endl << endl;
// }

// int main()
// {
//     int op=1;
//     printMenu();
//     while(op){
//         cout<<"Input your option: ";
//         cin>>op;
//         // system("cls");
//         // printMenu();
//         switch(op){
//             case 1:
//             {
//                 int x;
//                 cout << "Input x: ";
//                 cin >> x;
//                 cout << "absVal: " << absVal(x) << " " << absVal_standard(x) << endl;
//                 break;
//             }
//             case 2:
//             {
//                 int x;
//                 cout << "Input x: ";
//                 cin >> x;
//                 cout << "negate: " << my_negate(x) << " " << negate_standard(x) << endl;
//                 break;
//             }
//             case 3:
//             {
//                 int x,y;
//                 cout << "Input x, y: ";
//                 cin >> x >> y;
//                 cout << "bitAnd: " << bitAnd(x, y) << " " << bitAnd_standard(x, y) << endl;
//                 break;
//             }
//             case 4:
//             {
//                 int x,y;
//                 cout << "Input x, y: ";
//                 cin >> x >> y;
//                 cout << "bitOr: " << bitOr(x, y) << " " << bitOr_standard(x, y) << endl;
//                 break;
//             }
//             case 5:
//             {
//                 int x,y;
//                 cout << "Input x, y: ";
//                 cin >> x >> y;
//                 cout << "bitXor: " << bitXor(x, y) << " " << bitXor_standard(x, y) << endl;
//                 break;
//             }
//             case 6:
//             {
//                 int x;
//                 cout << "Input x: ";
//                 cin >> x;
//                 cout << "isTmax: " << isTmax(x) << " " << isTmax_standard(x) << endl;
//                 break;
//             }
//             case 7:
//             {
//                 int x;
//                 cout << "Input x: ";
//                 cin >> x;
//                 cout << "bitCount: " << bitCount(x) << " " << bitCount_standard(x) << endl;
//                 break;
//             }
//             case 8:
//             {
//                 int n,m;
//                 cout << "Input highbit, lowbit: ";
//                 cin >> n >> m;
//                 cout << "bitMask: "<< bitMask(n, m) << " " << bitMask_standard(n, m) << endl;
//                 break;
//             }
//             case 9:
//             {
//                 int x,y;
//                 cout << "Input x, y: ";
//                 cin >> x >> y;
//                 cout << "addOK: " << addOK(x, y) << " " << addOK_standard(x, y) << endl;
//                 break;
//             }
//             case 10:
//             {
//                 int x,n,m; 
//                 cout << "Input x, n, m: ";
//                 cin >> x ;
//                 cin >> n >> m;
//                 cout << "byteSwap: " << byteSwap(x, n, m) << endl;
//                 break;
//             }
//             case 0:
//                 break;
//             default:
//                 cout << "Invalid input\n";
//         }
//     }
//     return 0;
// }


# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <assert.h>

int absVal(int x) {
    if(x >> 31) {
        return (~x) + 1;
    } else {
        return x;
    }
}

int negate(int x) {
    return (~x) + 1;
}

int bitAnd(int x, int y) {
    return ~((~x) | (~y));
}

int bitOr(int x, int y) {
    return ~((~x) & (~y));
}

int bitXor(int x, int y) {
    int v1 = bitAnd(~x, y), v2 = bitAnd(x, ~y);
    return bitOr(v1, v2);
}

int isTmax(int x) {
    return bitAnd(!((x + 1) ^ (~x)), !!(x + 1));
}

int bitCount(int xx) {
    unsigned int x = xx;
    unsigned int t1 = x & 0x55555555, t2 = x ^ t1;
    x = t1 + (t2 >> 1);
    t1 = x & 0x33333333; t2 = x ^ t1;
    x = t1 + (t2 >> 2);
    t1 = x & 0x0f0f0f0f; t2 = x ^ t1;
    x = t1 + (t2 >> 4);
    t1 = x & 0x00ff00ff; t2 = x ^ t1;
    x = t1 + (t2 >> 8);
    t1 = x & 0x0000ffff; t2 = x ^ t1;
    x = t1 + (t2 >> 16);
    return x;
}
    
int bitMask(int highbit, int lowbit) {
    if(!(highbit ^ 31)) {
        return 0xffffffff >> lowbit << lowbit;
    } else {
        return ((1u << highbit + 1) + 0xffffffff) >> lowbit << lowbit;
    }
}

int addOK(int x, int y) {
    int xh = (x >> 31) & 1, yh = (y >> 31) & 1, xo = x & 0x7fffffff, yo = y & 0x7fffffff, ad = ((xo + yo) >> 31) & 1;
    if(xh & yh & (!ad) | (!xh) & (!yh) & ad) return 1;
    else return 0;
}

int byteSwap(int x, int n, int m) {
    int u = n << 3, v = m << 3;
    int x1 = (x >> u) & 255, x2 = (x >> v) & 255;
    return x ^ (x1 << u) ^ (x2 << v) ^ (x1 << v) ^ (x2 << u);
}

int absVal_standard(int x)  { return (x < 0) ? -x : x;}

int netgate_standard(int x)  { return -x;}

int bitAnd_standard(int x, int y)  {return x & y;}

int bitOr_standard(int x, int y) {return x | y;}

int bitXor_standard(int x, int y) {return x ^ y;}

int isTmax_standard(int x) {return x == 0x7fffffff;}

int bitCount_standard(int x) {
    int ans = 0;
    for(int i = 0; i < 32; i++) ans += (x & 1), x >>= 1;
    return ans;
}

int bitMask_standard(int highbit, int lowbit) {
    unsigned int ans = 0;
    for(int i = lowbit; i <= highbit; i++) ans |= 1u << i;
    return ans;
}

int addOK_standard(int x, int y) {
    long long xl = x, yl = y, sum = xl + yl;
    if(sum > __INT_MAX__ || sum < (int)0x80000000) return 1;
    else return 0;
}

int byteSwap_standard(int x, int n, int m) {
    int tmp = x;
    char *p = (char *)&tmp;
    int a = *(p + n);
    int b = *(p + m);
    *(p + n) = b;
    *(p + m) = a;
    return tmp;
}

int myrand() {
    int x = rand();
    if(rand() % 2) x = -x;
    return x;
}

int main() {
    srand(time(0));

    puts("abs_test...");
    for(int i = 0; i < 10; i++) {
        int x = myrand();
        assert(absVal(x) == absVal_standard(x));
        printf("    %d-th passed\n", i);
    }
    puts("abs_test passed");

    puts("negate_test...");
    for(int i = 0; i < 10; i++) {
        int x = myrand();
        assert(negate(x) == netgate_standard(x));
        printf("    %d-th passed\n", i);
    }
    puts("negate_test passed");

    puts("and_test...");
    for(int i = 0; i < 10; i++) {
        int x = myrand(), y = myrand();
        assert(bitAnd(x, y) == bitAnd_standard(x, y));
        printf("    %d-th passed\n", i);
    }
    puts("and_test passed");

    puts("or_test...");
    for(int i = 0; i < 10; i++) {
        int x = myrand(), y = myrand();
        assert(bitOr(x, y) == bitOr_standard(x, y));
        printf("    %d-th passed\n", i);
    }
    puts("or_test passed");

    puts("xor_test...");
    for(int i = 0; i < 10; i++) {
        int x = myrand(), y = myrand();
        assert(bitXor(x, y) == bitXor_standard(x, y));
        printf("    %d-th passed\n", i);
    }
    puts("xor_test passed");

    puts("tmax_test...");
    for(int i = 0; i < 10; i++) {
        int x;
        if(i == 0) x = 0x7fffffff;
        else if(i == 1) x = -1;
        else x = myrand();
        assert(isTmax(x) == isTmax_standard(x));
        printf("    %d-th passed\n", i);
    }
    puts("tmax_test passed");

    puts("bitcount_test...");
    for(int i = 0; i < 10; i++) {
        int x = myrand();
        assert(bitCount(x) == bitCount_standard(x));
        printf("    %d-th passed\n", i);
    }
    puts("bitcount_test passed");

    puts("bitmask_test...");
    for(int i = 0; i < 10; i++) {
        int x = rand() % 32, y = rand() % 32;
        if(x < y) {
            int t;
            t = x; x = y; y = t;
        }
        if(i < 3) x = 31;
        assert(bitMask(x, y) == bitMask_standard(x, y));
        printf("    %d-th passed\n", i);
    }
    puts("bitmask_test passed");

    puts("addok_test...");
    for(int i = 0; i < 10; i++) {
        int x = myrand(), y = myrand();
        assert(addOK(x, y) == addOK_standard(x, y));
        printf("    %d-th passed\n", i);
    }
    puts("addok_test passed");

    puts("byteswap_test...");
    for(int i = 0; i < 10; i++) {
        int x = myrand(), n = rand() % 4, m = rand() % 4;
        assert(byteSwap(x, n, m) == byteSwap_standard(x, n, m));
        printf("    %d-th passed\n", i);
    }
    puts("byteswap_test passed");
    return 0;
}