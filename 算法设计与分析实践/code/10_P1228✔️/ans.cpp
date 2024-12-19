#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cmath>

using namespace std;

// 函数声明，slove函数用于递归填补迷宫
void slove(int k, int x, int y, int a1, int b1, int a2, int b2);

int main()
{
    int k, x, y;
    // 读取输入的k值和公主所在的坐标(x, y)
    cin >> k >> x >> y;
    // 调用slove函数开始填补迷宫
    slove(k, x, y, 1, 1, pow(2, k), pow(2, k));
    return 0;
}

// slove函数定义，k表示当前迷宫的大小为2^k x 2^k，(x, y)为公主所在的坐标
// (a1, b1)和(a2, b2)表示当前迷宫的左上角和右下角的坐标
void slove(int k, int x, int y, int a1, int b1, int a2, int b2)
{
    // 基本情况，当k为1时，直接输出结果
    if (k == 1)
    {
        // 根据公主所在的位置输出对应的地毯形状和位置
        if (x == a1 && y == b1)
            cout << a2 << ' ' << b2 << ' ' << 1 << endl;
        if (x == a1 && y == b2)
            cout << a2 << ' ' << b1 << ' ' << 2 << endl;
        if (x == a2 && y == b1)
            cout << a1 << ' ' << b2 << ' ' << 3 << endl;
        if (x == a2 && y == b2)
            cout << a1 << ' ' << b1 << ' ' << 4 << endl;
        return;
    }

    // 计算当前迷宫的中点坐标
    int midx = (a1 + a2) / 2;
    int midy = (b1 + b2) / 2;

    // 判断公主所在的象限，并递归填补其他象限
    if (x <= midx && y <= midy) // 如果公主在左上象限
    {
        // 输出地毯的右下角坐标和形状编号
        cout << midx + 1 << ' ' << midy + 1 << ' ' << 1 << endl;
        // 递归填补左上象限
        slove(k - 1, x, y, a1, b1, midx, midy);
        // 递归填补右上象限
        slove(k - 1, midx, midy + 1, a1, midy + 1, midx, b2);
        // 递归填补左下象限
        slove(k - 1, midx + 1, midy, midx + 1, b1, a2, midy);
        // 递归填补右下象限
        slove(k - 1, midx + 1, midy + 1, midx + 1, midy + 1, a2, b2);
    }
    if (x <= midx && y > midy) // 如果公主在右上象限
    {
        // 输出地毯的左下角坐标和形状编号
        cout << midx + 1 << ' ' << midy << ' ' << 2 << endl;
        // 递归填补左上象限
        slove(k - 1, midx, midy, a1, b1, midx, midy);
        // 递归填补右上象限
        slove(k - 1, x, y, a1, midy + 1, midx, b2);
        // 递归填补左下象限
        slove(k - 1, midx + 1, midy, midx + 1, b1, a2, midy);
        // 递归填补右下象限
        slove(k - 1, midx + 1, midy + 1, midx + 1, midy + 1, a2, b2);
    }
    if (x > midx && y <= midy) // 如果公主在左下象限
    {
        // 输出地毯的右上角坐标和形状编号
        cout << midx << ' ' << midy + 1 << ' ' << 3 << endl;
        // 递归填补左上象限
        slove(k - 1, midx, midy, a1, b1, midx, midy);
        // 递归填补右上象限
        slove(k - 1, midx, midy + 1, a1, midy + 1, midx, b2);
        // 递归填补左下象限
        slove(k - 1, x, y, midx + 1, b1, a2, midy);
        // 递归填补右下象限
        slove(k - 1, midx + 1, midy + 1, midx + 1, midy + 1, a2, b2);
    }
    if (x > midx && y > midy) // 如果公主在右下象限
    {
        // 输出地毯的左上角坐标和形状编号
        cout << midx << ' ' << midy << ' ' << 4 << endl;
        // 递归填补左上象限
        slove(k - 1, midx, midy, a1, b1, midx, midy);
        // 递归填补右上象限
        slove(k - 1, midx, midy + 1, a1, midy + 1, midx, b2);
        // 递归填补左下象限
        slove(k - 1, midx + 1, midy, midx + 1, b1, a2, midy);
        // 递归填补右下象限
        slove(k - 1, x, y, midx + 1, midy + 1, a2, b2);
    }
}