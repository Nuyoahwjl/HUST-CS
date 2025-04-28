#include <iostream>
#include <cstdio>
#include <algorithm>

using namespace std;

int height[10005]; // 用于存储每个位置的最大高度，假设位置范围为 0-10000

int main()
{
    int l, h, r; // 分别表示建筑的左坐标、高度和右坐标
    int left = 10005, right = 0; // 用于记录建筑群的最左和最右位置范围

    // 读取输入，直到文件结束
    while (scanf("%d%d%d", &l, &h, &r) != EOF)
    {
        // 更新建筑群的最左位置和最右位置
        left = min(left, l);
        right = max(right, r);

        // 遍历建筑的范围 [l, r)，将对应位置的高度更新为最大值
        for (int i = l; i < r; i++)
            height[i] = max(height[i], h);
    }

    h = 0; // 当前高度，初始为 0
    // 遍历建筑群的有效范围
    for (int i = left; i <= right; i++)
    {
        // 如果当前位置的高度发生变化，则说明是轮廓线的一个拐点
        if (height[i] != h)
        {
            h = height[i]; // 更新当前高度

            // 如果不是第一个拐点，输出一个空格用于分隔
            if (i != left)
                printf(" ");

            // 输出拐点的 x 坐标和对应的高度
            printf("%d %d", i, height[i]);
        }
    }

    return 0;
}
