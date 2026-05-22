/* 
 *trans.c - 矩阵转置B=A^T
 *每个转置函数都必须具有以下形式的原型：
 *void trans（int M，int N，int a[N][M]，int B[M][N]）；
 *通过计算，块大小为32字节的1KB直接映射缓存上的未命中数来计算转置函数。
 */ 
#include <stdio.h>
#include "cachelab.h"
int is_transpose(int M, int N, int A[N][M], int B[M][N]);
char transpose_submit_desc[] = "Transpose submission";  //请不要修改“Transpose_submission”


void transpose_submit(int M, int N, int A[N][M], int B[M][N]) {
    const int BLOCK = 8;
    const int HALF  = 4;

    int bi, bj, k;
    int a0, a1, a2, a3, a4, a5, a6, a7;

    /* 64x64：使用专门的 8x8 -> 4x4 分块交换策略，降低冲突缺失 */
    if (M == 64 && N == 64) {
        for (bj = 0; bj < N; bj += BLOCK) {
            for (bi = 0; bi < M; bi += BLOCK) {

                /* 
                 * 把当前 8x8 块的前 4 行读出来：
                 * - 左上 4x4 直接转置到 B 的左上
                 * - 右上 4x4 先“暂存”到 B 的右上
                 */
                for (k = 0; k < HALF; ++k) {
                    a0 = A[bj + k][bi + 0];
                    a1 = A[bj + k][bi + 1];
                    a2 = A[bj + k][bi + 2];
                    a3 = A[bj + k][bi + 3];
                    a4 = A[bj + k][bi + 4];
                    a5 = A[bj + k][bi + 5];
                    a6 = A[bj + k][bi + 6];
                    a7 = A[bj + k][bi + 7];

                    B[bi + 0][bj + k] = a0;
                    B[bi + 1][bj + k] = a1;
                    B[bi + 2][bj + k] = a2;
                    B[bi + 3][bj + k] = a3;

                    B[bi + 0][bj + 4 + k] = a4;
                    B[bi + 1][bj + 4 + k] = a5;
                    B[bi + 2][bj + 4 + k] = a6;
                    B[bi + 3][bj + 4 + k] = a7;
                }

                /*
                 * 处理“交换区”：
                 * - B 右上目前暂存的是 A 右上块的转置
                 * - 现在读 A 左下块，把它放到 B 右上（这是最终位置）
                 * - 同时把刚才暂存的内容搬到 B 左下（也是最终位置）
                 */
                for (k = 0; k < HALF; ++k) {
                    a0 = B[bi + k][bj + 4];
                    a1 = B[bi + k][bj + 5];
                    a2 = B[bi + k][bj + 6];
                    a3 = B[bi + k][bj + 7];

                    a4 = A[bj + 4][bi + k];
                    a5 = A[bj + 5][bi + k];
                    a6 = A[bj + 6][bi + k];
                    a7 = A[bj + 7][bi + k];

                    B[bi + k][bj + 4] = a4;
                    B[bi + k][bj + 5] = a5;
                    B[bi + k][bj + 6] = a6;
                    B[bi + k][bj + 7] = a7;

                    B[bi + 4 + k][bj + 0] = a0;
                    B[bi + 4 + k][bj + 1] = a1;
                    B[bi + 4 + k][bj + 2] = a2;
                    B[bi + 4 + k][bj + 3] = a3;
                }

                /* 右下 4x4 直接转置到 B 的右下 */
                for (k = 0; k < HALF; ++k) {
                    a4 = A[bj + 4 + k][bi + 4];
                    a5 = A[bj + 4 + k][bi + 5];
                    a6 = A[bj + 4 + k][bi + 6];
                    a7 = A[bj + 4 + k][bi + 7];

                    B[bi + 4][bj + 4 + k] = a4;
                    B[bi + 5][bj + 4 + k] = a5;
                    B[bi + 6][bj + 4 + k] = a6;
                    B[bi + 7][bj + 4 + k] = a7;
                }
            }
        }
        return;
    }

    /* 通用路径：按 8x8 分块转置 */
    for (bj = 0; bj + BLOCK <= N; bj += BLOCK) {
        for (bi = 0; bi + BLOCK <= M; bi += BLOCK) {
            for (k = 0; k < BLOCK; ++k) {
                a0 = A[bj + k][bi + 0];
                a1 = A[bj + k][bi + 1];
                a2 = A[bj + k][bi + 2];
                a3 = A[bj + k][bi + 3];
                a4 = A[bj + k][bi + 4];
                a5 = A[bj + k][bi + 5];
                a6 = A[bj + k][bi + 6];
                a7 = A[bj + k][bi + 7];

                B[bi + 0][bj + k] = a0;
                B[bi + 1][bj + k] = a1;
                B[bi + 2][bj + k] = a2;
                B[bi + 3][bj + k] = a3;
                B[bi + 4][bj + k] = a4;
                B[bi + 5][bj + k] = a5;
                B[bi + 6][bj + k] = a6;
                B[bi + 7][bj + k] = a7;
            }
        }
    }

    /* 处理右边剩余列：M 不是 8 的倍数时 */
    for (int row = 0; row < N; ++row) {
        for (int col = (M / BLOCK) * BLOCK; col < M; ++col) {
            B[col][row] = A[row][col];
        }
    }

    /* 处理底部剩余行：N 不是 8 的倍数时 */
    for (int row = (N / BLOCK) * BLOCK; row < N; ++row) {
        for (int col = 0; col < (M / BLOCK) * BLOCK; ++col) {
            B[col][row] = A[row][col];
        }
    }
}

/* 
 * 我们在下面定义了一个简单的方法来帮助您开始，您可以根据下面的例子把上面值置补充完整。
 */ 

/* 
 * 简单的基线转置功能，未针对缓存进行优化。
 */
char trans_desc[] = "Simple row-wise scan transpose";
void trans(int M, int N, int A[N][M], int B[M][N])
{
    int i, j, tmp;

    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            tmp = A[i][j];
            B[j][i] = tmp;
        }
    }    

}

/*
 * registerFunctions-此函数向驱动程序注册转置函数。
 *在运行时，驱动程序将评估每个注册的函数并总结它们的性能。这是一种试验不同转置策略的简便方法。
 */
void registerFunctions()
{
    /* 注册解决方案函数  */
    registerTransFunction(transpose_submit, transpose_submit_desc); 

    /* 注册任何附加转置函数 */
    registerTransFunction(trans, trans_desc); 

}

/* 
 * is_transpose - 函数检查B是否是A的转置。在从转置函数返回之前，可以通过调用它来检查转置的正确性。
 */
int is_transpose(int M, int N, int A[N][M], int B[M][N])
{
    int i, j;

    for (i = 0; i < N; i++) {
        for (j = 0; j < M; ++j) {
            if (A[i][j] != B[j][i]) {
                return 0;
            }
        }
    }
    return 1;
}

