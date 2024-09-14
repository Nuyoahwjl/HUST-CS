/*----------------------------X-Sudoku----------------------------*/

#include "SAT.hpp"

/*
 @ 函数名称: X_Sudoku
 @ 函数功能: X数独交互界面
 @ 返回值: void
 */
void X_Sudoku()
{
    system("cls");
    PrintMenu_X();
    int num;                           // 提示数的个数
    bool isFixed[SIZE + 1][SIZE + 1];  // 记录是否为提示数字
    int board[SIZE + 1][SIZE + 1];     // 生成的初始数独
    int newBoard[SIZE + 1][SIZE + 1];  // 用来玩的数独
    int newBoard2[SIZE + 1][SIZE + 1]; // 保存答案的数独
    bool value[SIZE * SIZE * SIZE + 1];// 记录DPLL的结果
    for (int i = 1; i <= SIZE * SIZE * SIZE; i++)
        value[i] = FALSE;
    int op = 1;   // 操作
    int flag = 0; // 是否生成数独
    while (op)
    {
        printf("\n|********************************************|\n");
        printf("|--------Please Choose Your Operation--------|\n");
        printf("|********************************************|\n\n");
        printf("               Your choice: ");
        scanf("%d", &op);
        system("cls");
        PrintMenu_X();
        switch (op)
        {
        case 1:
        {
            printf(" Please enter the number of prompts(>=18): ");
            scanf("%d", &num);
            while (num < 18 || num > 81) // 提示数的个数必须大于等于18小于等于81
            {
                printf(" Invalid input, please enter again: ");
                scanf("%d", &num);
            }
            if (Generate_Sudoku(board, newBoard, newBoard2, isFixed, num, value))
            {
                printf(" Generate successfully.\n");
                flag = 1; // 生成成功
            }
            else
                printf(" Generate failed.\n");
            break;
        }
        case 2:
        {
            if (flag)
            {
                Play_Sudoku(newBoard, isFixed);
                PrintMenu_X(); // 每次玩完跳转回来重新打印菜单
            }
            else
                printf(" Please generate the X_Sudoku first.\n");
            break;
        }
        case 3:
        {
            if (flag)
            {
                printf(" Original X_Sudoku:\n");
                Print_Sudoku(board); // 打印原始数独
                printf("\n");
                if (Slove(newBoard2, value)) // 求解数独
                {
                    printf(" Reference answer:\n");
                    Print_Sudoku(newBoard2); // 打印答案
                }
                // WriteToFile(newBoard2, num,"X_Sudoku.cnf");
                // int value[SIZE * SIZE * SIZE + 1];
                // for (int i = 1; i <= SIZE * SIZE * SIZE; i++)
                //     value[i] = FALSE;
                // if (Slove(newBoard2, value))
                // {
                //     printf(" Reference answer:\n");
                //     Print_Sudoku(newBoard2);
                // }
                else
                    printf(" No answer.\n"); // 无解
            }
            else
                printf(" Please generate the X_Sudoku first.\n");
            break;
        }
        case 0:
        {
            system("cls"); // 退出时清屏
            break;
        }
        default:
        {
            printf(" Invalid input.\n");
            break;
        }
        }
    }
}

/*
 @ 函数名称: PrintMenu_X
 @ 函数功能: 打印X数独菜单
 @ 返回值: void
 */
void PrintMenu_X()
{
    printf("|**************Menu for X_Sudoku*************|\n");
    printf("|--------------------------------------------|\n");
    printf("|            1. Generate a X_Sudoku          |\n");
    printf("|            2. Play the X_Sudoku            |\n");
    printf("|            3. Reference answer             |\n");
    printf("|                 0. EXIT                    |\n");
    printf("|********************************************|\n\n");
}

/*
 @ 函数名称: Generate_Sudoku
 @ 接受参数: int[][],bool[][],int
 @ 函数功能: 生成数独
 @ 返回值: status
 */
status Generate_Sudoku(int board[SIZE + 1][SIZE + 1], int newBoard[SIZE + 1][SIZE + 1], int newBoard2[SIZE + 1][SIZE + 1], bool isFixed[SIZE + 1][SIZE + 1], int num, bool value[SIZE * SIZE * SIZE + 1])
{
    char name[100] = "X_Sudoku.cnf"; // 文件名
START:
    srand(time(NULL));
    // 初始化棋盘
    for (int i = 1; i <= SIZE; i++)
        for (int j = 1; j <= SIZE; j++)
        {
            board[i][j] = 0;
            newBoard[i][j] = 0;
            newBoard2[i][j] = 0;
            isFixed[i][j] = TRUE; // 默认全为为提示数字
        }
    int n[SIZE]; // 1-9的数组
    for (int i = 0; i < SIZE; i++)
        n[i] = i + 1;
    Shuffle(n, SIZE); // 打乱数组
    int index = 0;
    for (int i = 1; i <= SIZE; i++) // 对角线填入1-9
    {
        board[i][i] = n[index];
        newBoard[i][i] = n[index];
        newBoard2[i][i] = n[index];
        index++;
    }
    for (int i = 1; i <= SIZE; i += 3)
    {
        Fill_Box(board, newBoard, newBoard2, i, i); // 主对角线3个填充3x3的宫格
    }
    WriteToFile(board, 27, name); // 将数独约束条件写入文件
    CNF p = (CNF)malloc(sizeof(cnfNode));
    p->root = NULL;
    ReadFile(p, name); // 读取文件并解析CNF
    for (int i = 1; i <= SIZE * SIZE * SIZE; i++)
        value[i] = FALSE;
    if (DPLL(p, value, 3) == ERROR) // 求解数独
        goto START;
    // 将DPLL的结果填入数独
    for (int i = 1; i <= SIZE * SIZE * SIZE + 1; i++)
    {
        if (value[i] == TRUE)
        {
            int row = (i - 1) / (SIZE * SIZE) + 1;
            int col = (i - 1) / SIZE % SIZE + 1;
            int v = (i - 1) % SIZE + 1;
            board[row][col] = v;
            newBoard[row][col] = v;
            newBoard2[row][col] = v;
        }
    }
    // 挖洞,剩下num个提示数
    int remove = 81 - num;
    int single = remove / 9; // single一定小于等于7
    int res = remove - 9 * single;
    // int c[SIZE]={9-single};
    for (int row = 1; row <= 9; row++) // 每行挖single个
    {
        int s = single;
        while (s)
        {
            int col = rand() % SIZE + 1;
            if (board[row][col] != 0) // 没有被挖
            {
                board[row][col] = 0;
                newBoard[row][col] = 0;
                newBoard2[row][col] = 0;
                isFixed[row][col] = FALSE;
                s--;
            }
        }
    }
    while (res) // 挖剩下的
    {
        int row = rand() % SIZE + 1;
        // while(c[row-1]<=2)
        // row = rand() % SIZE + 1;
        int col = rand() % SIZE + 1;
        if (board[row][col] != 0)
        {
            board[row][col] = 0;
            newBoard[row][col] = 0;
            newBoard2[row][col] = 0;
            isFixed[row][col] = FALSE;
            res--;
            // c[row-1]--;
        }
    }
    return OK;
}

/*
 @ 函数名称: Is_Valid
 @ 接受参数: int[][],int,int,int
 @ 函数功能: 判断board[row][col]是否可以填入v
 @ 返回值: status
 */
status Is_Valid(int board[SIZE + 1][SIZE + 1], int row, int col, int v)
{
    // 检查行和列
    for (int i = 1; i <= SIZE; i++)
    {
        if (board[row][i] == v || board[i][col] == v) // 行或列有重复
            return FALSE;
    }
    // 检查3x3宫格
    int startRow = (row - 1) / 3 * 3 + 1; // 宫格的起始行
    int startCol = (col - 1) / 3 * 3 + 1; // 宫格的起始列
    for (int i = startRow; i < startRow + 3; i++)
    {
        for (int j = startCol; j < startCol + 3; j++)
        {
            if (board[i][j] == v) // 宫格内有重复
                return FALSE;
        }
    }
    // 检查主对角线
    if (row == col)
    {
        for (int i = 1; i <= SIZE; i++)
        {
            if (board[i][i] == v)
                return FALSE;
        }
    }
    // 检查副对角线
    if (row + col == SIZE + 1)
    {
        for (int i = 1; i <= SIZE; i++)
        {
            if (board[i][SIZE - i + 1] == v)
                return FALSE;
        }
    }
    return TRUE;
}

/*
 @ 函数名称: Print_Sudoku
 @ 接受参数: int[][]
 @ 函数功能: 打印数独
 @ 返回值: void
 */
void Print_Sudoku(int board[SIZE + 1][SIZE + 1])
{
    for (int i = 1; i <= SIZE; i++)
    {
        for (int j = 1; j <= SIZE; j++)
        {
            if (board[i][j] == 0) // 未填入
                printf(" .");
            else // 已填入
                printf("%2d", board[i][j]);
            if (j % 3 == 0 && j != SIZE) // 每3列打印一个竖线
                printf(" |");
        }
        printf("\n");
        if (i % 3 == 0 && i != SIZE) // 每3行打印一个横线
            printf("-------+-------+-------\n");
    }
}

/*
 @ 函数名称: Play_Sudoku
 @ 接受参数: int[][], bool[][]
 @ 函数功能: 玩数独的交互界面
 @ 返回值: void
 */
void Play_Sudoku(int board[SIZE + 1][SIZE + 1], bool isFixed[SIZE + 1][SIZE + 1])
{
    system("cls");       // 清屏
    Print_Sudoku(board); // 打印初始数独
    printf("\n");
    while (1)
    {
        int row, col, v;
        printf(" Please enter the row, col and value(0 to EXIT): ");
        scanf("%d", &row);
        if (row == 0) // 退出
        {
            system("cls");
            return;
        }
        scanf("%d%d", &col, &v);
        if (row < 1 || row > SIZE || col < 1 || col > SIZE || v < 1 || v > SIZE) // 输入不合法
        {
            printf(" Invalid input.\n");
            continue;
        }
        if (isFixed[row][col]) // 是提示数
        {
            printf(" This is a fixed number.\n");
            continue;
        }
        if (!Is_Valid(board, row, col, v)) // 不符合数独规则
        {
            printf(" Wrong answer.\n");
            continue;
        }
        else // 符合数独规则
        {
            board[row][col] = v;
            system("cls");
            Print_Sudoku(board); // 打印新数独
            printf("\n");
        }
    }
}

/*
 @ 函数名称: WriteToFile
 @ 接受参数: int[][],int
 @ 函数功能: 将数独约束条件写入文件
 @ 返回值: status
 */
status WriteToFile(int board[SIZE + 1][SIZE + 1], int num, char name[])
{
    FILE *fp;
    if (fopen_s(&fp, name, "w"))
    {
        printf(" Fail!\n");
        return ERROR;
    }
    fprintf(fp, "c %s\n", name);
    fprintf(fp, "p cnf 729 %d\n", num + 12654); // 12654是数独的约束条件
    // 数字ijk表示第i行第j列的数字是k
    // 用公式(i-1)*81+(j-1)*9+k将每个变元映射到1-729的变元上
    /*提示数约束(写在前面,便于单子句规则进行)*/
    for (int i = 1; i <= SIZE; i++)
    {
        for (int j = 1; j <= SIZE; j++)
        {
            if (board[i][j] != 0)
                fprintf(fp, "%d 0\n", (i - 1) * SIZE * SIZE + (j - 1) * SIZE + board[i][j]);
        }
    }
    /*每个格子的约束*/
    // 每个格子必须填入一个数字
    for (int i = 1; i <= SIZE; i++)
    {
        for (int j = 1; j <= SIZE; j++)
        {
            for (int k = 1; k <= SIZE; k++)
            {
                fprintf(fp, "%d ", (i - 1) * SIZE * SIZE + (j - 1) * SIZE + k);
            }
            fprintf(fp, "0\n");
        }
    }
    // 每个格子不能填入两个数字
    for (int i = 1; i <= SIZE; i++)
    {
        for (int j = 1; j <= SIZE; j++)
        {
            for (int k = 1; k <= SIZE; k++)
            {
                for (int l = k + 1; l <= SIZE; l++)
                {
                    fprintf(fp, "%d %d 0\n", 0 - ((i - 1) * SIZE * SIZE + (j - 1) * SIZE + k), 0 - ((i - 1) * SIZE * SIZE + (j - 1) * SIZE + l));
                }
            }
        }
    }
    /*行约束*/
    // 每一行必须填入1-9
    for (int i = 1; i <= SIZE; i++)
    {
        for (int j = 1; j <= SIZE; j++)
        {
            for (int k = 1; k <= SIZE; k++)
            {
                fprintf(fp, "%d ", (i - 1) * SIZE * SIZE + (k - 1) * SIZE + j);
            }
            fprintf(fp, "0\n");
        }
    }
    // 每一行不能填入两个相同的数字
    for (int i = 1; i <= SIZE; i++)
    {
        for (int j = 1; j <= SIZE; j++)
        {
            for (int k = 1; k <= SIZE; k++)
            {
                for (int l = k + 1; l <= SIZE; l++)
                {
                    fprintf(fp, "%d %d 0\n", 0 - ((i - 1) * SIZE * SIZE + (k - 1) * SIZE + j), 0 - ((i - 1) * SIZE * SIZE + (l - 1) * SIZE + j));
                }
            }
        }
    }
    /*列约束*/
    // 每一列必须填入1-9
    for (int i = 1; i <= SIZE; i++)
    {
        for (int j = 1; j <= SIZE; j++)
        {
            for (int k = 1; k <= SIZE; k++)
            {
                fprintf(fp, "%d ", (k - 1) * SIZE * SIZE + (i - 1) * SIZE + j);
            }
            fprintf(fp, "0\n");
        }
    }
    // 每一列不能填入两个相同的数字
    for (int i = 1; i <= SIZE; i++)
    {
        for (int j = 1; j <= SIZE; j++)
        {
            for (int k = 1; k <= SIZE; k++)
            {
                for (int l = k + 1; l <= SIZE; l++)
                {
                    fprintf(fp, "%d %d 0\n", 0 - ((k - 1) * SIZE * SIZE + (i - 1) * SIZE + j), 0 - ((l - 1) * SIZE * SIZE + (i - 1) * SIZE + j));
                }
            }
        }
    }
    /*3x3宫格约束*/
    // 每个3x3宫格必须填入1-9
    for (int i = 1; i <= SIZE; i += 3)
    {
        for (int j = 1; j <= SIZE; j += 3)
        {
            for (int k = 1; k <= SIZE; k++)
            {
                for (int l = 0; l < 3; l++)
                {
                    for (int m = 0; m < 3; m++)
                    {
                        fprintf(fp, "%d ", ((i + l - 1) * SIZE * SIZE + (j + m - 1) * SIZE + k));
                    }
                }
                fprintf(fp, "0\n");
            }
        }
    }
    // 每个3x3宫格不能填入两个相同的数字
    for (int i = 1; i <= SIZE; i += 3)
    {
        for (int j = 1; j <= SIZE; j += 3)
        {
            for (int k = 1; k <= SIZE; k++)
            {
                for (int l = 0; l < 3; l++)
                {
                    for (int m = 0; m < 3; m++)
                    {
                        for (int n = k + 1; n <= SIZE; n++)
                        {
                            fprintf(fp, "%d %d 0\n", 0 - ((i + l - 1) * SIZE * SIZE + (j + m - 1) * SIZE + k), 0 - ((i + l - 1) * SIZE * SIZE + (j + m - 1) * SIZE + n));
                        }
                    }
                }
            }
        }
    }
    /*对角线约束*/
    // 主对角线必须填入1-9
    for (int i = 1; i <= SIZE; i++)
    {
        for (int j = 1; j <= SIZE; j++)
        {
            fprintf(fp, "%d ", (j - 1) * SIZE * SIZE + (j - 1) * SIZE + i);
        }
        fprintf(fp, "0\n");
    }
    // 主对角线不能填入两个相同的数字
    for (int i = 1; i <= SIZE; i++)
    {
        for (int j = 1; j <= SIZE; j++)
        {
            for (int k = j + 1; k <= SIZE; k++)
            {
                fprintf(fp, "%d %d 0\n", 0 - ((j - 1) * SIZE * SIZE + (j - 1) * SIZE + i), 0 - ((k - 1) * SIZE * SIZE + (k - 1) * SIZE + i));
            }
        }
    }
    // 副对角线必须填入1-9
    for (int i = 1; i <= SIZE; i++)
    {
        for (int j = 1; j <= SIZE; j++)
        {
            fprintf(fp, "%d ", (j - 1) * SIZE * SIZE + (SIZE - j + 1) * SIZE + i);
        }
        fprintf(fp, "0\n");
    }
    // 副对角线不能填入两个相同的数字
    for (int i = 1; i <= SIZE; i++)
    {
        for (int j = 1; j <= SIZE; j++)
        {
            for (int k = j + 1; k <= SIZE; k++)
            {
                fprintf(fp, "%d %d 0\n", 0 - ((j - 1) * SIZE * SIZE + (SIZE - j + 1) * SIZE + i), 0 - ((k - 1) * SIZE * SIZE + (SIZE - k + 1) * SIZE + i));
            }
        }
    }
    fclose(fp);
    return OK;
}

/*
 @ 函数名称: Slove
 @ 接受参数: int[][],int[]
 @ 函数功能: DPLL求解数独
 @ 返回值: status
 */
status Slove(int board[SIZE + 1][SIZE + 1], bool value[SIZE * SIZE * SIZE + 1])
{
    // clauseList p=NULL;
    // char name[100]="X_Sudoku.cnf";
    // if(ReadFile(p,name)==OK)
    // if(DPLL(p,value)==ERROR)
    //     return ERROR;
    // 利用公式(i-1)*81+(j-1)*9+k反解
    for (int i = 1; i <= SIZE * SIZE * SIZE + 1; i++)
    {
        if (value[i] == TRUE)
        {
            int row = (i - 1) / (SIZE * SIZE) + 1;
            int col = (i - 1) / SIZE % SIZE + 1;
            int v = (i - 1) % SIZE + 1;
            board[row][col] = v;
        }
    }
    return OK;
}

/*
 @ 函数名称: Fill_Box
 @ 接受参数: int[][],int[][],int[][],int,int
 @ 函数功能: 填充3x3的宫格
 @ 返回值: status
 */
status Fill_Box(int board[SIZE + 1][SIZE + 1], int newBoard[SIZE + 1][SIZE + 1], int newBoard2[SIZE + 1][SIZE + 1], int rowStart, int colStart)
{
    int n[SIZE - 3];       // 除了对角线以外的6个格子
    bool flag[SIZE] = {0}; // 标记1-9是否已经填入
    for (int i = 0; i < 3; i++)
    {
        flag[board[rowStart + i][colStart + i] - 1] = 1; // 对角线的3个格子上的数字标记为已填入
    }
    int index = 0;
    for (int i = 0; i < SIZE; i++)
    {
        if (!flag[i])
            n[index++] = i + 1;
    }
    Shuffle(n, 6); // 打乱数组
    index = 0;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            if (board[rowStart + i][colStart + j] == 0) // 没有填入
            {
                board[rowStart + i][colStart + j] = n[index];
                newBoard[rowStart + i][colStart + j] = n[index];
                newBoard2[rowStart + i][colStart + j] = n[index];
                index++;
            }
        }
    }
    return OK;
}

/*
 @ 函数名称: Shuffle
 @ 接受参数: int[],int
 @ 函数功能: 洗牌算法,打乱数组顺序
 @ 返回值: void
 */
void Shuffle(int arr[], int n)
{
    srand(time(NULL)); // 用时间做种子
    // 每次从后面的数中随机选一个数与前面的数交换
    for (int i = n - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

// status Generate_Sudoku(int board[SIZE + 1][SIZE + 1], int newBoard[SIZE + 1][SIZE + 1], int newBoard2[SIZE+1][SIZE+1],bool isFixed[SIZE + 1][SIZE + 1], int num)
// {
//     srand(time(NULL));
//     // 初始化棋盘
//     for (int i = 1; i <= SIZE; i++)
//         for (int j = 1; j <= SIZE; j++)
//         {
//             board[i][j] = 0;
//             newBoard[i][j] = 0;
//             newBoard2[i][j] = 0;
//             isFixed[i][j] = FALSE;
//         }
//     // 生成数独
//     while (num)
//     {
//         int row = rand() % SIZE + 1;
//         int col = rand() % SIZE + 1;
//         if (board[row][col] == 0)
//         {
//             int v = rand() % SIZE + 1;
//             if (Is_Valid(board, row, col, v))
//             {
//                 board[row][col] = v;
//                 newBoard[row][col] = v;
//                 newBoard2[row][col] = v;
//                 isFixed[row][col] = TRUE;
//                 num--;
//             }
//         }
//     }
//     return OK;
// }
