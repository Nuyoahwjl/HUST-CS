/*----------------------------display----------------------------*/

#include "SAT.hpp"

/*
 @ 函数名称: DisPlay
 @ 函数功能: 交互界面
 @ 返回值: void
 */
void DisPlay()
{
    bool *value = NULL; // 存储文字的真值
    // clauseList cL = NULL;
    CNF cnf = (CNF)malloc(sizeof(cnfNode));
    cnf->root = NULL;
    char fileName[100]; // 文件名
    PrintMenu();        // 打印菜单
    int op = 1;
    while (op)
    {
        printf("\n|--------------------------------------------|\n");
        printf("|--------Please Choose Your Operation--------|\n");
        printf("|--------------------------------------------|\n\n");
        printf("               Your choice: ");
        scanf("%d", &op);
        system("cls"); // 每次进来清屏
        PrintMenu();   // 打印菜单
        switch (op)
        {
        case 1:
        {
            if (cnf->root != NULL) // 如果已经打开了CNF文件
            {
                printf(" The CNF has been read.\n");
                printf(" Do you want to read another? (1/0): ");
                int choice;
                scanf("%d", &choice);
                if (choice == 0)
                    break;
                else // 重新读取
                {
                    DestroyCnf(cnf->root); // 销毁当前解析的CNF
                }
            }
            printf(" Please input the file name: ");
            scanf("%s", fileName);
            if (ReadFile(cnf, fileName) == OK) // 读取文件并解析CNF
                printf(" Read successfully.\n");
            else
                printf(" Read failed.\n");
            break;
        }
        case 2:
        {
            if (cnf->root == NULL) // 如果没有打开CNF文件
                printf(" You haven't open the CNF file.\n");
            else
                PrintCnf(cnf); // 打印CNF文件
            break;
        }
        case 3:
        {
            if (cnf->root == NULL) // 如果没有打开CNF文件
            {
                printf(" You haven't open the CNF file.\n");
                break;
            }
            else
            {
                CNF newCnf = (CNF)malloc(sizeof(cnfNode));
                newCnf->root = CopyCnf(cnf->root); // 复制CNF
                newCnf->boolCount = cnf->boolCount;
                newCnf->clauseCount = cnf->clauseCount;
                value = (bool *)malloc(sizeof(bool) * (cnf->boolCount + 1));
                for (int i = 1; i <= cnf->boolCount; i++)
                    value[i] = TRUE;                    // 初始化，均赋为TRUE
                LARGE_INTEGER frequency, frequency_;    // 计时器频率
                LARGE_INTEGER start, start_, end, end_; // 设置时间变量
                double time, time_;
                // 未优化的时间
                QueryPerformanceFrequency(&frequency);
                QueryPerformanceCounter(&start); // 计时开始;
                int result = DPLL(newCnf, value, 1);
                QueryPerformanceCounter(&end);                                       // 结束
                time = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart; // 计算运行时间
                // 输出SAT结果
                if (result == OK) // SAT
                {
                    printf(" SAT\n\n");
                    // 输出文字的真值
                    for (int i = 1; i <= cnf->boolCount; i++)
                    {
                        if (value[i] == TRUE)
                            printf(" %-4d: TRUE\n", i);
                        else
                            printf(" %-4d: FALSE\n", i);
                    }
                }
                else // UNSAT
                    printf(" UNSAT\n");
                // 输出优化前的时间
                printf("\n Time: %lf ms(not optimized)\n", time * 1000);
                // 是否优化
                int ch;
                printf("\n Do you want to optimize the algorithm? (1/0): ");
                scanf("%d", &ch);
                if (ch == 0)
                    time_ = 0;
                else
                {
                    DestroyCnf(newCnf->root);          // 销毁未优化的CNF
                    newCnf->root = CopyCnf(cnf->root); // 复制CNF
                    QueryPerformanceFrequency(&frequency_);
                    QueryPerformanceCounter(&start_); // 计时开始;
                    DPLL(newCnf, value, 3);
                    QueryPerformanceCounter(&end_);                                          // 结束
                    time_ = (double)(end_.QuadPart - start_.QuadPart) / frequency_.QuadPart; // 计算运行时间
                    printf("\n Time: %lf ms(optimized)\n", time_ * 1000);
                }
                // 是否保存
                printf("\n Save the result to file? (1/0): ");
                int choice;
                scanf("%d", &choice);
                printf("\n");
                if (choice == 1)
                {
                    // 保存求解结果
                    if (SaveResult(result, time, time_, value, fileName, cnf->boolCount))
                        printf(" Save successfully.\n");
                    else
                        printf(" Save failed.\n");
                }
            }
            break;
        }
        case 4:
        {
            X_Sudoku();  // X数独界面
            PrintMenu(); // 跳转回来时重新打印菜单
            break;
        }
        case 0:
        {
            printf(" Exit successfully.\n");
            return; // 退出
        }
        default:
        {
            printf(" Invalid input.\n"); // 无效输入
            break;
        }
        }
    }
    if (cnf->root != NULL)
        DestroyCnf(cnf->root); // 退出时销毁CNF
    free(cnf);
    return;
}

/*
 @ 函数名称: PrintMenu
 @ 函数功能: 打印菜单
 @ 返回值: void
 */
void PrintMenu()
{
    printf("|================Menu for SAT================|\n");
    printf("|--------------------------------------------|\n");
    printf("|            1. Open the CNF file            |\n");
    printf("|     2. Traverse and output each clause     |\n");
    printf("|   3. Solve using DPLL and save the result  |\n");
    printf("|              4. X_Sudoku game              |\n");
    printf("|                 0.  EXIT                   |\n");
    printf("|============================================|\n\n");
}
