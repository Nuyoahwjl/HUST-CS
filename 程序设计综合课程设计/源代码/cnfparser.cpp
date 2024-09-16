/*----------------------------cnfparser----------------------------*/

#include "SAT.hpp"
 
/*
 @ 函数名称: ReadFile
 @ 接受参数: clauseList &,char[]
 @ 函数功能: 用文件指针fp打开用户指定的文件，并读取文件内容保存到给定参数中
 @ 返回值: status
 */
status ReadFile(CNF &cnf, char fileName[])
{
    FILE *fp = fopen(fileName, "r");
    while (fp == NULL)
    {
        printf(" File not found, please input again: ");
        scanf("%s", fileName);
        fp = fopen(fileName, "r");
    }
    char ch;
    // 跳过注释
    while ((ch = getc(fp)) == 'c')
    {
        while ((ch = getc(fp)) != '\n')
            continue;
    }
    // 跳过p cnf
    getc(fp);
    getc(fp);
    getc(fp);
    getc(fp);
    fscanf(fp, "%d%d", &cnf->boolCount, &cnf->clauseCount); // 读取布尔变元个数和子句个数                              // 初始化布尔变元个数
    cnf->root = NULL;                                       // 初始化CNF
    clauseList lastClause = NULL;                           // 用于记录上一个一个子句
    for (int i = 0; i < cnf->clauseCount; i++)
    {
        // 读取子句
        clauseList newClause = (clauseList)malloc(sizeof(clauseNode));
        newClause->head = NULL;
        newClause->next = NULL;
        literalList lastLiteral = NULL; // 用于记录上一个文字
        int number;                     // 读取文字
        fscanf(fp, "%d", &number);
        while (number != 0)
        {
            literalList newLiteral = (literalList)malloc(sizeof(literalNode));
            newLiteral->literal = number;
            newLiteral->next = NULL;
            if (newClause->head == NULL) // 如果是第一个文字
                newClause->head = newLiteral;
            else
                lastLiteral->next = newLiteral;
            lastLiteral = newLiteral; // 更新上一个文字
            fscanf(fp, "%d", &number);
        }
        if (cnf->root == NULL) // 如果是第一个子句
            cnf->root = newClause;
        else
            lastClause->next = newClause;
        lastClause = newClause; // 更新上一个子句
    }
    fclose(fp);
    return OK;
}

/*
 @ 函数名称: DestroyCnf
 @ 接受参数: clauseList &
 @ 函数功能: 销毁给定的CNF文件
 @ 返回值: status
 */
status DestroyCnf(clauseList &cL)
{
    while (cL != NULL)
    {
        clauseList tempClause = cL;
        cL = cL->next;                    // 指向下一个子句
        literalList p = tempClause->head; // 指向当前子句的第一个文字
        while (p != NULL)
        {
            literalList tempLiteral = p;
            p = p->next;
            free(tempLiteral); // 释放文字
        }
        free(tempClause); // 释放子句
    }
    cL = NULL; // cL指向NULL
    return OK;
}

/*
 @ 函数名称: PrintCnf
 @ 接受参数: clauseList
 @ 函数功能: 打印给定的CNF文件
 @ 返回值: status
 */
status PrintCnf(CNF cnf)
{
    clauseList p = cnf->root;
    if (p == NULL) // 如果没有子句
    {
        printf(" No clauses.\n");
        return ERROR;
    }
    printf("  The CNF is:\n");
    printf("  boolCount:%d\n", cnf->boolCount);     // 打印布尔变元个数
    printf("  clauseCount:%d\n", cnf->clauseCount); // 打印子句个数
    while (p)
    {
        literalList q = p->head; // 指向子句的第一个文字
        printf("  ");
        while (q)
        {
            printf("%-5d", q->literal); // 打印文字
            q = q->next;
        }
        printf("0\n"); // 子句结束
        p = p->next;   // 指向下一个子句
    }
    return OK;
}
