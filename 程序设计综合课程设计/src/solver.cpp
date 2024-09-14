/*----------------------------solver----------------------------*/

#include "SAT.hpp"

/*
 @ 函数名称: IsUnitClause
 @ 接受参数: literalList
 @ 函数功能: 判断是否为单子句
 @ 返回值: status
 */
status IsUnitClause(literalList l)
{
	if (l != NULL && l->next == NULL) // 只有一个文字
		return TRUE;
	else
		return FALSE;
}

/*
 @ 函数名称: FindUnitClause
 @ 接受参数: clauseList
 @ 函数功能: 找到单子句并返回该文字
 @ 返回值: int
 */
int FindUnitClause(clauseList cL)
{
	clauseList p = cL;
	while (p)
	{
		if (IsUnitClause(p->head))	 // 是单子句
			return p->head->literal; // 返回该文字
		p = p->next;
	}
	return 0;
}

/*
 @ 函数名称: DestroyClause
 @ 接受参数: clauseList &
 @ 函数功能: 销毁子句
 @ 返回值: status
 */
status DestroyClause(clauseList &cL)
{
	literalList p = cL->head;
	while (p)
	{
		literalList temp = p;
		p = p->next; // 指向下一个文字
		free(temp);	 // 释放文字
	}
	free(cL);  // 释放子句
	cL = NULL; // cL指向NULL
	return OK;
}

/*
 @ 函数名称: Simplify
 @ 接受参数: clauseList &, int
 @ 函数功能: 根据选择的文字化简
 @ 返回值: void
 */
void Simplify(clauseList &cL, int literal)
{
	clauseList pre = NULL, p = cL; // pre指向前一个子句
	while (p != NULL)
	{
		bool clauseDeleted = false;			  // 是否删除子句
		literalList lpre = NULL, q = p->head; // lpre指向前一个文字
		while (q != NULL)
		{
			if (q->literal == literal) // 删除该子句
			{
				if (pre == NULL) // 删除的是第一个子句
					cL = p->next;
				else // 删除的不是第一个子句
					pre->next = p->next;
				DestroyClause(p);					// 销毁该子句
				p = (pre == NULL) ? cL : pre->next; // 指向下一个子句
				clauseDeleted = true;				// 子句已删除
				break;
			}
			else if (q->literal == -literal) // 删除该文字
			{
				if (lpre == NULL) // 删除的是第一个文字
					p->head = q->next;
				else // 删除的不是第一个文字
					lpre->next = q->next;
				free(q);								   // 释放该文字
				q = (lpre == NULL) ? p->head : lpre->next; // 指向下一个文字
			}
			else // 未删除
			{
				lpre = q;
				q = q->next;
			}
		}
		if (!clauseDeleted) // 子句未删除
		{
			pre = p;
			p = p->next;
		}
	}
}

/*
 @ 函数名称: CopyCnf
 @ 接受参数: clauseList
 @ 函数功能: 复制cnf
 @ 返回值: clauseList
 */
clauseList CopyCnf(clauseList cL)
{
	// 初始化新的CNF
	clauseList newCnf = (clauseList)malloc(sizeof(clauseNode));
	clauseList lpa, lpb;  // lpa指向新的子句,lpb指向旧的子句
	literalList tpa, tpb; // tpa指向新的文字,tpb指向旧的文字
	newCnf->head = (literalList)malloc(sizeof(literalNode));
	newCnf->next = NULL;
	newCnf->head->next = NULL;
	for (lpb = cL, lpa = newCnf; lpb != NULL; lpb = lpb->next, lpa = lpa->next)
	{
		for (tpb = lpb->head, tpa = lpa->head; tpb != NULL; tpb = tpb->next, tpa = tpa->next)
		{
			tpa->literal = tpb->literal;
			tpa->next = (literalList)malloc(sizeof(literalNode));
			tpa->next->next = NULL;
			if (tpb->next == NULL) // 旧的子句中的文字已经复制完
			{
				free(tpa->next);
				tpa->next = NULL;
			}
		}
		lpa->next = (clauseList)malloc(sizeof(clauseNode));
		lpa->next->head = (literalList)malloc(sizeof(literalNode));
		lpa->next->next = NULL;
		lpa->next->head->next = NULL;
		if (lpb->next == NULL) // 旧的CNF中的子句已经复制完
		{
			free(lpa->next->head);
			free(lpa->next);
			lpa->next = NULL;
		}
	}
	return newCnf;
}

/*
 @ 函数名称: ChooseLiteral_1(未优化)
 @ 接受参数: clauseList
 @ 函数功能: 选择文字(第一个文字)
 @ 返回值: int
 */
int ChooseLiteral_1(CNF cnf)
{
	return cnf->root->head->literal;
}

/*
 @ 函数名称: ChooseLiteral_2(优化)
 @ 接受参数: clauseList
 @ 函数功能: (没有单子句时的策略)选择文字(出现次数最多的文字)
 @ 返回值: int
 */
int ChooseLiteral_2(CNF cnf)
{
	clauseList lp = cnf->root;
	literalList dp;
	int *count, MaxWord, max; // count记录每个文字出现次数,MaxWord记录出现最多次数的文字
	count = (int *)malloc(sizeof(int) * (cnf->boolCount * 2 + 1));
	for (int i = 0; i <= cnf->boolCount * 2; i++)
		count[i] = 0; // 初始化
	// 计算子句中各文字出现次数
	for (lp = cnf->root; lp != NULL; lp = lp->next)
	{
		for (dp = lp->head; dp != NULL; dp = dp->next)
		{
			if (dp->literal > 0) // 正文字
				count[dp->literal]++;
			else
				count[cnf->boolCount - dp->literal]++; // 负文字
		}
	}
	max = 0;
	// 找到出现次数最多的正文字
	for (int i = 1; i <= cnf->boolCount; i++)
	{
		if (max < count[i])
		{
			max = count[i];
			MaxWord = i;
		}
	}
	if (max == 0)
	{
		// 若没有出现正文字,找到出现次数最多的负文字
		for (int i = cnf->boolCount + 1; i <= cnf->boolCount * 2; i++)
		{
			if (max < count[i])
			{
				max = count[i];
				MaxWord = cnf->boolCount - i;
			}
		}
	}
	free(count);
	return MaxWord;
}

/*
 @ 函数名称: ChooseLiteral_3(优化)
 @ 接受参数: clauseList
 @ 函数功能: 选择最短子句中出现次数最多的文字
 @ 返回值: int
*/
int ChooseLiteral_3(CNF cnf)
{
	clauseList p = cnf->root;
	int *count = (int *)calloc(cnf->boolCount * 2 + 1, sizeof(int));
	int minSize = INT_MAX; // 初始化为大于可能的最大子句长度
	int literal = 0;
	clauseList temp = NULL;
	// 遍历子句，找到最小子句并统计其文字
	while (p != NULL)
	{
		literalList q = p->head;
		int clauseSize = 0;
		while (q != NULL)
		{
			clauseSize++;
			q = q->next;
		}
		if (clauseSize < minSize)
		{
			minSize = clauseSize; // 更新最小子句大小
			temp = p;
		}
		p = p->next;
	}
	// 遍历子句，统计最小子句中各文字出现次数
	literalList q = temp->head;
	while (q != NULL)
	{
		count[q->literal + cnf->boolCount]++;
		q = q->next;
	}
	// 找到最频繁的文字
	int maxCount = 0;
	for (int i = 0; i < cnf->boolCount * 2 + 1; i++)
	{
		if (count[i] > maxCount)
		{
			maxCount = count[i];
			literal = i - cnf->boolCount;
		}
	}
	free(count);
	return literal;
}

/*
 @ 函数名称: Satisfy
 @ 接受参数: clauseList
 @ 函数功能: 是否满足
 @ 返回值: status
*/
status Satisfy(clauseList cL)
{
	if (cL == NULL)
		return OK;
	else
		return ERROR;
}

/*
 @ 函数名称: EmptyClause
 @ 接受参数: clauseList
 @ 函数功能: 是否有空子句
 @ 返回值: status
 */
status EmptyClause(clauseList cL)
{
	clauseList p = cL;
	while (p)
	{
		if (p->head == NULL) // 空子句，返回UNSAT
			return TRUE;
		p = p->next;
	}
	return FALSE;
}

/*
 @ 函数名称: DPLL
 @ 接受参数: clauseList,int[]
 @ 函数功能: DPLL算法求解SAT问题
 @ 返回值: status
 */
status DPLL(CNF cnf, bool value[], int flag)
{
	/*1.单子句规则*/
	int unitLiteral = FindUnitClause(cnf->root);
	while (unitLiteral != 0)
	{
		value[abs(unitLiteral)] = (unitLiteral > 0) ? TRUE : FALSE;
		Simplify(cnf->root, unitLiteral);
		// 终止条件
		if (Satisfy(cnf->root) == OK)
			return OK;
		if (EmptyClause(cnf->root) == TRUE)
			return ERROR;
		unitLiteral = FindUnitClause(cnf->root);
	}
	/*2.选择一个未赋值的文字*/
	int literal;
	if (flag == 1)
		literal = ChooseLiteral_1(cnf); // 未优化
	else if (flag == 2)
		literal = ChooseLiteral_2(cnf); // 优化
	else
		literal = ChooseLiteral_3(cnf);
	/*3.将该文字赋值为真，递归求解*/
	CNF newCnf = (CNF)malloc(sizeof(cnfNode));
	newCnf->root = CopyCnf(cnf->root); // 复制CNF
	newCnf->boolCount = cnf->boolCount;
	newCnf->clauseCount = cnf->clauseCount;
	clauseList p = (clauseList)malloc(sizeof(clauseNode));
	p->head = (literalList)malloc(sizeof(literalNode));
	p->head->literal = literal;
	p->head->next = NULL;
	p->next = newCnf->root;
	newCnf->root = p; // 插入到表头
	if (DPLL(newCnf, value, flag) == 1)
		return 1; // 在第一分支中搜索
	DestroyCnf(newCnf->root);
	/*4.将该文字赋值为假，递归求解*/
	// newCnf = CopyCnf(cL);
	// newCnf=cL;
	clauseList q = (clauseList)malloc(sizeof(clauseNode));
	q->head = (literalList)malloc(sizeof(literalNode));
	q->head->literal = -literal;
	q->head->next = NULL;
	q->next = cnf->root;
	cnf->root = q; // 插入到表头
	status re = DPLL(cnf, value, flag); // 回溯到执行分支策略的初态进入另一分支
	// DestroyCnf(cL);
	return re;
}

/*
 @ 函数名称: SaveResult
 @ 接受参数: int,double,int[],char[]
 @ 函数功能: 保存求解结果
 @ 返回值: status
 */
status SaveResult(int result, double time, double time_, bool value[], char fileName[], int boolCount)
{
	FILE *fp;
	char name[100];
	for (int i = 0; fileName[i] != '\0'; i++)
	{
		// 修改拓展名.res
		if (fileName[i] == '.' && fileName[i + 4] == '\0')
		{
			name[i] = '.';
			name[i + 1] = 'r';
			name[i + 2] = 'e';
			name[i + 3] = 's';
			name[i + 4] = '\0';
			break;
		}
		name[i] = fileName[i];
	}
	if (fopen_s(&fp, name, "w"))
	{
		printf(" Fail!\n");
		return ERROR;
	}
	fprintf(fp, "s %d", result); // 求解结果
	if (result == 1)
	{
		fprintf(fp, "\nv");
		// 保存解值
		for (int i = 1, cnt = 1; i <= boolCount; i++, cnt++)
		{
			if (value[i] == true)
				fprintf(fp, " %d", i);
			else
				fprintf(fp, " %d", -i);
		}
	}
	fprintf(fp, "\nt %lfms", time * 1000); // 运行时间/毫秒
	if (time_ != 0)
	{
		fprintf(fp, "\nt %lfms(optimized)", time_ * 1000);		  // 运行时间/毫秒
		double optimization_rate = ((time - time_) / time) * 100; // 优化率
		fprintf(fp, "\nOptimization Rate: %.2lf%%", optimization_rate);
	}
	fclose(fp);
	return OK;
}
