import csv
from itertools import combinations
from collections import defaultdict
import os

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 输入文件路径
input_path = os.path.join(project_root, 'data', 'Groceries.csv')
# 输出文件路径
output_path = os.path.join(project_root, 'output/1-Apriori')

# 加载交易数据
def load_transactions(filename):
    transactions = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            # 解析每一行的商品列表
            items = [item.strip() for item in row[1].strip('{}').split(',')]
            transactions.append(items)
    return transactions

# 计算候选项集的支持度
def compute_support(transactions, candidate):
    count = 0
    for transaction in transactions:
        # 如果候选项集是交易的子集，则计数加一
        if set(candidate).issubset(set(transaction)):
            count += 1
    return count

# Apriori算法实现
def apriori(transactions, min_support, max_k=3):
    N = len(transactions)  # 总交易数
    min_support_count = min_support * N  # 最小支持度计数

    # 生成频繁1项集
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[item] += 1
    frequent = {1: {frozenset([item]): count for item, count in item_counts.items() if count >= min_support_count}}

    # 生成频繁k项集
    for k in range(2, max_k + 1):
        prev_frequent = frequent[k-1]  # 上一轮的频繁项集
        candidates = generate_candidates(prev_frequent, k)  # 生成候选项集
        current_frequent = {}
        for candidate in candidates:
            count = compute_support(transactions, candidate)
            if count >= min_support_count:
                current_frequent[frozenset(candidate)] = count
        frequent[k] = current_frequent
    return frequent

# 生成候选项集
def generate_candidates(prev_frequent, k):
    candidates = []
    prev_itemsets = list(prev_frequent.keys())  # 将字典键转换为列表

    if k == 2:
        # 通过组合单个项生成候选项集
        items = sorted({item for itemset in prev_itemsets for item in itemset})
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                candidates.append(frozenset([items[i], items[j]]))
    else:
        # 通过连接(k-1)项集生成候选项集
        for i in range(len(prev_itemsets)):
            for j in range(i + 1, len(prev_itemsets)):
                itemset1 = prev_itemsets[i]
                itemset2 = prev_itemsets[j]
                # 求并集
                union = itemset1.union(itemset2)
                if len(union) == k:
                    # 剪枝：如果(k-1)子集不频繁，则跳过
                    valid = True
                    for subset in combinations(union, k-1):
                        if frozenset(subset) not in prev_frequent:
                            valid = False
                            break
                    if valid:
                        candidates.append(union)
    return candidates

# 生成关联规则
def generate_rules(frequent, min_confidence, N):
    rules = []
    for k in frequent:
        if k == 1:
            continue  # 跳过频繁1项集
        for itemset, count in frequent[k].items():
            items = list(itemset)
            for i in range(1, len(items)):
                for antecedent in combinations(items, i):
                    antecedent = frozenset(antecedent)
                    consequent = frozenset(items) - antecedent
                    if antecedent in frequent[len(antecedent)]:
                        support_antecedent = frequent[len(antecedent)][antecedent]
                        confidence = count / support_antecedent  # 计算置信度
                        if confidence >= min_confidence:
                            rules.append((antecedent, consequent, count/N, confidence))
    return rules

# 保存结果到文件
def save_results(frequent, rules, N):
    for k in frequent:
        path = os.path.join(output_path, f'L{k}.csv')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['itemset', 'support'])
            for itemset, count in frequent[k].items():
                writer.writerow([','.join(itemset), count/N])
            f.write("Total Frequent Itemsets: " + str(len(frequent[k])) + "\n")
    
    path = os.path.join(output_path, 'rules.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rule', 'confidence'])
        for ant, cons, _, conf in rules:
            rule = f"{','.join(ant)} --> {','.join(cons)}"
            writer.writerow([rule, conf])
        f.write("Total Rules: " + str(len(rules)) + "\n")

# 主函数
def main():
    transactions = load_transactions(input_path)  # 加载交易数据
    frequent = apriori(transactions, 0.005, 3)  # 运行Apriori算法
    rules = generate_rules(frequent, 0.5, len(transactions))  # 生成关联规则
    save_results(frequent, rules, len(transactions))  # 保存结果
    print("Frequent Itemsets Counts:", {k: len(v) for k, v in frequent.items()})
    print("Total Rules:", len(rules))

if __name__ == '__main__':
    main()