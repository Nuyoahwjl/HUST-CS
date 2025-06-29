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
output_path = os.path.join(project_root, 'output/2-PCY')

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

# 实现 PCY 算法
def pcy(transactions, min_support, min_confidence, bucket_size=1000003):
    N = len(transactions)  # 总交易数
    min_support_count = min_support * N  # 最小支持度计数

    # 第一次扫描：统计单项和桶的计数
    item_counts = defaultdict(int)  # 单项计数
    bucket_counts = defaultdict(int)  # 桶计数
    for transaction in transactions:
        items = sorted(transaction)
        # 统计单项计数
        for item in items:
            item_counts[item] += 1
        # 统计桶计数
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                h = hash((items[i], items[j])) % bucket_size
                bucket_counts[h] += 1

    # 筛选频繁单项和频繁桶
    frequent_items = {item for item, count in item_counts.items() if count >= min_support_count}
    frequent_buckets = {h for h, count in bucket_counts.items() if count >= min_support_count}
    bit_vector = [1 if h in frequent_buckets else 0 for h in range(bucket_size)]  # 位向量表示频繁桶

    # 生成频繁 1 项集
    frequent = {1: {frozenset([item]): item_counts[item] for item in frequent_items}}

    # 第二次扫描：生成频繁 2 项集候选集
    candidates = []
    items_list = sorted(frequent_items)
    for i in range(len(items_list)):
        for j in range(i+1, len(items_list)):
            a, b = items_list[i], items_list[j]
            h = hash((a, b)) % bucket_size
            if bit_vector[h]:  # 检查桶是否频繁
                candidates.append(frozenset([a, b]))
    # 统计频繁 2 项集
    frequent_2 = {}
    for candidate in candidates:
        count = 0
        for transaction in transactions:
            if set(candidate).issubset(set(transaction)):
                count += 1
        if count >= min_support_count:
            frequent_2[frozenset(candidate)] = count
    frequent[2] = frequent_2

    # 生成频繁 3 项集候选集
    candidates_3 = []
    # for itemset1 in frequent_2:
    #     for itemset2 in frequent_2:
    frequent_2_items = list(frequent_2.keys())
    for i in range(len(frequent_2_items)):
        for j in range(i+1, len(frequent_2_items)):
            itemset1 = frequent_2_items[i]
            itemset2 = frequent_2_items[j]
            if len(itemset1.union(itemset2)) == 3:
                candidate = itemset1.union(itemset2)
                valid = True
                # 检查所有子集是否频繁
                for subset in combinations(candidate, 2):
                    if frozenset(subset) not in frequent_2:
                        valid = False
                        break
                if valid and candidate not in candidates_3:
                    candidates_3.append(candidate)
    # 统计频繁 3 项集
    frequent_3 = {}
    for candidate in candidates_3:
        count = 0
        for transaction in transactions:
            if set(candidate).issubset(set(transaction)):
                count += 1
        if count >= min_support_count:
            frequent_3[frozenset(candidate)] = count
    frequent[3] = frequent_3

    # 生成关联规则
    rules = []
    for k in [2, 3]:
        for itemset, count in frequent[k].items():
            items = list(itemset)
            for i in range(1, len(items)):
                for antecedent in combinations(items, i):
                    antecedent = frozenset(antecedent)
                    consequent = frozenset(items) - antecedent
                    if antecedent in frequent[len(antecedent)]:
                        support_antecedent = frequent[len(antecedent)][antecedent]
                        confidence = count / support_antecedent
                        if confidence >= min_confidence:
                            rules.append((antecedent, consequent, count/N, confidence))
    return frequent, rules, bit_vector

# 保存结果到文件
def save_results(frequent, rules, bit_vector, N):
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
    
    path = os.path.join(output_path, 'pcy_vector.txt')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(''.join(map(str, bit_vector)))

# 主函数
def main():
    transactions = load_transactions(input_path)  # 加载交易数据
    frequent, rules, bit_vector = pcy(transactions, 0.005, 0.5)  # 执行 PCY 算法
    save_results(frequent, rules, bit_vector, len(transactions))  # 保存结果
    print("Frequent Itemsets Counts:", {k: len(v) for k, v in frequent.items()})
    print("Total Rules:", len(rules))

if __name__ == '__main__':
    main()