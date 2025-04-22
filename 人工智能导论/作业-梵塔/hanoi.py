def hanoi(n, source, target, auxiliary):
    """
    解决汉诺塔问题的递归函数。
    参数：
    n         : 圆盘数量
    source    : 起始柱子
    target    : 目标柱子
    auxiliary : 辅助柱子
    """
    if n == 1:
        print(f"将圆盘 1 从 {source} 移动到 {target}")
        return
    # 将前 n-1 个圆盘从 source 移到 auxiliary，借助 target
    hanoi(n-1, source, auxiliary, target)
    # 将第 n 个圆盘从 source 移到 target
    print(f"将圆盘 {n} 从 {source} 移动到 {target}")
    # 将前 n-1 个圆盘从 auxiliary 移到 target，借助 source
    hanoi(n-1, auxiliary, target, source)

# 测试
n = 7  # 圆盘数量
hanoi(n, "A", "C", "B")