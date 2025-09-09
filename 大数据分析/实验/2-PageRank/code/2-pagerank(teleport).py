import numpy as np
import os
import time  # 导入time模块

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

input_path = project_root + '/data/sent_receive.csv'

if __name__ == '__main__':
    start_time = time.time()  # 记录程序开始时间

    with open(input_path, 'r') as f:
        edges = [line.strip('\n').split(',') for line in f.readlines()[1:]]  # 跳过第一行
    nodes = []
    for edge in edges:
        if edge[1] not in nodes:
            nodes.append(edge[1])
        if edge[2] not in nodes:
            nodes.append(edge[2])

    print(nodes) 

    N = len(nodes)
    L = len(edges)

    M = np.zeros([N, N])
    for edge in edges:
        start = nodes.index(edge[1])
        end = nodes.index(edge[2])
        M[end, start] = 1  # 初始化M矩阵

    for j in range(N):
        sum_of_col = sum(M[:, j])
        for i in range(N):
            if M[i, j]:
                M[i, j] /= sum_of_col  # 构造M矩阵,M矩阵每一列的和为1

    r = np.ones(N) / N
    next_r = np.zeros(N)
    e = 300000  # 误差初始化
    b = 0.8  # 阻尼系数
    k = 0  # 记录迭代次数

    iteration_start_time = time.time()  # 记录迭代开始时间

    while e > 10e-8:  # 开始迭代
        next_r = np.dot(M, r) * b + (1-b) / N * np.ones(N) # 迭代公式
        sum_of_col = sum(next_r)
        next_r = next_r / sum_of_col
        error_vector = next_r - r
        e = np.abs(error_vector).max()
        r = next_r
        k += 1

    iteration_end_time = time.time()  # 记录迭代结束时间

    output_path = project_root + '/output/2-res(teleport)'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as output_file:
        output_file.write("ID,PageRank\n")
        for i in range(N):
            output_file.write(f"{nodes[i]},{r[i]}\n")

    end_time = time.time()  # 记录程序结束时间

    print('迭代次数: %s' % str(k))
    print('迭代耗时: %.6f 秒' % (iteration_end_time - iteration_start_time))
    print('程序总耗时: %.6f 秒' % (end_time - start_time))
