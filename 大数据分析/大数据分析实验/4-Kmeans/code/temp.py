import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import gaussian_kde
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 输入文件路径（数据文件）
input_path = os.path.join(project_root, 'data', 'normalizedwinedata.csv')
# 输出文件路径（用于保存结果）
output_path = os.path.join(project_root, 'output\output_temp')

# K-means算法实现
def kmeans(X, k=3, max_iter=100):
    """
    实现K-means聚类算法
    参数:
        X: 输入数据，形状为(n_samples, n_features)
        k: 聚类的数量
        max_iter: 最大迭代次数
    返回:
        labels: 每个样本的聚类标签
        centroids: 聚类中心
    """
    n_samples, n_features = X.shape
    np.random.seed(15)  # 固定随机种子以确保结果可重复
    # 随机初始化聚类中心
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    for _ in range(max_iter):
        # 计算每个样本到每个初始质心的距离平方
        distances = ((X[:, np.newaxis] - centroids) ** 2).sum(axis=2)
        # 为每个样本分配最近的质心标签
        labels = np.argmin(distances, axis=1)
        # 更新质心
        new_centroids = np.zeros_like(centroids) # 创建一个新的质心数组，初始为0
        for i in range(k):
            cluster_points = X[labels == i]  # 获取属于当前聚类的样本
            if len(cluster_points) == 0:  # 如果某个聚类为空，重新随机选择一个样本作为质心
                new_centroids[i] = X[np.random.choice(n_samples, 1)]
            else:
                new_centroids[i] = cluster_points.mean(axis=0)  # 计算当前聚类的均值作为新质心
        # 如果质心不再变化，则停止迭代
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

# 计算ACC（聚类准确率）
def calculate_acc(y_true, y_pred):
    """
    计算聚类的准确率（ACC）
    参数:
        y_true: 真实标签
        y_pred: 预测标签
    返回:
        acc: 准确率
    """
    # 计算混淆矩阵，cm[i, j]表示真实标签为i的样本被预测为j的数量
    cm = confusion_matrix(y_true, y_pred)
    # 使用匈牙利算法最大化匹配
    # row_ind为最佳分配的行索引，col_ind为最佳分配的列索引
    row_ind, col_ind = linear_sum_assignment(-cm)
    acc = cm[row_ind, col_ind].sum() / len(y_true)
    # 绘制混淆矩阵并保存
    os.makedirs(output_path, exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='viridis', xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    confusion_matrix_path = os.path.join(output_path, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close()
    return acc

if __name__ == "__main__":
    # 加载数据
    data = np.genfromtxt(input_path, delimiter=',')
    y_true = (data[:, 0] - 1).astype(int)  # 将标签转换为0-based索引
    X = data[:, 1:]  # 提取特征数据
    # 运行K-means算法
    labels_pred, centroids = kmeans(X, k=3)
    # 计算聚类准确率
    acc = calculate_acc(y_true, labels_pred)
    # 计算SSE（每个样本到其对应质心的距离平方和）
    # labels_pred是一个长度为n_samples的数组，表示每个样本的聚类标签
    # centroids[labels_pred]是一个形状为(n_samples, n_features)的数组，表示每个样本对应的质心
    diff = X - centroids[labels_pred]
    sse = np.sum(diff ** 2)
    print(f"Accuracy (ACC): {acc:.4f}")
    print(f"Sum of Squared Errors (SSE): {sse:.4f}")
    # 可视化聚类结果（随机选取两个维度）
    rng = np.random.default_rng()  # 创建一个独立的随机数生成器
    feature_indices = rng.choice(X.shape[1], 2, replace=False)
    plt.figure(figsize=(12, 8))
    colors = ['red', 'green', 'blue']  # 每个聚类的颜色
    markers = ['o', 's', 'D']  # 每个聚类的标记形状
    for i in range(3):
        cluster_data = X[labels_pred == i]  # 获取属于当前聚类的样本
        plt.scatter(cluster_data[:, feature_indices[0]], cluster_data[:, feature_indices[1]], 
                    c=colors[i], marker=markers[i], label=f'Cluster {i+1}', alpha=0.7, edgecolors='k')
        plt.scatter(centroids[i, feature_indices[0]], centroids[i, feature_indices[1]], 
                    c=colors[i], marker='X', s=250, linewidths=2, 
                    edgecolors='black', label=f'Centroid {i+1}')
    plt.xlabel(f'Feature {feature_indices[0]+1} (Normalized)', fontsize=12)  # X轴标签
    plt.ylabel(f'Feature {feature_indices[1]+1} (Normalized)', fontsize=12)  # Y轴标签
    plt.title(f'K-means Clustering on Wine Dataset (K=3)\nAccuracy (ACC): {acc:.4f}, SSE: {sse:.4f}', fontsize=14)
    plt.legend(fontsize=10, loc='best', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    # 保存聚类结果图像到output_path目录
    os.makedirs(output_path, exist_ok=True)
    clustering_result_path = os.path.join(output_path, 'kmeans.png')
    plt.savefig(clustering_result_path, dpi=300)
    plt.close()



