import numpy as np
import struct
import gzip
import matplotlib.pyplot as plt
import os
from prettytable import PrettyTable

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)

def load_mnist(path, kind='train'):
    """
    加载 MNIST 数据集
    参数:
        path: 数据集所在路径
        kind: 数据集类型 ('train' 或 't10k')
    返回:
        images: 图像数据 (numpy 数组)
        labels: 标签数据 (numpy 数组)
    """
    # 构造标签和图像文件路径
    labels_path = f'{path}/{kind}-labels-idx1-ubyte.gz'
    images_path = f'{path}/{kind}-images-idx3-ubyte.gz'
    # 读取标签文件
    with gzip.open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))  # 读取文件头信息
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)  # 读取标签数据
    # 读取图像文件
    with gzip.open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))  # 读取文件头信息
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)  # 读取图像数据并重塑为二维数组
    return images, labels

def show_image(train_images, train_labels, test_images, test_labels):
    """
    显示数据集信息和部分训练图像
    参数:
        train_images: 训练集图像数据
        train_labels: 训练集标签数据
        test_images: 测试集图像数据
        test_labels: 测试集标签数据
    """
    # 创建表格显示数据集信息
    table = PrettyTable()
    table.field_names = ["Dataset", "Images Shape", "Labels Shape"]
    table.add_row(["Train", train_images.shape, train_labels.shape])
    table.add_row(["Test ", test_images.shape, test_labels.shape])
    print(table)
    # 定义类别名称
    class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # 创建一个 10x10 英寸的画布
    plt.figure(figsize=(10, 10))
    for i in range(0, 30):  # 显示前 30 张训练图像
        # 创建子图（6 行 5 列，索引从 1 开始）
        plt.subplot(6, 5, i + 1)
        # 隐藏刻度
        plt.xticks([])
        plt.yticks([])
        # 关闭网格线
        plt.grid(False)
        # 显示图像
        plt.imshow(train_images[i].reshape(28, 28))  # 将图像数据重塑为 28x28
        # 设置标题为对应的类别名称
        plt.title(class_names[train_labels[i]])
    # 设置整体标题
    plt.suptitle('Train Images (First 30)', fontsize=16, y=0.95)
    plt.show()

if __name__ == "__main__":
    # 数据集目录
    data_dir = os.path.join(project_root, 'data')
    # 加载训练集和测试集
    train_images, train_labels = load_mnist(data_dir, kind='train')
    test_images, test_labels = load_mnist(data_dir, kind='t10k')
    # 显示数据集信息和部分训练图像
    show_image(train_images, train_labels, test_images, test_labels)
