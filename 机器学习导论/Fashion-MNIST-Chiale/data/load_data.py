import numpy as np
import gzip
import matplotlib.pyplot as plt
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
# project_root = current_dir


def load_images(filename):
    """
    加载并预处理Fashion-MNIST图像数据
    参数:filename (str): 图像压缩文件路径
    返回:np.ndarray: 形状为(样本数, 28, 28, 1)的归一化图像数据,值域[0.0, 1.0]
    """
    with gzip.open(filename, 'rb') as f: 
        # 从文件读取数据并转换为numpy数组
        # offset=16: 跳过前16字节的头部信息（幻数+图像数量+行数+列数）
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # 数据预处理流程:
    # 1. 调整形状 -> (样本数, 通道数, 高度, 宽度)
    # 2. 转换为float32类型
    # 3. 归一化到0-1范围（原始像素值0-255）
    return data.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

def load_labels(filename):
    """
    加载并预处理Fashion-MNIST标签数据
    参数:filename (str): 标签压缩文件路径
    返回:np.ndarray: 形状为(样本数,)的整数标签数组,值域0-9
    """
    with gzip.open(filename, 'rb') as f: 
        # 从文件读取数据并转换为numpy数组
        # offset=8: 跳过前8字节的头部信息（幻数+标签数量）
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    # 将标签转换为int64类型
    return data.astype(np.int64)

def load_data():
    data_dir = os.path.join(project_root, 'data')  # 获取项目根目录下的data路径
    x_train = load_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    y_train = load_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    x_test = load_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    y_test = load_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))
    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()
    print('X_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)

    class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # 创建一个 10x10 英寸的画布
    plt.figure(figsize = (10,10))  
    for i in range(0, 30):      
        # 创建子图（6行5列，索引从1开始）      
        plt.subplot(6, 5, i+1)  
        # 隐藏刻度
        plt.xticks([])
        plt.yticks([])
        # 关闭网格线
        plt.grid(False)
        # 显示图像
        plt.imshow(x_train[i].reshape(28, 28))
        # 显示灰度图像
        # plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
        # 设置标题为对应的类别名称
        plt.title(class_names[y_train[i]])  