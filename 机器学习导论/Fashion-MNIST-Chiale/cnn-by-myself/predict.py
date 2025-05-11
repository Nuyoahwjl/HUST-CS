import numpy as np
from data_loader import load_mnist
from cnn_model import CNN
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def predict(test_images, test_labels):
    # 将测试图像的像素值归一化到[0, 1]范围
    test_images = test_images / 255.0
    model = CNN()  # 初始化CNN模型
    # 定义要加载的模型文件名称列表
    model_names = ['model_step1000.npz', 'model_step2000.npz', 'model_step3000.npz',
                   'model_step4000.npz', 'model_step5000.npz']
    batch_size = 1000  # 每次预测的批量大小
    total_samples = len(test_labels)  # 测试样本总数

    print("model                   accuracy") 
    print("================================")

    # 创建一个1行5列的子图，用于显示混淆矩阵
    fig, axs = plt.subplots(1, 5, figsize=(30, 6))  
    for idx, model_name in enumerate(model_names):
        # 加载模型文件
        model_path = os.path.join(current_dir, 'models', model_name)
        model.load_model(model_path)
        all_preds = []  # 存储所有预测结果
        # 按批量对测试数据进行预测
        for i in range(0, total_samples, batch_size):
            batch_images = test_images[i:i+batch_size]  # 获取当前批量的图像
            y_pred = model.forward(batch_images)  # 前向传播得到预测结果
            preds = np.argmax(y_pred, axis=1)  # 获取每个样本的预测类别
            all_preds.extend(preds)  # 将预测结果添加到列表中
        all_preds = np.array(all_preds)  # 转换为NumPy数组
        accuracy = np.mean(all_preds == test_labels)  # 计算预测准确率
        print(f'{model_name.ljust(24)} {accuracy:.4f}')  # 打印模型名称和准确率
        # 计算混淆矩阵
        cm = confusion_matrix(test_labels, all_preds)
        # 显示混淆矩阵
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axs[idx], xticks_rotation='horizontal', cmap='Reds', colorbar=False)
        axs[idx].set_title(model_name.replace('.npz', '')+"_confusion_matrices", fontsize=16)

    plt.tight_layout()  # 调整子图布局
    plt.show()  # 显示图像


if __name__ == '__main__':
    data_dir = os.path.join(project_root, 'data')
    test_images, test_labels = load_mnist(data_dir, kind='t10k')
    test_mask = np.random.choice(len(test_images), 100, replace=False)
    test_images = test_images[test_mask]
    test_labels = test_labels[test_mask]
    predict(test_images=test_images, test_labels=test_labels)
