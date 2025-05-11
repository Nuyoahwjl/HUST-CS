from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

# 可视化训练过程的函数
def visualize_history(train_loss, train_acc, test_loss, test_acc):
    """
    绘制训练和测试的损失及准确率随时间变化的曲线图。
    参数:
    - train_loss: 训练集的损失列表
    - train_acc: 训练集的准确率列表
    - test_loss: 测试集的损失列表
    - test_acc: 测试集的准确率列表
    """
    plt.figure(figsize=(12, 5))  # 设置画布大小
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, color='tab:green', label='Train Loss')  # 训练损失曲线
    plt.plot(test_loss, color='tab:red', label='Test Loss')  # 测试损失曲线
    plt.title('Loss History')  # 图标题
    plt.xlabel('Epoch')  # x轴标签
    plt.ylabel('Loss')  # y轴标签
    plt.xticks(np.arange(0, len(train_loss)+1, 5))  # 设置x轴刻度
    plt.legend()  # 显示图例
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, color='tab:green', label='Train Accuracy')  # 训练准确率曲线
    plt.plot(test_acc, color='tab:red', label='Test Accuracy')  # 测试准确率曲线
    plt.title('Accuracy History')  # 图标题
    plt.xlabel('Epoch')  # x轴标签
    plt.ylabel('Accuracy')  # y轴标签
    plt.xticks(np.arange(0, len(train_loss)+1, 5))  # 设置x轴刻度
    plt.legend()  # 显示图例
    plt.tight_layout()  # 调整子图间距
    plt.show()  # 显示图像
    plt.close()  # 关闭图像窗口

# 绘制混淆矩阵的函数
def plot_confusion_matrix(model, test_loader):
    """
    绘制模型在测试集上的混淆矩阵。
    参数:
    - model: 已训练的PyTorch模型
    - test_loader: 测试集的数据加载器
    """
    model.eval()  # 设置模型为评估模式
    all_preds = []  # 存储所有预测值
    all_labels = []  # 存储所有真实标签
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():  # 禁用梯度计算
        for images, labels in test_loader:
            images = images.to(device)  # 将图像数据移动到设备
            labels = labels.to(device)  # 将标签数据移动到设备
            outputs = model(images)  # 获取模型输出
            _, preds = torch.max(outputs, 1)  # 获取预测类别
            all_preds.extend(preds.cpu().numpy())  # 将预测值移动到CPU并存储
            all_labels.extend(labels.cpu().numpy())  # 将真实标签移动到CPU并存储
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))  # 设置画布大小
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # 绘制热力图
    plt.xlabel('Predicted')  # x轴标签
    plt.ylabel('True')  # y轴标签
    plt.title('Confusion Matrix')  # 图标题
    plt.show()  # 显示图像
    plt.close()  # 关闭图像窗口


if __name__ == "__main__":
    train_loss = np.random.rand(20)
    train_acc = np.random.rand(20)
    test_loss = np.random.rand(20)
    test_acc = np.random.rand(20)
    visualize_history(train_loss, train_acc, test_loss, test_acc)