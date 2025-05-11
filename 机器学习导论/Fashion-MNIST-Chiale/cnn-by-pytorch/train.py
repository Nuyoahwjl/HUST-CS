import torch
from torch.utils.data import DataLoader, TensorDataset
import logging
from predict import evaluate_model

# 数据预处理函数
def preprocess_data(train_images, train_labels, test_images, test_labels):
    """
    对输入的训练和测试数据进行预处理，包括归一化、转换为Tensor以及创建数据加载器。
    参数:
    - train_images: 训练集图像数据
    - train_labels: 训练集标签数据
    - test_images: 测试集图像数据
    - test_labels: 测试集标签数据
    返回:
    - train_loader: 训练数据加载器
    - test_loader: 测试数据加载器
    """
    # 归一化图像数据并转换为Tensor格式
    x_train = torch.tensor(train_images, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
    y_train = torch.tensor(train_labels, dtype=torch.long)
    x_test = torch.tensor(test_images, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
    y_test = torch.tensor(test_labels, dtype=torch.long)
    # 创建训练和测试数据集
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    return train_loader, test_loader


# 模型训练函数
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=20):
    """
    训练模型并在每个epoch后评估测试集性能，同时保存最佳模型。
    参数:
    - model: 待训练的神经网络模型
    - train_loader: 训练数据加载器
    - test_loader: 测试数据加载器
    - criterion: 损失函数
    - optimizer: 优化器
    - num_epochs: 训练的总轮数
    返回:
    - train_loss_history: 每个epoch的训练损失历史
    - train_acc_history: 每个epoch的训练准确率历史
    - test_loss_history: 每个epoch的测试损失历史
    - test_acc_history: 每个epoch的测试准确率历史
    """
    best_acc = 0.0  # 初始化最佳准确率
    train_loss_history = []  # 记录训练损失的历史
    train_acc_history = []  # 记录训练准确率的历史
    test_loss_history = []  # 记录测试损失的历史
    test_acc_history = []  # 记录测试准确率的历史
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0  # 累计损失
        correct = 0  # 累计正确预测数
        total = 0  # 累计样本数
        # 检查是否可以使用GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 训练阶段
        for images, labels in train_loader:
            images = images.to(device)  # 将图像数据移动到设备
            labels = labels.to(device)  # 将标签数据移动到设备
            optimizer.zero_grad()  # 清空梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            # 计算预测结果
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)  # 累计样本数
            correct += (predicted == labels).sum().item()  # 累计正确预测数
            running_loss += loss.item() * images.size(0)  # 累计损失
        # 计算训练集的平均损失和准确率
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        # 测试阶段，评估模型在测试集上的性能
        test_loss, test_acc = evaluate_model(model, test_loader, criterion)
        # 保存训练和测试的历史数据
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)
        # 如果测试准确率更高，则保存当前模型为最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
        # 保存当前epoch的模型
        torch.save(model.state_dict(), f'models/model_epoch_{epoch+1}.pth')
        # 记录日志信息
        log = (f'Epoch [{epoch+1}/{num_epochs}] | '
               f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | '
               f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
        print(log)
        logging.info(log)
    return train_loss_history, train_acc_history, test_loss_history, test_acc_history