import torch

# 评估函数
def evaluate_model(model, test_loader, criterion):
    # 设置模型为评估模式
    model.eval()
    running_loss = 0.0  # 累计损失
    correct = 0  # 正确预测的样本数
    total = 0  # 总样本数
    
    # 检查是否可以使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():  # 禁用梯度计算
        for images, labels in test_loader:
            # 将数据移动到设备（CPU或GPU）
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)  # 计算损失
            
            # 获取预测结果
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)  # 累计样本数
            correct += (predicted == labels).sum().item()  # 累计正确预测数
            running_loss += loss.item() * images.size(0)  # 累计损失
    
    # 返回平均损失和准确率
    return running_loss / len(test_loader.dataset), correct / total
