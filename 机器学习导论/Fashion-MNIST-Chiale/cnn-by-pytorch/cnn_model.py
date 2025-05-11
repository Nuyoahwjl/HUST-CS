import torch
import torch.nn as nn

# 定义CNN模型
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        # 第一层卷积：输入通道数为1，输出通道数为32，卷积核大小为3x3，使用padding=1保持尺寸
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        # 第一层批归一化
        self.bn1 = nn.BatchNorm2d(32)
        # 最大池化层：池化窗口为2x2，步幅为2
        self.pool = nn.MaxPool2d(2, 2)
        # 第二层卷积：输入通道数为32，输出通道数为64，卷积核大小为3x3，使用padding=1保持尺寸
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # 第二层批归一化
        self.bn2 = nn.BatchNorm2d(64)
        # 全连接层1：输入特征数为64*7*7，输出特征数为512
        self.fc1 = nn.Linear(64*7*7, 512)
        # Dropout层：随机丢弃50%的神经元
        self.dropout = nn.Dropout(0.5)
        # 全连接层2：输入特征数为512，输出特征数为10（对应10个分类）
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # 前向传播：卷积 -> 批归一化 -> ReLU激活 -> 池化
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        # 展平张量以输入全连接层
        x = x.view(-1, 64*7*7)
        # 全连接层1 -> ReLU激活 -> Dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        # 全连接层2（输出层）
        x = self.fc2(x)
        return x

# 打印模型参数信息的函数
def print_model(model):
    # 打印表头
    print("{:<20} {:<20} {:<10}".format("Param", "Shape", "Num #"))
    print("="*50)
    total_params = 0
    # 遍历模型的所有参数
    for name, param in model.named_parameters():
        if param.requires_grad:  # 只统计需要梯度更新的参数
            param_count = param.numel()  # 参数总数
            total_params += param_count
            # 打印参数名称、形状和数量
            print("{:<20} {:<20} {:<10}".format(name, str(tuple(param.shape)), str(param_count)))
    print("="*50)
    # 打印参数总数
    print(f"Total params: {total_params}")

if __name__ == "__main__":
    model = FashionCNN()
    print_model(model)