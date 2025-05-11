import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    将输入数据转换为列形式，用于卷积运算的高效实现。
    :param input_data: 输入数据，形状为 (N, C, H, W)
    :param filter_h: 滤波器的高度
    :param filter_w: 滤波器的宽度
    :param stride: 步幅
    :param pad: 填充大小
    :return: 转换后的列形式数据
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1  # 输出高度
    out_w = (W + 2 * pad - filter_w) // stride + 1  # 输出宽度
    # 对输入数据进行填充
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    将列形式的数据转换回原始图像形式，用于反向传播。
    :param col: 列形式数据
    :param input_shape: 原始输入数据的形状
    :param filter_h: 滤波器的高度
    :param filter_w: 滤波器的宽度
    :param stride: 步幅
    :param pad: 填充大小
    :return: 转换回的图像形式数据
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1  # 输出高度
    out_w = (W + 2 * pad - filter_w) // stride + 1  # 输出宽度
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    return img[:, :, pad:H + pad, pad:W + pad]


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        卷积层初始化。
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小
        :param stride: 步幅
        :param padding: 填充大小
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # He 初始化权重
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2 / (in_channels * kernel_size * kernel_size))
        self.bias = np.zeros((out_channels, 1))
        self.dweights = np.zeros_like(self.weights)  # 权重梯度
        self.dbias = np.zeros_like(self.bias)  # 偏置梯度
        # 动量项
        self.v_weights = np.zeros_like(self.weights)
        self.v_bias = np.zeros_like(self.bias)

    def forward(self, X):
        """
        前向传播。
        :param X: 输入数据，形状为 (N, C, H, W)
        :return: 卷积后的输出
        """
        self.X = X
        N, C, H, W = X.shape
        F, _, HH, WW = self.weights.shape
        H_out = 1 + (H + 2 * self.padding - HH) // self.stride  # 输出高度
        W_out = 1 + (W + 2 * self.padding - WW) // self.stride  # 输出宽度
        col = im2col(X, HH, WW, self.stride, self.padding)  # 将输入转换为列形式
        col_W = self.weights.reshape(F, -1).T  # 卷积核展开为列
        out = np.dot(col, col_W) + self.bias.T  # 卷积运算
        out = out.reshape(N, H_out, W_out, -1).transpose(0, 3, 1, 2)  # 调整输出形状
        return out

    def backward(self, dout):
        """
        反向传播。
        :param dout: 上一层的梯度
        :return: 当前层的梯度
        """
        N, F, H_out, W_out = dout.shape
        _, C, HH, WW = self.weights.shape
        H = (H_out - 1) * self.stride + HH - 2 * self.padding  # 输入高度
        W = (W_out - 1) * self.stride + WW - 2 * self.padding  # 输入宽度
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, F)  # 调整梯度形状
        col = im2col(self.X, HH, WW, self.stride, self.padding)  # 输入转换为列形式
        self.dweights = np.dot(col.T, dout).transpose(1, 0).reshape(F, C, HH, WW)  # 计算权重梯度
        self.dbias = np.sum(dout, axis=0, keepdims=True).T  # 计算偏置梯度
        dcol = np.dot(dout, self.weights.reshape(F, -1))  # 梯度反向传播到列形式
        dX = col2im(dcol, self.X.shape, HH, WW, self.stride, self.padding)  # 列形式转换回输入形状
        return dX

    def update_params(self, learning_rate, momentum=0.9):
        """
        更新参数。
        :param learning_rate: 学习率
        :param momentum: 动量因子
        """
        self.v_weights = momentum * self.v_weights - learning_rate * self.dweights  # 更新权重动量
        self.weights += self.v_weights  # 更新权重
        self.v_bias = momentum * self.v_bias - learning_rate * self.dbias  # 更新偏置动量
        self.bias += self.v_bias  # 更新偏置


class BatchNormalization:
    def __init__(self, channels):
        """
        批量归一化层初始化。
        :param channels: 通道数
        """
        self.gamma = np.ones((1, channels, 1, 1))  # 缩放参数
        self.beta = np.zeros((1, channels, 1, 1))  # 平移参数
        self.dgamma = np.zeros_like(self.gamma)  # 缩放参数的梯度
        self.dbeta = np.zeros_like(self.beta)  # 平移参数的梯度
        self.moving_mean = np.zeros((1, channels, 1, 1))  # 移动平均的均值
        self.moving_var = np.ones((1, channels, 1, 1))  # 移动平均的方差
        self.eps = 1e-5  # 防止除零的小值
        self.momentum = 0.9  # 动量因子

    def forward(self, x, train_flg=True):
        """
        前向传播。
        :param x: 输入数据
        :param train_flg: 是否为训练模式
        :return: 归一化后的输出
        """
        self.train_flg = train_flg
        N, C, H, W = x.shape
        if train_flg:
            mu = x.mean(axis=(0, 2, 3), keepdims=True)  # 计算均值
            xc = x - mu  # 去均值
            var = np.mean(xc ** 2, axis=(0, 2, 3), keepdims=True)  # 计算方差
            std = np.sqrt(var + self.eps)  # 计算标准差
            xn = xc / std  # 标准化
            self.xc = xc
            self.xn = xn
            self.std = std
            # 更新移动平均的均值和方差
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * mu
            self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * var
        else:
            # 推理模式下使用移动平均的均值和方差
            xc = x - self.moving_mean
            xn = xc / np.sqrt(self.moving_var + self.eps)
        out = self.gamma * xn + self.beta  # 缩放和平移
        return out

    def backward(self, dout):
        """
        反向传播。
        :param dout: 上一层的梯度
        :return: 当前层的梯度
        """
        N, C, H, W = dout.shape
        dbeta = dout.sum(axis=(0, 2, 3), keepdims=True)  # 平移参数的梯度
        dgamma = np.sum(self.xn * dout, axis=(0, 2, 3), keepdims=True)  # 缩放参数的梯度
        dxn = self.gamma * dout  # 对标准化输出的梯度
        dxc = dxn / self.std  # 对去均值后的输入的梯度
        dstd = -np.sum((dxn * self.xc) / (self.std ** 2), axis=(0, 2, 3), keepdims=True)  # 对标准差的梯度
        dvar = 0.5 * dstd / self.std  # 对方差的梯度
        dxc += (2.0 / (N * H * W)) * self.xc * dvar  # 对去均值后的输入的梯度
        dmu = np.sum(dxc, axis=(0, 2, 3), keepdims=True)  # 对均值的梯度
        dx = dxc - dmu / (N * H * W)  # 对输入的梯度
        self.dgamma = dgamma
        self.dbeta = dbeta
        return dx

    def update_params(self, learning_rate):
        """
        更新参数。
        :param learning_rate: 学习率
        """
        self.gamma -= learning_rate * self.dgamma
        self.beta -= learning_rate * self.dbeta


class ReLU:
    def __init__(self):
        """
        ReLU 激活函数初始化。
        """
        self.mask = None

    def forward(self, x):
        """
        前向传播。
        :param x: 输入数据
        :return: 激活后的输出
        """
        self.mask = (x <= 0)  # 记录小于等于0的位置
        out = x.copy()
        out[self.mask] = 0  # 小于等于0的值置为0
        return out

    def backward(self, dout):
        """
        反向传播。
        :param dout: 上一层的梯度
        :return: 当前层的梯度
        """
        dout[self.mask] = 0  # 小于等于0的位置梯度为0
        dx = dout
        return dx


class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        """
        最大池化层初始化。
        :param pool_size: 池化窗口大小
        :param stride: 步幅
        """
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, X):
        """
        前向传播。
        :param X: 输入数据，形状为 (N, C, H, W)
        :return: 池化后的输出
        """
        self.X = X
        N, C, H, W = X.shape
        H_out = 1 + (H - self.pool_size) // self.stride  # 输出高度
        W_out = 1 + (W - self.pool_size) // self.stride  # 输出宽度
        out = np.zeros((N, C, H_out, W_out))
        self.arg_max = np.zeros((N, C, H_out, W_out), dtype=np.int64)  # 记录最大值索引

        for i in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        # 计算池化窗口内的最大值
                        out[i, c, h, w] = np.max(X[i, c, h_start:h_end, w_start:w_end])
                        # 记录最大值的位置索引
                        self.arg_max[i, c, h, w] = np.argmax(X[i, c, h_start:h_end, w_start:w_end])
        return out

    def backward(self, dout):
        """
        反向传播。
        :param dout: 上一层的梯度，形状为 (N, C, H_out, W_out)
        :return: 当前层的梯度
        """
        N, C, H_out, W_out = dout.shape
        _, _, H, W = self.X.shape
        dX = np.zeros_like(self.X)  # 初始化输入梯度
        for i in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        # 将梯度传递到最大值的位置
                        idx = self.arg_max[i, c, h, w]
                        dX[i, c, h_start + idx // self.pool_size, w_start + idx % self.pool_size] = dout[i, c, h, w]
        return dX


class FullyConnected:
    def __init__(self, input_size, output_size):
        """
        全连接层初始化。
        :param input_size: 输入大小
        :param output_size: 输出大小
        """
        # He 初始化权重
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.bias = np.zeros((1, output_size))
        self.dweights = np.zeros_like(self.weights)  # 权重梯度
        self.dbias = np.zeros_like(self.bias)  # 偏置梯度
        # 动量项
        self.v_weights = np.zeros_like(self.weights)
        self.v_bias = np.zeros_like(self.bias)

    def forward(self, X):
        """
        前向传播。
        :param X: 输入数据
        :return: 输出数据
        """
        self.X = X
        return np.dot(X, self.weights) + self.bias

    def backward(self, dout):
        """
        反向传播。
        :param dout: 上一层的梯度
        :return: 当前层的梯度
        """
        dX = np.dot(dout, self.weights.T)  # 计算输入梯度
        self.dweights = np.dot(self.X.T, dout)  # 计算权重梯度
        self.dbias = np.sum(dout, axis=0, keepdims=True)  # 计算偏置梯度
        return dX

    def update_params(self, learning_rate, momentum=0.9):
        """
        更新参数。
        :param learning_rate: 学习率
        :param momentum: 动量因子
        """
        self.v_weights = momentum * self.v_weights - learning_rate * self.dweights  # 更新权重动量
        self.weights += self.v_weights  # 更新权重
        self.v_bias = momentum * self.v_bias - learning_rate * self.dbias  # 更新偏置动量
        self.bias += self.v_bias  # 更新偏置


class Softmax:
    def forward(self, X):
        """
        前向传播。
        :param X: 输入数据
        :return: Softmax 激活后的输出
        """
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))  # 防止溢出
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)

    def backward(self, y_pred, y_true):
        """
        反向传播。
        :param y_pred: 预测值
        :param y_true: 真实标签
        :return: 当前层的梯度
        """
        N = y_pred.shape[0]
        return (y_pred - y_true) / N  # 计算梯度


class CNN:
    def __init__(self):
        """
        初始化 CNN 模型。
        """
        self.conv1 = Conv2D(in_channels=1, out_channels=6, kernel_size=5, padding=2)  # 第一层卷积
        self.bn1 = BatchNormalization(6)  # 批量归一化
        self.relu1 = ReLU()  # ReLU 激活函数
        self.pool1 = MaxPool2D(pool_size=2, stride=2)  # 最大池化
        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5, padding=0)  # 第二层卷积
        self.bn2 = BatchNormalization(16)  # 批量归一化
        self.relu2 = ReLU()  # ReLU 激活函数
        self.pool2 = MaxPool2D(pool_size=2, stride=2)  # 最大池化
        self.fc1 = FullyConnected(input_size=5 * 5 * 16, output_size=120)  # 全连接层 1
        self.relu3 = ReLU()  # ReLU 激活函数
        self.fc2 = FullyConnected(input_size=120, output_size=84)  # 全连接层 2
        self.relu4 = ReLU()  # ReLU 激活函数
        self.fc3 = FullyConnected(input_size=84, output_size=10)  # 全连接层 3
        self.softmax = Softmax()  # Softmax 激活函数

    def forward(self, X):
        """
        前向传播。
        :param X: 输入数据
        :return: 模型输出
        """
        X = X.reshape(-1, 1, 28, 28)  # 调整输入形状为 (N, C, H, W)
        out = self.conv1.forward(X)  # 第一层卷积
        out = self.bn1.forward(out)  # 批量归一化
        out = self.relu1.forward(out)  # ReLU 激活
        out = self.pool1.forward(out)  # 最大池化
        out = self.conv2.forward(out)  # 第二层卷积
        out = self.bn2.forward(out)  # 批量归一化
        out = self.relu2.forward(out)  # ReLU 激活
        out = self.pool2.forward(out)  # 最大池化
        out = out.reshape(out.shape[0], -1)  # 展平
        out = self.fc1.forward(out)  # 全连接层 1
        out = self.relu3.forward(out)  # ReLU 激活
        out = self.fc2.forward(out)  # 全连接层 2
        out = self.relu4.forward(out)  # ReLU 激活
        out = self.fc3.forward(out)  # 全连接层 3
        out = self.softmax.forward(out)  # Softmax 激活
        return out

    def backward(self, y_pred, y_true):
        """
        反向传播。
        :param y_pred: 模型预测值
        :param y_true: 真实标签
        :return: 梯度
        """
        dout = self.softmax.backward(y_pred, y_true)  # Softmax 反向传播
        dout = self.fc3.backward(dout)  # 全连接层 3 反向传播
        dout = self.relu4.backward(dout)  # ReLU 激活反向传播
        dout = self.fc2.backward(dout)  # 全连接层 2 反向传播
        dout = self.relu3.backward(dout)  # ReLU 激活反向传播
        dout = self.fc1.backward(dout)  # 全连接层 1 反向传播
        dout = dout.reshape(-1, 16, 5, 5)  # 调整形状
        dout = self.pool2.backward(dout)  # 最大池化反向传播
        dout = self.relu2.backward(dout)  # ReLU 激活反向传播
        dout = self.bn2.backward(dout)  # 批量归一化反向传播
        dout = self.conv2.backward(dout)  # 第二层卷积反向传播
        dout = self.pool1.backward(dout)  # 最大池化反向传播
        dout = self.relu1.backward(dout)  # ReLU 激活反向传播
        dout = self.bn1.backward(dout)  # 批量归一化反向传播
        dout = self.conv1.backward(dout)  # 第一层卷积反向传播
        return dout

    def update_params(self, learning_rate, momentum=0.9):
        """
        更新模型参数。
        :param learning_rate: 学习率
        :param momentum: 动量因子
        """
        self.conv1.update_params(learning_rate, momentum)  # 更新第一层卷积参数
        self.bn1.update_params(learning_rate)  # 更新批量归一化参数
        self.conv2.update_params(learning_rate, momentum)  # 更新第二层卷积参数
        self.bn2.update_params(learning_rate)  # 更新批量归一化参数
        self.fc1.update_params(learning_rate, momentum)  # 更新全连接层 1 参数
        self.fc2.update_params(learning_rate, momentum)  # 更新全连接层 2 参数
        self.fc3.update_params(learning_rate, momentum)  # 更新全连接层 3 参数

    def save_model(self, filename):
        """
        保存模型参数到文件。
        :param filename: 文件名
        """
        model_params = {
            'conv1_weights': self.conv1.weights,
            'conv1_bias': self.conv1.bias,
            'bn1_gamma': self.bn1.gamma,
            'bn1_beta': self.bn1.beta,
            'conv2_weights': self.conv2.weights,
            'conv2_bias': self.conv2.bias,
            'bn2_gamma': self.bn2.gamma,
            'bn2_beta': self.bn2.beta,
            'fc1_weights': self.fc1.weights,
            'fc1_bias': self.fc1.bias,
            'fc2_weights': self.fc2.weights,
            'fc2_bias': self.fc2.bias,
            'fc3_weights': self.fc3.weights,
            'fc3_bias': self.fc3.bias,
        }
        np.savez(filename, **model_params)  # 保存为 .npz 文件

    def load_model(self, filename):
        """
        从文件加载模型参数。
        :param filename: 文件名
        """
        model_params = np.load(filename)  # 加载 .npz 文件
        self.conv1.weights = model_params['conv1_weights']
        self.conv1.bias = model_params['conv1_bias']
        self.bn1.gamma = model_params['bn1_gamma']
        self.bn1.beta = model_params['bn1_beta']
        self.conv2.weights = model_params['conv2_weights']
        self.conv2.bias = model_params['conv2_bias']
        self.bn2.gamma = model_params['bn2_gamma']
        self.bn2.beta = model_params['bn2_beta']
        self.fc1.weights = model_params['fc1_weights']
        self.fc1.bias = model_params['fc1_bias']
        self.fc2.weights = model_params['fc2_weights']
        self.fc2.bias = model_params['fc2_bias']
        self.fc3.weights = model_params['fc3_weights']
        self.fc3.bias = model_params['fc3_bias']
    
    def print_model(self):
        """
        打印模型结构和参数数量。
        """
        layers = [
            (self.conv1, 'Conv2D', 'conv1'),
            (self.bn1, 'BatchNorm', 'bn1'),
            (self.relu1, 'ReLU', 'relu1'),
            (self.pool1, 'MaxPool2D', 'pool1'),
            (self.conv2, 'Conv2D', 'conv2'),
            (self.bn2, 'BatchNorm', 'bn2'),
            (self.relu2, 'ReLU', 'relu2'),
            (self.pool2, 'MaxPool2D', 'pool2'),
            (self.fc1, 'FullyConnected', 'fc1'),
            (self.relu3, 'ReLU', 'relu3'),
            (self.fc2, 'FullyConnected', 'fc2'),
            (self.relu4, 'ReLU', 'relu4'),
            (self.fc3, 'FullyConnected', 'fc3'),
        ]
        current_shape = (1, 28, 28)  # 输入形状: (channels, height, width)
        total_params = 0
        print("Layer (type)         Output Shape         Param #")
        print("===================================================")
        for layer_info in layers:
            layer_obj, layer_type, layer_name = layer_info
            params = 0
            output_shape = current_shape
            if layer_type == 'Conv2D':
                in_channels, H_in, W_in = current_shape
                padding = layer_obj.padding
                kernel_size = layer_obj.kernel_size
                stride = layer_obj.stride
                out_channels = layer_obj.out_channels                
                H_out = (H_in + 2 * padding - kernel_size) // stride + 1
                W_out = (W_in + 2 * padding - kernel_size) // stride + 1
                output_shape = (out_channels, H_out, W_out)
                params = (in_channels * kernel_size**2) * out_channels + out_channels
                layer_desc = f"Conv2D"
            elif layer_type == 'BatchNorm':
                channels = current_shape[0]
                params = 2 * channels  # gamma和beta
                layer_desc = f"BatchNorm"
                output_shape = current_shape
            elif layer_type == 'ReLU':
                layer_desc = "ReLU"
                params = 0
                output_shape = current_shape
            elif layer_type == 'MaxPool2D':
                pool_size = layer_obj.pool_size
                stride = layer_obj.stride
                channels, H_in, W_in = current_shape
                H_out = (H_in - pool_size) // stride + 1
                W_out = (W_in - pool_size) // stride + 1
                output_shape = (channels, H_out, W_out)
                layer_desc = f"MaxPool2D"
                params = 0
            elif layer_type == 'FullyConnected':
                if len(current_shape) == 3:
                    input_dim = current_shape[0] * current_shape[1] * current_shape[2]
                else:
                    input_dim = current_shape[0]
                output_dim = layer_obj.weights.shape[1]
                params = input_dim * output_dim + output_dim
                output_shape = (output_dim,)
                layer_desc = f"FullyConnected"
            # 格式化输出
            print(f"{layer_desc.ljust(20)} {str(output_shape).ljust(20)} {params}")
            total_params += params
            current_shape = output_shape
        print("===================================================")
        print(f"Total params: {total_params}")


if __name__ == "__main__":
    model = CNN()
    model.print_model()