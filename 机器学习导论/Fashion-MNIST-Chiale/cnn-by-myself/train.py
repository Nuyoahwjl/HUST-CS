import numpy as np
from data_loader import load_mnist
from cnn_model import CNN
import datetime
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)


def train(train_images, train_labels, model, max_steps=5000, batch_size=64, learning_rate=0.0005):  
    # 数据预处理
    train_images = train_images / 255.0  # 归一化
    train_labels = np.eye(10)[train_labels]  # one-hot编码
    # 创建log目录
    log_dir = os.path.join(current_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 创建日志文件路径
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_logs_{timestamp}.log')
    # loss&accuracy
    losses = []
    accuracies = []
    for step in range(max_steps):
        # # 学习率每1000步衰减50%
        # current_lr = learning_rate * (0.5 ** (step // 1000))
        batch_mask = np.random.choice(len(train_images), batch_size)
        X_batch = train_images[batch_mask]
        y_batch = train_labels[batch_mask]
        # 前向传播
        y_pred = model.forward(X_batch)
        # 计算损失
        loss = -np.sum(y_batch * np.log(y_pred + 1e-8)) / batch_size
        losses.append(loss)
        # 计算准确率
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
        accuracies.append(accuracy)
        # 反向传播和参数更新
        model.backward(y_pred, y_batch)
        model.update_params(learning_rate)
        # model.update_params(current_lr)
        # 实时保存损失和准确率
        np.save(os.path.join(log_dir, 'losses.npy'), np.array(losses))
        np.save(os.path.join(log_dir, 'accuracies.npy'), np.array(accuracies))
        # 写入日志和打印信息
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(log_file, 'a') as f:
            f.write(f"{timestamp}, Step: {step+1}/{max_steps}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}\n")
            f.flush()
        print(f'Step: {step+1}/{max_steps}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
        # 每1000步保存模型
        if (step+1) % 1000 == 0:
            model_dir = os.path.join(current_dir, 'models')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model.save_model(os.path.join(model_dir, f'model_step{step+1}.npz'))


if __name__ == '__main__':
    data_dir = os.path.join(project_root, 'data')
    train_images, train_labels = load_mnist(data_dir, kind='train')
    model = CNN()
    max_steps = 5000
    batch_size = 64  
    learning_rate = 0.0005
    train(train_images, train_labels, model, max_steps, batch_size, learning_rate)
