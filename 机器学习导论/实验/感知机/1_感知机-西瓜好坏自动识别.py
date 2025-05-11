#encoding=utf8
import numpy as np
#构建感知机算法
class Perceptron(object):
    def __init__(self, learning_rate = 0.01, max_iter = 200):
        self.lr = learning_rate
        self.max_iter = max_iter
    def fit(self, data, label):
        '''
        input:data(ndarray):训练数据特征
              label(ndarray):训练数据标签
        output:w(ndarray):训练好的权重
               b(ndarry):训练好的偏置
        '''
        #编写感知机训练方法，w为权重，b为偏置
        self.w = np.array([1.]*data.shape[1])
        self.b = np.array([1.])
        #********* Begin *********#
        for _ in range(self.max_iter):
            has_error = False  
            for i in range(len(data)):
                x = data[i]
                y = label[i]
                # 判断是否分类错误
                if y * (np.dot(x, self.w) + self.b) <= 0:
                    # 更新权重和偏置
                    self.w += self.lr * y * x
                    self.b += self.lr * y
                    has_error = True
            # 如果没有分类错误，提前退出
            if not has_error:
                break
        #********* End *********#
    def predict(self, data):
        '''
        input:data(ndarray):测试数据特征
        output:predict(ndarray):预测标签
        '''
        #********* Begin *********#
        # 计算预测值
        scores = np.dot(data, self.w) + self.b
        # 根据符号返回预测标签
        predict = np.where(scores > 0, 1, -1)
        #********* End *********#
        return predict
    



#测试
data = np.array([[3, 3], [4, 3], [1, 1]])
label = np.array([1, 1, -1])
perceptron = Perceptron()
perceptron.fit(data, label)
test_data = np.array([[2, 2], [5, 5]])
print(perceptron.predict(test_data))