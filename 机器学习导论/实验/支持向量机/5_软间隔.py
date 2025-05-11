#encoding=utf8
import numpy as np
class SVM:
    def __init__(self, max_iter=100, kernel='linear'):
        '''
        input:max_iter(int):最大训练轮数
              kernel(str):核函数，等于'linear'表示线性，等于'poly'表示多项式
        '''
        self.max_iter = max_iter
        self._kernel = kernel
    #初始化模型
    def init_args(self, features, labels):
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels
        self.b = 0.0
        # 将Ei保存在一个列表里
        self.alpha = np.zeros(self.m)
        self.E = [self._E(i) for i in range(self.m)]
        # 松弛变量
        self.C = 0.95

    #********* Begin *********#    
    # kkt条件    
    def _kkt(self, i):
        y_g = self.Y[i] * self.g(self.X[i])
        if (self.alpha[i] < self.C and y_g < 1 - 1e-3) or (self.alpha[i] > 0 and y_g > 1 + 1e-3):
            return False
        return True

    # g(x)预测值，输入xi（X[i]）
    def g(self, x):
        g_x = self.b
        for i in range(self.m):
            g_x += self.alpha[i] * self.Y[i] * self.kernel(x, self.X[i])
        return g_x
    
    # 核函数
    def kernel(self, x1, x2):
        if self._kernel == 'linear':
            return np.dot(x1, x2)
        elif self._kernel == 'poly':
            return (np.dot(x1, x2) + 1) ** 2
        else:
            raise ValueError('Unsupported kernel type')
        
    # E（x）为g(x)对输入x的预测值和y的差
    def _E(self, i):
        return self.g(self.X[i]) - self.Y[i]
    def update_E(self):
        for i in range(self.m):
            self.E[i] = self._E(i)

    #选择参数   
    def select_j(self, i):
        max_delta = 0
        selected_j = -1
        for j in range(self.m):
            if j == i:
                continue
            delta = abs(self.E[i] - self.E[j])
            if delta > max_delta:
                max_delta = delta
                selected_j = j
        if selected_j == -1:
            selected_j = np.random.choice(list(set(range(self.m)) - {i}))
        return selected_j
        
    #训练
    def fit(self, features, labels):
        self.init_args(features, labels)
        for _ in range(self.max_iter):
            changed = False
            for i in range(self.m):
                if self._kkt(i):
                    continue
                j = self.select_j(i)
                alpha_i_old = self.alpha[i].copy()
                alpha_j_old = self.alpha[j].copy()
                if self.Y[i] != self.Y[j]:
                    L = max(0, alpha_j_old - alpha_i_old)
                    H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                else:
                    L = max(0, alpha_i_old + alpha_j_old - self.C)
                    H = min(self.C, alpha_i_old + alpha_j_old)
                if L == H:
                    continue
                eta = self.kernel(self.X[i], self.X[i]) + self.kernel(self.X[j], self.X[j]) - 2 * self.kernel(self.X[i], self.X[j])
                if eta <= 0:
                    continue
                alpha_j_new = alpha_j_old + self.Y[j] * (self.E[i] - self.E[j]) / eta
                alpha_j_new = np.clip(alpha_j_new, L, H)
                if abs(alpha_j_new - alpha_j_old) < 1e-5:
                    continue
                alpha_i_new = alpha_i_old + self.Y[i] * self.Y[j] * (alpha_j_old - alpha_j_new)
                self.alpha[i] = alpha_i_new
                self.alpha[j] = alpha_j_new
                b1 = self.b - self.E[i] - self.Y[i] * (alpha_i_new - alpha_i_old) * self.kernel(self.X[i], self.X[i]) - self.Y[j] * (alpha_j_new - alpha_j_old) * self.kernel(self.X[i], self.X[j])
                b2 = self.b - self.E[j] - self.Y[i] * (alpha_i_new - alpha_i_old) * self.kernel(self.X[i], self.X[j]) - self.Y[j] * (alpha_j_new - alpha_j_old) * self.kernel(self.X[j], self.X[j])
                if 0 < alpha_i_new < self.C:
                    self.b = b1
                elif 0 < alpha_j_new < self.C:
                    self.b = b2
                else:
                    self.b = (b1 + b2) / 2
                self.update_E()
                changed = True
            if not changed:
                break
    #********* End *********# 
           
    def predict(self, data):
        r = self.b
        for i in range(self.m):
            r += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])
        return 1 if r > 0 else -1
    
    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            result = self.predict(X_test[i])
            if result == y_test[i]:
                right_count += 1
        return right_count / len(X_test)
    def _weight(self):

        yx = self.Y.reshape(-1, 1)*self.X
        self.w = np.dot(yx.T, self.alpha)
        return self.w