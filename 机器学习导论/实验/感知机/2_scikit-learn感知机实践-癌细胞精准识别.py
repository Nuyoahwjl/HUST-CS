#encoding=utf8
import os
if os.path.exists('./step2/result.csv'):
    os.remove('./step2/result.csv')

#********* Begin *********#

import pandas as pd
#获取训练数据
train_data = pd.read_csv('./step2/train_data.csv')
#获取训练标签
train_label = pd.read_csv('./step2/train_label.csv')
train_label = train_label['target']
#获取测试数据
test_data = pd.read_csv('./step2/test_data.csv')

from sklearn.linear_model import Perceptron
clf = Perceptron(eta0 = 0.01, max_iter = 500)
clf.fit(train_data, train_label)
result = clf.predict(test_data)
# 将预测结果转换为 DataFrame
result_df = pd.DataFrame(result, columns=['result'])
# 将 DataFrame 保存为 CSV 文件
result_df.to_csv('./step2/result.csv', index=False)

#********* End *********#