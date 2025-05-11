from sklearn.tree import DecisionTreeClassifier
import pandas as pd
clf = DecisionTreeClassifier()
X_train = pd.read_csv('./step7/train_data.csv').as_matrix()
Y_train = pd.read_csv('./step7/train_label.csv').as_matrix() 
clf.fit(X_train, Y_train)
X_test = pd.read_csv('./step7/test_data.csv').as_matrix()
result = clf.predict(X_test)
# 将结果保存为CSV文件
pd.DataFrame(result, columns=['label']).to_csv('./step7/predict.csv', index=False)