from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def classification(train_feature, train_label, test_feature):
    '''
    对test_feature进行红酒分类
    :param train_feature: 训练集数据 类型为ndarray
    :param train_label: 训练集标签 类型为ndarray
    :param test_feature: 测试集数据 类型为ndarray
    :return: 测试集数据的分类结果
    '''

    #********* Begin *********#
    scaler = StandardScaler()
    after_scaler = scaler.fit_transform(train_feature)
    after_scaler2 = scaler.fit_transform(test_feature)
    #生成K近邻分类器
    clf=KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    #训练分类器
    clf.fit(after_scaler, train_label)  
    #进行预测         
    predict_result=clf.predict(after_scaler2)
    return predict_result
    #********* End **********#