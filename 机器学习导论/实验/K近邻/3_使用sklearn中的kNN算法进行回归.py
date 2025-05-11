from sklearn.neighbors import KNeighborsRegressor

def regression(train_feature, train_label, test_feature):
    '''
    使用KNeighborsRegressor对test_feature进行分类
    :param train_feature: 训练集数据
    :param train_label: 训练集标签
    :param test_feature: 测试集数据
    :return: 测试集预测结果
    '''
    
    #********* Begin *********#
    #生成K近邻回归器
    clf=KNeighborsRegressor(n_neighbors=5, metric='minkowski', p=2)
    #训练分类器
    clf.fit(train_feature, train_label)  
    #进行预测         
    predict_result=clf.predict(test_feature)
    return predict_result
    #********* End *********#