from sklearn.linear_model import LogisticRegression

def digit_predict(train_image, train_label, test_image):
    '''
    实现功能：训练模型并输出预测结果
    :param train_sample: 包含多条训练样本的样本集，类型为ndarray,shape为[-1, 8, 8]
    :param train_label: 包含多条训练样本标签的标签集，类型为ndarray
    :param test_sample: 包含多条测试样本的测试集，类型为ndarry
    :return: test_sample对应的预测标签
    '''

    #************* Begin ************#
    train_image = train_image.reshape(-1, 64)
    test_image = test_image.reshape(-1, 64)
    clf = LogisticRegression(solver='lbfgs',max_iter =100,C=0.1)
    clf.fit(train_image, train_label)
    return clf.predict(test_image)
    #************* End **************#