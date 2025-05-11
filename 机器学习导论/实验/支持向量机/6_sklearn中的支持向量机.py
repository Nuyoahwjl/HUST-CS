#encoding=utf8
from sklearn.svm import SVC

def svm_classifier(train_data,train_label,test_data):
      '''
      input:train_data(ndarray):训练样本
            train_label(ndarray):训练标签
            test_data(ndarray):测试样本
      output:predict(ndarray):预测结果      
      '''
      #********* Begin *********#
      svc=SVC(C=1.0,kernel="rbf",max_iter=100)
      svc.fit(train_data,train_label)
      predict = svc.predict(test_data)
      #********* End *********#
      return predict