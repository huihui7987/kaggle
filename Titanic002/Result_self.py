'''
VARIABLE DESCRIPTIONS:
survival        Survival
                (0 = No; 1 = Yes)
pclass          Passenger Class社会地位
                (1 = 1st; 2 = 2nd; 3 = 3rd)
name            Name
sex             Sex
age             Age
sibsp           Number of Siblings/Spouses Aboard亲戚配偶
parch           Number of Parents/Children Aboard父母孩子
ticket          Ticket Number票号
fare            Passenger Fare价位
cabin           Cabin船舱编号
embarked        Port of Embarkation码头
                (C = Cherbourg; Q = Queenstown; S = Southampton)
'''

import pandas as pd
from numpy import *
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree,preprocessing
from sklearn import cross_validation
import csv

def loadtrainDataSet(filename):
    '''

    :param filename:
    :return:
    '''
    dataSet = pd.read_csv(filename)

    #处理年龄缺失值,取平均值，也可以考虑直接去掉
    m = ma.masked_array(dataSet['Age'],isnan(dataSet['Age']))
    mean = np.mean(m).astype(int)
    dataSet['Age'] = dataSet['Age'].map(lambda x: mean if np.isnan(x) else x)

    exc_cols = ['PassengerId', 'Survived', 'Name', 'Ticket', 'Embarked', 'Cabin']
    cols = [c for c in dataSet.columns if c not in exc_cols]
    features = dataSet.ix[:,cols]

    label = dataSet['Survived'].values
    return features,label



def loadtestDataSet(filename):
    dataSet = pd.read_csv(filename)
    #处理年龄NAN
    m = ma.masked_array(dataSet['Age'], isnan(dataSet['Age']))
    mean = np.mean(m).astype(int)
    dataSet['Age'] = dataSet['Age'].map(lambda x: mean if np.isnan(x) else x)
    mm = ma.masked_array(dataSet['Fare'], isnan(dataSet['Fare']))
    meanm = np.mean(mm).astype(int)
    dataSet['Fare'] = dataSet['Fare'].map(lambda x: meanm if np.isnan(x) else x)

    exc_cols = ['PassengerId', 'Name', 'Ticket', 'Embarked', 'Cabin']
    cols = [c for c in dataSet.columns if c not in exc_cols]
    testfeatures = dataSet.ix[:, cols]
    ids = dataSet['PassengerId'].values

    return testfeatures,ids

def preProcessToVec(features):
    '''
    :param feature:
    :return:
    '''

    pre_X = features.to_dict(orient = 'records')
    vec = DictVectorizer()
    pre_X = vec.fit_transform(pre_X)
    return pre_X



def trainDicisionTree(feature,label):
    '''
    :return:
    '''
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(feature, label, test_size=0.1, random_state=0)
    dt = tree.DecisionTreeClassifier(max_depth=5)
    dt.fit(X_train, y_train)

    print(dt.predict(X_test))
    print (dt.score(X_test, y_test))#交叉验证大概0.8 的正确率
    #clf = tree.DecisionTreeClassifier()
    #res_X = clf.fit(feature,label)
    #with open("/Users/ghuihui/PycharmProjects/kaggle/Titanic002/allElectronicInformationGainOri.dot", 'w') as f:
    #    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)
    # '1 打开cmd进入dos环境下，并进入../Tarfile/Tname.dot路径下;#2 输入dot -Tpdf name.dot -o name.pdf命令，将dos转化为pdf格式'
    #return res_X


def DicisionTreePredict(feature,label,newRowX):

    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf = clf.fit(feature,label)
    predictedY = clf.predict(newRowX)
    return predictedY


def saveresult(ids,result):
    predictions_file = open("/Users/ghuihui/PycharmProjects/kaggle/Titanic002/dt_submission.csv", "w")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(ids,result))
    predictions_file.close()

def mainDT():

    trainfilename = '/Users/ghuihui/PycharmProjects/kaggle/Titanic002/train.csv'
    feature,label = loadtrainDataSet(trainfilename)

    trainfeatureVEC = preProcessToVec(feature)
    #print(trainfeatureVEC)

    testfilename = '/Users/ghuihui/PycharmProjects/kaggle/Titanic002/test.csv'
    testfeature,ids = loadtestDataSet(testfilename)
    testfeatureVEC = preProcessToVec(testfeature)


    res = DicisionTreePredict(trainfeatureVEC,label,testfeatureVEC)
    print(res)
    #saveresult(ids,res)

    localdd = '/Users/ghuihui/PycharmProjects/kaggle/Titanic002/genderclassmodel.csv'
    localdataSet = pd.read_csv(localdd)
    localvalue = localdataSet['Survived'].values
    print(localvalue)
    count = 0

    for i in range(len(localvalue)):
        if localvalue[i] == res[i]:
            count += 1
    rr = count / len(localvalue)
    print(rr)




mainDT()