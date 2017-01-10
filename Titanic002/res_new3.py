#!/usr/bin/env python
#coding:utf-8

'''
Created on 2014年11月25日
@author: zhaohf
主要是其中一些程序的写法可以借鉴
'''
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import cross_validation
import csv
df = pd.read_csv('/Users/ghuihui/PycharmProjects/kaggle/Titanic002/train.csv',header=0)
print(df.info())
df = df.drop(['Ticket','Name','Cabin','Embarked'],axis=1)

m = np.ma.masked_array(df['Age'], np.isnan(df['Age']))
mean = np.mean(m).astype(int)
df['Age'] = df['Age'].map(lambda x : mean if np.isnan(x) else x)
df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)#剩下的仅有age与sex不是数值型

X = df.values
y = df['Survived'].values
X = np.delete(X,1,axis=1)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2,random_state=0)
dt = tree.DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)
print (dt.score(X_test,y_test))

test = pd.read_csv('/Users/ghuihui/PycharmProjects/kaggle/Titanic002/test.csv',header=0)
tf = test.drop(['Ticket','Name','Cabin','Embarked'],axis=1)
m = np.ma.masked_array(tf['Age'], np.isnan(tf['Age']))
mean = np.mean(m).astype(int)
tf['Age'] = tf['Age'].map(lambda x : mean if np.isnan(x) else int(x))
tf['Sex'] = tf['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
tf['Fare'] = tf['Fare'].map(lambda x : 0 if np.isnan(x) else int(x)).astype(int)
predicts = dt.predict(tf)
ids = tf['PassengerId'].values

predictions_file = open("/Users/ghuihui/PycharmProjects/kaggle/Titanic002/dt_new.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, predicts))
predictions_file.close()

#下面是生成的决策树的各节点的重要度，也即信息熵，第三个对应的就是性别，占了一半以上的重要性，也说明性别在这次灾难中对于幸存的重要性。

#[ 0.06664883  0.14876052  0.52117953  0.10608185  0.08553209  0.00525581 0.06654137]
#最后的得分是 0.77990。也算是很低的分数了，有人能做到100%正确！