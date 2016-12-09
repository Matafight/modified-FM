#_*_ coding:utf-8_*_
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import my_pyfmlib as pylibfm


train_data_name = 'u2.base'
test_data_name = 'u2.test'
def loadData(filename):
    data=[]
    y = []
    users=set()
    items=set()
    with open(filename) as f:
        for line in f:
            (user,movieid,rating,ts)=line.split('\t')
            data.append({"user_id":str(user),"movie_id":str(movieid)})
            y.append(float(rating))
            users.add(user)
            items.add(movieid)
    return (data,np.array(y),users,items)

(train_data,train_label,train_users,train_items)= loadData('../data/'+train_data_name)
(test_data,test_label,test_users,test_items)=loadData('../data/'+test_data_name)
v = DictVectorizer()
x_train=v.fit_transform(train_data)
x_test = v.fit_transform(test_data)

fm = pylibfm.FM(num_factors = 10,num_iter=100,verbose = True,task="regression",initial_learning_rate=0.001,learning_rate_schedule="optimal",dataname=train_data_name)

fm.fit(x_train,train_label)
pre_label = fm.predict(x_test)

diff = 0.5*np.sum((pre_label-test_label)**2)/test_label.size
fh = open('./results/'+train_data_name,'a')
fh.write("--test--RMSE---"+str(diff))
print(diff)

