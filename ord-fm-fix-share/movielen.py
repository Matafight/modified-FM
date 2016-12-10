#_*_ coding:utf-8_*_
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import my_pyfmlib as pylibfm
#import my_cross_validation as mcv # multi-thread failed
import mymultiprocess_crossvalidation as mcv

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

if __name__=='__main__':
    print("helo")
    train_data_name = 'u2.base'
    test_data_name = 'u2.test'
    (train_data,train_label,train_users,train_items)= loadData('../data/'+train_data_name)
    (test_data,test_label,test_users,test_items)=loadData('../data/'+test_data_name)
    v = DictVectorizer()
    x_train=v.fit_transform(train_data)
    x_test = v.fit_transform(test_data)

    mycv = mcv.cross_val_regularization(x_train,train_label,train_data_name)
    best_reg = mycv.sele_para()
    fm = pylibfm.FM(num_factors = 10,num_iter=100,verbose = True,task="regression",initial_learning_rate=0.001,learning_rate_schedule="optimal",dataname=train_data_name,reg_1 = best_reg[0], reg_2 = best_reg[1],x_test,test_label)

    fm.fit(x_train,train_label)
    pre_label = fm.predict(x_test)

    diff = 0.5*np.sum((pre_label-test_label)**2)/test_label.size
    fh = open('./results/'+train_data_name,'a')
    fh.write("--test--RMSE---"+str(diff))
    print(diff)
