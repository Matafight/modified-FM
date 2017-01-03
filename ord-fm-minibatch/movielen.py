#_*_ coding:utf-8_*_
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import my_pyfmlib as pylibfm
import mymultiprocess_crossvalidation as mcv
import time
import os

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


def performance_cross_validation(data_name,x_train,y_train,x_test,y_test,num_attributes):
    num_factors = 10
    mycv = mcv.cross_val_regularization(train_data = x_train,train_label = train_label,num_factors = num_factors,num_attributes = num_attributes, dataname = data_name)
    best_reg = mycv.sele_para()
    fm = pylibfm.FM(num_factors = num_factors,num_iter=100,verbose = True,task="regression",initial_learning_rate=0.001,learning_rate_schedule="optimal",dataname=data_name,reg_1 = best_reg[0], reg_2 = best_reg[1])
    fm.fit(x_train,y_train,x_test,y_test,num_attributes,ifall = True)


if __name__=='__main__':
    #train_data_name = 'ml-1m-train.txt'
    #test_data_name = 'ml-1m-test.txt'
    train_data_name = 'u2.base'
    test_data_name = 'u2.test'
    #precessing
    new_data_dir = './results/'+train_data_name
    if(os.path.isdir(new_data_dir)):
        print('dir exists')
    else:
        print('dir not exists, generate a new dir')
        os.mkdir(new_data_dir)
    if(os.path.isdir(new_data_dir+'/figures')):
        pass
    else:
        os.mkdir(new_data_dir+'/figures')
    (train_data,train_label,train_users,train_items)= loadData('../data/'+train_data_name)
    (test_data,test_label,test_users,test_items)=loadData('../data/'+test_data_name)
    v = DictVectorizer()
    x_train=v.fit_transform(train_data)
    x_test = v.fit_transform(test_data)
    
    if(train_data_name == 'ml-1m-train.txt'):
        num_attributes = 9940
    else:
        num_attributes = 2652
    print('dataset:'+train_data_name+'\n')
    print('num_attributes:'+str(num_attributes))
    performance_cross_validation(train_data_name,x_train,train_label,x_test,test_label,num_attributes= num_attributes)
    #performance_with_k(train_data_name,x_train,train_label,x_test,test_label)


