#_*_ coding:utf-8_*_
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import my_pyfmlib as pylibfm
import mymultiprocess_crossvalidation as mcv
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

def performance_with_k(dataname,x_train,y_train,x_test,y_test):
    candidate_k = [20,40,60,80,100,120]

def performance_cross_validation(dataname,x_train,y_train,x_test,y_test):
    num_factors = 10
    #mycv = mcv.cross_val_regularization(train_data = x_train,train_label = train_label,num_factors = num_factors, dataname = train_data_name)
    #best_reg = mycv.sele_para()
    best_reg = [0.0010,0.0010]
    fm = pylibfm.FM(num_factors = num_factors,num_iter=1000,verbose = True,task="regression",initial_learning_rate=0.001,learning_rate_schedule="optimal",dataname=dataname,reg_1 = best_reg[0], reg_2 = best_reg[1],gamma = 5)

    fm.fit(x_train,y_train,x_test,y_test)
    pre_label = fm.predict(x_test)

    diff = 0.5*np.sum((pre_label-y_test)**2)/y_test.size
    fh = open('./results/'+dataname+'/final_'+dataname,'a')
    fh.write("--test--RMSE---"+str(diff)+'\n')
    print(diff)

if __name__=='__main__':
    train_data_name = 'u4.base'
    test_data_name = 'u4.test'
    new_data_dir = './results/'+train_data_name
    if(os.path.isdir(new_data_dir)):
        print('dir exists')
    else:
        os.mkdir(new_data_dir)
    if(not os.path.isdir(new_data_dir+'/figures')):
        os.mkdir(new_data_dir + '/figures')
    (train_data,train_label,train_users,train_items)= loadData('../data/'+train_data_name)
    (test_data,test_label,test_users,test_items)=loadData('../data/'+test_data_name)
    v = DictVectorizer()
    x_train=v.fit_transform(train_data)
    x_test = v.fit_transform(test_data)

    performance_cross_validation(train_data_name,x_train,train_label,x_test,test_label)
   
