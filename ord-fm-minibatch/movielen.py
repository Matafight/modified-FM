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


def performance_with_k(dataname,x_train,y_train,x_test,y_test,num_attributes):
    candidate_k = [20,40,60,80,100,120]
    reg_1 = 0.00001
    reg_2 = 0.00001
    cur_time = time.strftime('%m-%d-%H-%M',time.localtime(time.time()))
    file_varing_k = open('./results/'+dataname+'/performance_varying_k_'+cur_time+'.txt','w')
    file_varing_k.write('reg_1:'+str(reg_1)+'\n')
    file_varing_k.write('reg_2:'+str(reg_2)+'\n')
    for num_factors in candidate_k:
        print('k:'+str(num_factors))
        file_varing_k.write('k:'+str(num_factors))
        fm = pylibfm.FM(num_factors = num_factors,num_iter = 1000,verbose = False,task = 'regression',initial_learning_rate=0.001,learning_rate_schedule='optimal',dataname = dataname,reg_1 = reg_1,reg_2 = reg_2)
        fm.fit(x_train,y_train,x_test,y_test,num_attributes)
        pre_label = fm.predict(x_test)
        diff = 0.5*np.sum((pre_label-y_test)**2)/y_test.size
        file_varing_k.write(str(diff)+'\n')
    file_varing_k.close()

def performance_cross_validation(data_name,x_train,y_train,x_test,y_test,num_attributes):
    num_factors = 10
    mycv = mcv.cross_val_regularization(train_data = x_train,train_label = train_label,num_factors = num_factors,num_attributes = num_attributes, dataname = data_name)
    best_reg = mycv.sele_para()
    #best_reg = [1,1]
    fm = pylibfm.FM(num_factors = num_factors,num_iter=1000,verbose = True,task="regression",initial_learning_rate=0.001,learning_rate_schedule="optimal",dataname=data_name,reg_1 = best_reg[0], reg_2 = best_reg[1])

    fm.fit(x_train,y_train,x_test,y_test,num_attributes)
    pre_label = fm.predict(x_test)

    diff = 0.5*np.sum((pre_label-y_test)**2)/y_test.size
    fh = open('./results/'+data_name+'/final_'+data_name,'a')
    fh.write("--test--RMSE---"+str(diff)+'\n')
    print(diff)

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
    print(num_attributes)
    performance_cross_validation(train_data_name,x_train,train_label,x_test,test_label,num_attributes= num_attributes)
    #performance_with_k(train_data_name,x_train,train_label,x_test,test_label)


