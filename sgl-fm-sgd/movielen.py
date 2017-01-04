#_*_ coding:utf-8_*_
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import my_pyfmlib as pylibfm
import mymultiprocess_crossvalidation as mcv
import os
import time
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

def performance_with_k(data_name,x_train,y_train,x_test,y_test,num_attributes,L_1,L_21,method):
    candidate_k = [20,40,60]
    cur_time = time.strftime('%m-%d-%H-%M',time.localtime(time.time()))
    file_varing_k = open('./results/'+data_name+'/'+method+'/performance_varying_k.txt','a')
    file_varing_k.write(cur_time+'\n')
    for num_factors in candidate_k:
        print('k:'+str(num_factors)+'\n') 
        print('start crossvalidation---')
        mycv = mcv.cross_val_regularization(train_data = x_train,train_label = train_label,num_factors = num_factors, dataname = data_name,num_attributes = num_attributes,L_1 = L_1,L_21 = L_21)
        best_reg = mycv.sele_para()
        reg_1 = best_reg[0]
        reg_2 = best_reg[1]
        file_varing_k.write('reg_1:'+str(reg_1)+'\n')
        file_varing_k.write('reg_2:'+str(reg_2)+'\n')
        file_varing_k.write('k:'+str(num_factors)+'\n')
        print('crossvalidation finished---')
        fm = pylibfm.FM(num_factors = num_factors,num_iter = 1000,verbose = False,L_1 = L_1,L_21 = L_21,task = 'regression',initial_learning_rate=0.001,dataname = data_name,reg_1 = reg_1,reg_2 = reg_2)
        fm.fit(x_train,y_train,x_test,y_test,num_attributes)
        pre_label = fm.predict(x_test,y_test)
        diff = 0.5*np.sum((pre_label-y_test)**2)/y_test.size
        file_varing_k.write(str(diff)+'\n')
    file_varing_k.close()


def sparsity_with_performance(data_name,x_train,y_train,x_test,y_test,num_attributes,L_1,L_21,method):
    #计算性能随着 alpha 的变化而 变化,alpha 的变化也对应着组稀疏和一范数的权重的变化,需要比较这三种方法随着alpha 的变化而引起的性能的变化，还要保存系数的稀疏度
    num_factors = 20
    alpha_set = [0.2,0.4]
    lambda_set = [0.00001]
    fh_sparsity_performance =  open('./results/'+data_name+'/'+method+'/sparsity_performance.txt','a')
    fh_sparsity_performance.write('---------start-----new------experiments'+'\n')
    fh_sparsity_performance.write('num_factors:'+str(num_factors)+'\n')
    for alpha in alpha_set:
        for mylambda in lambda_set:
            fh_sparsity_performance.write('\n'+'alpha:'+str(alpha)+'\n')
            fh_sparsity_performance.write('lambda:'+ str(mylambda)+'\n')
            fm = pylibfm.FM(num_factors = num_factors,num_iter=1000,verbose = False,L_1 = L_1,L_21 = L_21,task="regression",initial_learning_rate=0.001,dataname=data_name,reg_1 = alpha, reg_2 = mylambda)
            fm.fit(x_train,y_train,x_test,y_test,num_attributes)
            pre_label = fm.predict(x_test,y_test)
            diff = 0.5*np.sum((pre_label-y_test)**2)/y_test.size
            w_sparsity,total_sparsity = fm.return_sparsity()
            fh_sparsity_performance.write('w_sparsity:'+str(w_sparsity)+'\n')
            fh_sparsity_performance.write('totoal_sparsity:'+str(total_sparsity)+'\n')
            fh_sparsity_performance.write('rmse:'+str(diff)+'\n')
    fh_sparsity_performance.close()

def performance_cross_validation(dataname,x_train,y_train,x_test,y_test,num_attributes,L_1,L_21,method):
    num_factors = 10
    #mycv = mcv.cross_val_regularization(train_data = x_train,train_label = train_label,num_factors = num_factors,num_attributes=num_attributes, dataname = dataname,L_1 = L_1,L_21 = L_21)
    #best_reg = mycv.sele_para()
    best_reg = [0.10,0.0010]
    fm = pylibfm.FM(num_factors = num_factors,num_iter=100,verbose = True,L_1 = L_1,L_21 = L_21,task="regression",initial_learning_rate=0.001,dataname=dataname,method_name = method,reg_1 = best_reg[0], reg_2 = best_reg[1])
    fm.fit(x_train,y_train,x_test,y_test,num_attributes)
    pre_label = fm.predict(x_test,y_test)

    
if __name__=='__main__':
    #pre setting
    L_1 = True
    L_21 = True
    train_data_name = 'u2.base'
    test_data_name = 'u2.test'
    #train_data_name = 'ml-1m-train.txt'
    #test_data_name = 'ml-1m-test.txt'
    new_data_dir = './results/'+train_data_name
    if(not os.path.isdir(new_data_dir)):
        os.mkdir(new_data_dir)
    if(not os.path.isdir(new_data_dir+'/figures')):
        os.mkdir(new_data_dir + '/figures')
    if(L_1 and L_21):
        method = 'sgl'
        if(not os.path.isdir(new_data_dir+'/sgl')):
            os.mkdir(new_data_dir + '/sgl')
    elif(L_1):
        method = 'L1'
        if(not os.path.isdir(new_data_dir + '/L1')):
            os.mkdir(new_data_dir+'/L1')
    elif(L_21):
        method = 'L21'
        if(not os.path.isdir(new_data_dir+'/L21')):
            os.mkdir(new_data_dir+'/L21')

    (train_data,train_label,train_users,train_items)= loadData('../data/'+train_data_name)
    (test_data,test_label,test_users,test_items)=loadData('../data/'+test_data_name)
    v = DictVectorizer()
    x_train=v.fit_transform(train_data)
    x_test = v.fit_transform(test_data)

    if(train_data_name == 'ml-1m-train.txt'):
        num_attributes = 9940
    else:
        num_attributes = 2652


    print('method: '+ method)
    print('dataset:'+train_data_name)
    print('num_attributes:'+str(num_attributes))
    #performance_with_k(train_data_name,x_train,train_label,x_test,test_label,num_attributes,L_1,L_21,method)
    #sparsity_with_performance(train_data_name,x_train,train_label,x_test,test_label,num_attributes,L_1,L_21,method)
    performance_cross_validation(train_data_name,x_train,train_label,x_test,test_label,num_attributes,L_1,L_21,method)
   
