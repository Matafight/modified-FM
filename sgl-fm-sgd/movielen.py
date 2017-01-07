#_*_ coding:utf-8_*_
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import my_pyfmlib as pylibfm
import mymultiprocess_crossvalidation as mcv
import os
import time
from scipy import sparse
import argparse
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

def performance_with_k(x_train,y_train,x_test,y_test,x_valid,valid_label,num_attributes,L_1,L_21,path_detail,if_pd):
    candidate_k = [80,100,120]
    cur_time = time.strftime('%m-%d-%H-%M',time.localtime(time.time()))
    file_varing_k = open(path_detail + '/performance_varying_k.txt','a')
    file_varing_k.write(cur_time+'\n\n')
    for num_factors in candidate_k:
        print('k:'+str(num_factors)+'\n') 
        print('start crossvalidation---')
        mycv = mcv.cross_val_regularization(train_data = x_train,train_label = train_label,num_factors = num_factors, path_detail = path_detail,num_attributes = num_attributes,L_1 = L_1,L_21 = L_21,if_pd = if_pd)
        best_reg = mycv.sele_para()
        reg_1 = best_reg[0]
        reg_2 = best_reg[1]
        file_varing_k.write('reg_1:'+str(reg_1)+'\n')
        file_varing_k.write('reg_2:'+str(reg_2)+'\n')
        file_varing_k.write('k:'+str(num_factors)+'\n')
        print('crossvalidation finished---')
        fm = pylibfm.FM(num_factors = num_factors,num_iter = 1000,verbose = False,L_1 = L_1,L_21 = L_21,task = 'regression',initial_learning_rate=0.001,path_detail = path_detail,reg_1 = reg_1,reg_2 = reg_2,if_pd = if_pd)
        fm.fit(x_train,y_train,x_test,y_test,x_valid,valid_label,num_attributes)
        pre_label = fm.predict(x_test,y_test)
        diff = 0.5*np.sum((pre_label-y_test)**2)/y_test.size
        file_varing_k.write(str(diff)+'\n')
    file_varing_k.close()


def sparsity_with_performance(data_name,x_train,y_train,x_test,y_test,num_attributes,L_1,L_21,path_detail, if_pd):
    #计算性能随着 alpha 的变化而 变化,alpha 的变化也对应着组稀疏和一范数的权重的变化,需要比较这三种方法随着alpha 的变化而引起的性能的变化，还要保存系数的稀疏度
    num_factors = 20
    alpha_set = [0.2,0.4]
    lambda_set = [0.00001]
    fh_sparsity_performance =  open(path_detail + '/sparsity_performance.txt','a')
    fh_sparsity_performance.write('\n'+'---------start-----new------experiments'+'\n')
    fh_sparsity_performance.write('num_factors:'+str(num_factors)+'\n')
    for alpha in alpha_set:
        for mylambda in lambda_set:
            fh_sparsity_performance.write('\n'+'alpha:'+str(alpha)+'\n')
            fh_sparsity_performance.write('lambda:'+ str(mylambda)+'\n')
            fm = pylibfm.FM(num_factors = num_factors,num_iter=1000,verbose = False,L_1 = L_1,L_21 = L_21,task="regression",initial_learning_rate=0.001, path_detail = path_detail,reg_1 = alpha, reg_2 = mylambda, if_pd = if_pd)
            fm.fit(x_train,y_train,x_test,y_test,num_attributes)
            pre_label = fm.predict(x_test,y_test)
            diff = 0.5*np.sum((pre_label-y_test)**2)/y_test.size
            w_sparsity,total_sparsity = fm.return_sparsity()
            fh_sparsity_performance.write('w_sparsity:'+str(w_sparsity)+'\n')
            fh_sparsity_performance.write('totoal_sparsity:'+str(total_sparsity)+'\n')
            fh_sparsity_performance.write('mse:'+str(diff)+'\n')
    fh_sparsity_performance.close()

def performance_cross_validation(x_train,y_train,x_test,y_test,num_attributes,L_1,L_21,method,path_detail,if_pd):
    num_factors = 10
    #mycv = mcv.cross_val_regularization(train_data = x_train,train_label = train_label,num_factors = num_factors,num_attributes=num_attributes, path_detail  = path_detail,L_1 = L_1,L_21 = L_21, if_pd = if_pd)
    #best_reg = mycv.sele_para()
    best_reg = [0.10,0.0010]
    fm = pylibfm.FM(num_factors = num_factors,num_iter=100,verbose = True,L_1 = L_1,L_21 = L_21,task="regression",initial_learning_rate=0.001,path_detail = path_detail ,method_name = method,reg_1 = best_reg[0], reg_2 = best_reg[1], if_pd = if_pd)
    fm.fit(x_train,y_train,x_test,y_test,num_attributes)
    pre_label = fm.predict(x_test,y_test)

    
if __name__=='__main__':
    #pre setting
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataname', action= 'store', dest= 'short_data_name', help= 'enter the dataname', type = str)
    parser.add_argument('-algorithm', action= 'store', dest= 'alg_name',type= str)
    parser.add_argument('-pd',action = 'store_true',dest = 'if_para_diff',default = False)
    parser.add_argument('-pr',action = 'store_false',dest = 'if_para_diff',default = False)
    parser.add_argument('-pmz',action = 'store',dest = 'pmz',type = str )
    results_arg = parser.parse_args()
    if(results_arg.alg_name == 'sgl'):
        L_1 = True
        L_21 = True
    elif(results_arg.alg_name == 'L_1'):
        L_1 = True
        L_21 = False
    elif(results_arg.alg_name == 'L_21'):
        L_1 = False
        L_21 = True
    else:
        #raise error here
        pass
    sdn = results_arg.short_data_name
    if(sdn in 'train_Genedata'):
        training_names = ['train_Genedata.0','train_Genedata.1','train_Genedata.2','train_Genedata.4','train_Genedata.4']
        testing_names = ['test_Genedata.0','test_Genedata.1','test_Genedata.2','test_Genedata.3','test_Genedata.4']
    elif(sdn in 'u2.base'):
        training_names=['u1.base','u2.base','u3.base','u4.base','u5.base']
        testing_names=['u1.test','u2.test','u3.test','u4.test','u5.base']
    elif(sdn in 'ml-1m-train'):
        training_names = ['ml-1m-train.txt']
        testing_names = ['ml-1m-test.txt']
    else:
        #raise error here
        print('The data has not found!!!!!!!!!')
        
    if(results_arg.if_para_diff == True):
        print('we use different lambda_1 lambda_2')
        if_pd = True
    else:
        print('we use the relevant alpha lambda')
        if_pd = False

    for ind in range(len(training_names)):
        train_data_name = training_names[ind]
        test_data_name = testing_names[ind]
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
        if(if_pd == True):
            if(not os.path.isdir(new_data_dir +'/' + method +'/pd')):
                 os.mkdir(new_data_dir + '/'+ method +'/pd')
            path_detail = new_data_dir + '/'+ method +'/pd/'
        if(if_pd == False):
             if(not os.path.isdir(new_data_dir +'/'+ method +'/pr')):
                 os.mkdir(new_data_dir +'/' + method +'/pr')
             path_detail = new_data_dir +'/' + method +'/pr/'
        
        if('ml-1m' in train_data_name or 'base' in train_data_name):
            (train_data,train_label,train_users,train_items)= loadData('../data/'+train_data_name)
            (test_data,test_label,test_users,test_items)=loadData('../data/'+test_data_name)
            v = DictVectorizer()
            x_train=v.fit_transform(train_data)
            x_test = v.fit_transform(test_data)

            size_train = x_train.shape[0]
            size_valid = int(size_train*0.1)
            x_valid =  x_train[:size_valid,:]
            valid_label = train_label[:size_valid]
            x_train = x_train[size_valid:,:]
            train_label = train_label[size_valid:]
            if('ml-1m' in train_data_name):
                num_attributes = 9940
            elif('base' in train_data_name):
                num_attributes = 2652
        else:
            train_data = np.loadtxt('../data/'+train_data_name)
            test_data = np.loadtxt('../data/'+test_data_name)
            num_attributes = train_data.shape[1]
            x_train = train_data[:,0:num_attributes-1]
            x_train = sparse.csr_matrix(x_train)
            train_label = np.array(train_data[:,num_attributes-1])
            x_test = test_data[:,0:num_attributes-1]
            x_test = sparse.csr_matrix(x_test)
            test_label = np.array(test_data[:,num_attributes-1])

            size_train = x_train.shape[0]
            size_valid =int(size_train*0.1)
            x_valid = x_train[:size_valid,:]
            valid_label = train_label[:size_valid]
            x_train = x_train[size_valid:,:]
            train_label = train_label[size_valid:]

        print('method:'+ method)
        print('dataset:'+train_data_name)
        print('num_attributes:'+str(num_attributes))
        
       
        performance_with_k(x_train,train_label,x_test,test_label,x_valid,valid_label,num_attributes,L_1,L_21,path_detail,if_pd)
        #sparsity_with_performance(train_data_name,x_train,train_label,x_test,test_label,num_attributes,L_1,L_21, path_detail, if_pd)
        #performance_cross_validation(train_data_name,x_train,train_label,x_test,test_label,num_attributes,L_1,L_21,method)
       
