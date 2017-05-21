#_*_ coding:utf-8_*_
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from higher_fm import FM
import argparse
from hofmlib import HOFM
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

def choose_hyparmeter(x_train,train_label,x_test,test_label,path,num_factors):
    
    reg_1_set = [0.00001,0.00001,0.0001]
    reg_2_set = [0.00001,0.00001,0.0001]

    for reg_1 in reg_1_set:
        for reg_2 in reg_2_set:
            hofm_model = HOFM(reg_1 = reg_1,reg_2 = reg_2,num_factors = num_factors,num_iter = 100,path = path,num_order = 3,learning_rate = 0.001)
            hofm_model.fit(x_train,train_label,x_test,test_label) 

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataname', action= 'store', dest= 'short_data_name', help= 'enter the dataname', type = str)
    parser.add_argument('-algorithm', action= 'store', dest= 'alg_name',type= str)
    parser.add_argument('-num_factors',action='store',dest='num_factors',type=int)
    
    results_arg = parser.parse_args()
 
    sdn = results_arg.short_data_name
    num_factors = results_arg.num_factors
    import chardet 
    fencoding = chardet.detect(sdn)
    print fencoding
    if(sdn in 'train_Genedata'):
        training_names = ['train_Genedata.0','train_Genedata.1','train_Genedata.2','train_Genedata.4','train_Genedata.4']
        testing_names = ['test_Genedata.0','test_Genedata.1','test_Genedata.2','test_Genedata.3','test_Genedata.4']
    elif(sdn in 'u2.base'):
        #training_names=['u1.base','u2.base','u3.base','u4.base','u5.base']
        #testing_names=['u1.test','u2.test','u3.test','u4.test','u5.test']
        training_names=['u1.base']
        testing_names=['u1.test']
    elif(sdn in 'ml-1m-train'):
        #training_names = ['ml-1m-train-0.txt','ml-1m-train-1.txt','ml-1m-train-2.txt','ml-1m-train-3.txt','ml-1m-train-4.txt']
        #testing_names = ['ml-1m-test-0.txt','ml-1m-test-1.txt','ml-1m-test-2.txt','ml-1m-test-3.txt','ml-1m-test-4.txt']
        training_names = ['ml-1m-train-0.txt']
        testing_names=['ml-1m-test-0.txt']
    elif(sdn in 'lastfm_testing.txt'):
        training_names = ['lastfm_training.txt']
        testing_names = ['lastfm_testing.txt']
    else:
        #raise error here
        print('data not found!!!!!!!!!')

    for ind in range(len(training_names)):  
        train_data_name = training_names[ind]
        test_data_name = testing_names[ind]
        new_data_dir = './results/'+train_data_name
        
        if(not os.path.isdir(new_data_dir)):
            os.mkdir(new_data_dir)

        new_data_dir += '/'+str(num_factors)
        if(not os.path.isdir(new_data_dir)):
            os.mkdir(new_data_dir)
        path_detail = new_data_dir +'/'
       
        if('ml-1m' in train_data_name or 'base' in train_data_name):
            (train_data,train_label,train_users,train_items)= loadData('../data/'+train_data_name)
            (test_data,test_label,test_users,test_items)=loadData('../data/'+test_data_name)
            v = DictVectorizer()
            x_train=v.fit_transform(train_data)
            x_test = v.transform(test_data)

        elif('lastfm' in train_data_name):
            (train_data,train_label,train_artistes,train_tags_all) = loadData_lastfm('../data/'+train_data_name)
            (test_data,test_label,test_artistes,test_tags_all) = loadData_lastfm('../data/'+test_data_name)
            v = DictVectorizer()
            x_train=v.fit_transform(train_data)
            x_test = v.transform(test_data)
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
        print('dataset:'+train_data_name)
        
  
        choose_hyparmeter(x_train,train_label,x_test,test_label,path_detail,num_factors)

   
