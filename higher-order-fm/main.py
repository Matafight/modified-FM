#_*_ coding:utf-8_*_
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from higher_fm import FM
import mymultiprocess_crossvalidation as my_cv

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
    train_data_name = 'u1.base'
    test_data_name = 'u1.test'
    (train_data,train_label,train_users,train_items)= loadData('../data/'+train_data_name)
    (test_data,test_label,test_users,test_items)=loadData('../data/'+test_data_name)
    
    train_size=  len(train_data)
    all_data = train_data + test_data
    v = DictVectorizer()
    all_data_vec = v.fit_transform(all_data)
    x_train = all_data_vec[0:train_size]
    x_test = all_data_vec[train_size:]
#    x_train=v.fit_transform(train_data)
#    x_test = v.transform(test_data)

    num_attributes = x_train.shape[1]
    num_factors = 10

    bestreg = [0.01,0.01]
    myfm = FM(verbose = True,n_iter = 200,num_factors=num_factors,num_attributes = num_attributes,dataname = train_data_name, reg_1=bestreg[0],reg_2=bestreg[0])
    myfm.fit(x_train,train_label,x_test,test_label)
