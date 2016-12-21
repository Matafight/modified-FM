#_*_ coding:utf-8_*_
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from higher_fm import FM


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
    train_data_name = 'u2.base'
    test_data_name = 'u2.test'
    (train_data,train_label,train_users,train_items)= loadData('../data/'+train_data_name)
    (test_data,test_label,test_users,test_items)=loadData('../data/'+test_data_name)
    v = DictVectorizer()
    x_train=v.fit_transform(train_data)
    x_test = v.fit_transform(test_data)

    num_attributes = x_train.shape[1]
    num_factors = 20
    #初始化w,v_p,v_q
    w = np.zeros(num_attributes)
    v_p = np.zeros((num_attributes+1,num_factors+1))
    v_q = np.zeros((num_attributes+1,num_factors+1))
    myfm = FM(w=w,v_p=v_p,v_q=v_q,n_iter = 500,num_factors=num_factors,num_attributes = num_attributes,reg_1=0.001,reg_2=0.001)
    myfm.fit(x_train,train_label,x_test,test_label)
