#_*_ coding:utf-8_*_
import numpy as np
import my_pyfmlib as pylibfm
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from multiprocessing import Process, Lock
class cross_val_regularization:

    def __init__(self,train_data,train_label,num_factors,path_detail,num_attributes,L_1,L_21,if_pd,mini_batch):
        #self.reg_set_1 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

        self.if_pd = if_pd
        if(L_1 and L_21):
            self.reg_set_1 = [0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009]
            self.reg_set_2 = [0.000001,0.000002,0.000003,0.000004,0.000005,0.000006,0.000007,0.000008,0.000009]
        elif(L_1):

            self.reg_set_1 = [0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009]
            self.reg_set_2 = [0.1]
        elif(L_21):
            self.reg_set_1 = [0.1]
            self.reg_set_2 = [0.000001,0.000002,0.000003,0.000004,0.000005,0.000006,0.000007,0.000008,0.000009]
        else:
            self.reg_set_1 = [0.00001,0.0001,0.001,0.01,0.1,1]
            self.reg_set_2 = [0.00001,0.0001,0.001,0.01,0.1,1]
    
        self.length_1 = len(self.reg_set_1)
        self.length_2 = len(self.reg_set_2)
        self.numfactors = num_factors
        self.num_attributes = num_attributes


        self.path_detail = path_detail
        self.reg_ret = np.zeros((self.length_1,self.length_2))
        self.L_1 = L_1
        self.L_21 = L_21
        self.mini_batch = mini_batch

    def sele_para(self):
        kf = KFold(np.shape(self.train_data)[0],n_folds = 3)
        count = 1
        threadlist=[]
        for train_index,valid_index in kf:
            x_train,x_test = self.train_data[train_index],self.train_data[valid_index]
            y_train,y_test = self.train_label[train_index],self.train_label[valid_index]
            self.linear_crossvalidation(x_train,y_train,x_test,y_test,count)
            count += 1
        print("---------ALL subthread completed")

        #find the index of the minimum validationerror
        ind = np.argmin(self.reg_ret)
        print(self.reg_ret)
        best_reg_ind = np.unravel_index(ind,[self.length_1,self.length_2])
        print(best_reg_ind)
        # reg_1: best_reg[0],ret_2 : best_reg[1]
        best_reg = [self.reg_set_1[best_reg_ind[0]],self.reg_set_2[best_reg_ind[1]]]
        return best_reg
    
    def linear_crossvalidation(self,x_train,y_train,x_test,y_test,seq):
        print('---running---'+str(seq)+'----subthread')
        for reg_1_cro in range(self.length_1):
            for reg_2_cro in range(self.length_2):
                fm = pylibfm.FM(num_factors = self.numfactors,num_iter=200,verbose = False,L_1 = self.L_1,L_21=self.L_21,task="regression",initial_learning_rate=0.001,path_detail=self.path_detail,reg_1 = self.reg_set_1[reg_1_cro], reg_2 = self.reg_set_2[reg_2_cro],if_pd = self.if_pd, mini_batch = self.mini_batch)
                fm.fit(x_train,y_train,x_test,y_test,self.num_attributes)
                pre_label = fm.predict(x_test,y_test)
                diff = sqrt(np.sqrt(np.sum((pre_label-y_test)**2)/y_test.size))
                self.reg_ret[reg_1_cro,reg_2_cro] = self.reg_ret[reg_1_cro,reg_2_cro] + diff
        print('----complete---'+str(seq)+'---subthread') 
