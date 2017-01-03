#_*_ coding:utf-8_*_
import numpy as np
import my_pyfmlib as pylibfm
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from multiprocessing import Process, Lock
class cross_val_regularization:

    def __init__(self,train_data,train_label,num_factors,num_attributes,dataname,L_1,L_21):
        self.reg_set_1 = [0.2,0.4]
        self.reg_set_2 = [0.0001,0.00001]
        self.length_1 = len(self.reg_set_1)
        self.length_2 = len(self.reg_set_2)
        self.numfactors = num_factors
        self.num_attributes = num_attributes
        self.train_data = train_data
        self.train_label = train_label
        self.dataname = dataname
        self.L_1 = L_1
        self.L_21 = L_21
        self.reg_ret = np.zeros((self.length_1,self.length_2))
    def sele_para(self):
        kf = KFold(np.shape(self.train_data)[0],n_folds = 5)
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
        best_reg_ind = np.unravel_index(ind,[self.length_1,self.length_2])
        # reg_1: best_reg[0],ret_2 : best_reg[1]
        print(self.reg_ret)
        print(best_reg_ind)
        best_reg = [self.reg_set_1[best_reg_ind[0]],self.reg_set_2[best_reg_ind[1]]]
        print(best_reg)
        return best_reg


    def linear_crossvalidation(self,x_train,y_train,x_test,y_test,seq):
        print('---running---'+str(seq)+'----subthread')
        for reg_1_cro in range(self.length_1):
            for reg_2_cro in range(self.length_2):
                fm = pylibfm.FM(num_factors = self.numfactors,num_iter=500,verbose = False,L_1 = self.L_1,L_21=self.L_21,task="regression",initial_learning_rate=0.001,dataname=self.dataname,reg_1 = self.reg_set_1[reg_1_cro], reg_2 = self.reg_set_2[reg_2_cro])
                fm.fit(x_train,y_train,x_test,y_test,self.num_attributes,ifall = False)
                pre_label = fm.predict(x_test,y_test)
                diff = 0.5*np.sum((pre_label-y_test)**2)/y_test.size
                self.reg_ret[reg_1_cro,reg_2_cro] = self.reg_ret[reg_1_cro,reg_2_cro] + diff
                print(self.reg_ret)
        print('----complete---'+str(seq)+'---subthread') 
