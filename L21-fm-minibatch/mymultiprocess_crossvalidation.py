#_*_ coding:utf-8_*_
import numpy as np
import my_pyfmlib as pylibfm
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from multiprocessing import Process, Lock
class cross_val_regularization:

    def __init__(self,train_data,train_label,num_factors,num_attributes,dataname):
        self.reg_set = [0.00001,0.0001,0.001,0.01,0.1,1]
        self.length = len(self.reg_set)
        self.numfactors = num_factors
        self.num_attributes = num_attributes
        self.train_data = train_data
        self.train_label = train_label
        self.dataname = dataname
        self.reg_ret = np.zeros((len(self.reg_set),len(self.reg_set)))
    def sele_para(self):

        kf = KFold(np.shape(self.train_data)[0],n_folds = 5)
        count = 1
        threadlist=[]
        for train_index,valid_index in kf:
            x_train,x_test = self.train_data[train_index],self.train_data[valid_index]
            y_train,y_test = self.train_label[train_index],self.train_label[valid_index]
            if count >= 1:
                st = Process(target = self.sub_thread,args = (x_train,y_train,x_test,y_test,count))
                threadlist.append(st)
            count = count+1
        for st in threadlist:
            st.start()
        for st in threadlist:
            st.join()
        print("---------ALL subthread completed")


        #find the index of the minimum validationerror
        ind = np.argmin(self.reg_ret)
        best_reg_ind = np.unravel_index(ind,[self.length,self.length])
        # reg_1: best_reg[0],ret_2 : best_reg[1]
        best_reg = [self.reg_set[best_reg_ind[0]],self.reg_set[best_reg_ind[1]]]
        return best_reg
    def sub_thread(self,x_train,y_train,x_test,y_test,seq):
        print("---Runing---"+str(seq)+"---subthread")
        lock = Lock()
        for reg_1_cro in range(self.length):
            for reg_2_cro in range(self.length):
                fm = pylibfm.FM(num_factors = self.numfactors,num_iter=500,verbose = False,task="regression",initial_learning_rate=0.001,learning_rate_schedule="optimal",dataname=self.dataname,reg_1 = self.reg_set[reg_1_cro], reg_2 = self.reg_set[reg_2_cro])
                fm.fit(x_train,y_train,x_test,y_test,self.num_attributes)
                pre_label = fm.predict(x_test)
                diff = 0.5*np.sum((pre_label-y_test)**2)/y_test.size
                lock.acquire()
                try:
                    self.reg_ret[reg_1_cro,reg_2_cro] = self.reg_ret[reg_1_cro,reg_2_cro] + diff
                finally:
                    lock.release()
        print("---complete---"+str(seq)+"---subthread")
