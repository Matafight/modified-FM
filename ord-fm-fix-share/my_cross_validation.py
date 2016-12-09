import numpy as np
import my_pyfmlib as pylibfm
from sklearn import cross_validation
from sklearn.cross_validation import KFold
class cross_val_regularization:

    def __init__(self,train_data,train_label,dataname):
        self.reg_set = [0.001,0.01,0.1]
        self.length = len(self.reg_set)
        self.numfactors = 10
        self.train_data = train_data
        self.train_label = train_label
        self.dataname = dataname
        self.reg_ret = np.zeros((len(self.reg_set),len(self.reg_set)))
    def sele_para(self):
        #x_train,x_test,y_train,y_test = cross_validation.train_test_split(self.train_data,self.train_label,test_size = 0.1)

        kf = KFold(np.shape(self.train_data)[0],n_folds = 5)
        count = 1
        for train_index,valid_index in kf:
            x_train,x_test = self.train_data[train_index],self.train_data[valid_index]
            y_train,y_test = self.train_label[train_index],self.train_label[valid_index]
            print("-----"+str(count)+" fold"+"--total 5 fold")
            count = count+1
            for reg_1_cro in range(self.length):
                for reg_2_cro in range(self.length):
                    fm = pylibfm.FM(num_factors = self.numfactors,num_iter=10,verbose = False,task="regression",initial_learning_rate=0.001,learning_rate_schedule="optimal",dataname=self.dataname,reg_1 = self.reg_set[reg_1_cro], reg_2 = self.reg_set[reg_2_cro])
                    fm.fit(x_train,y_train)
                    pre_label = fm.predict(x_test)
                    diff = 0.5*np.sum((pre_label-y_test)**2)/y_test.size
                    print("--In "+ str(count-1) + " fold  "+" reg_1 = "+str(self.reg_set[reg_1_cro]) + "reg_2 = "+str( self.reg_set[reg_2_cro]))
                    print("validation_error:---"+str(diff)+'\n')
                    self.reg_ret[reg_1_cro,reg_2_cro] = self.reg_ret[reg_1_cro,reg_2_cro] + diff
        #find the index of the minimum validationerror
        ind = np.argmin(self.reg_ret)
        best_reg = np.unravel_index(ind,[self.length,self.length])
        # reg_1: best_reg[0],ret_2 : best_reg[1]
        return best_reg
