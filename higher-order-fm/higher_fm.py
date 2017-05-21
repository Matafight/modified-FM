#_*_ coding:utf-8 _*_
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import random
import matplotlib.pyplot as plt#
from tqdm import tqdm
class FM():

    def __init__(self,verbose = True, num_order = 3, n_iter=100, num_factors=10, num_attributes=10, dataname="unknown", reg_1=0.01,reg_2=0.01):
        # w,v_p,v_q 需要随机初始化,v_p 和 v_q  的 size 是 (num_attributes+1,num_factors+1)
        self.verbose = verbose
        self.w0 = 1
        self.w = np.zeros(num_attributes)
        self.v_p = np.zeros((num_attributes+1,num_factors+1))
        self.v_q = np.zeros((num_attributes+1,num_factors+1))
        #随机初始化
        #(b-a)*np.random.random_sample(size)-1 can generate numbers in range(a,b)
        # self.w = 2*np.random.random_sample(num_attributes)-1
        # self.v_p = 2*np.random.random_sample((num_attributes+1,num_factors+1))-1
        # self.v_q = 2*np.random.random_sample((num_attributes+1,num_factors+1))-1
        
        self.num_factors = num_factors
        self.num_attributes  = num_attributes
        self.num_order = num_order
        self.dataname = dataname
        self.reg_1 = reg_1
        self.reg_2 = reg_2
        # 设定learning_rate
        self.learning_rate = 1
        self.t = 1
        self.t0 = 1
        self.sum_loss = 0.0
        #一阶梯度
        self.grad_w = np.zeros(num_attributes)
        #这里假定 二阶和三阶的 factors 是一样的
        self.grad_v_p = np.zeros((num_attributes+1,num_factors+1))
        self.grad_v_q = np.zeros((num_attributes+1,num_factors+1))

        self.DP_table_sec = np.zeros((self.num_factors,2+1,self.num_attributes+1))
        self.DP_table_thi = np.zeros((self.num_factors,num_order+1,self.num_attributes+1))
        self.DP_table_sec[:,0,:] = np.ones(num_attributes + 1)
        self.DP_table_thi[:,0,:] = np.ones(num_attributes + 1)

        self.grad_DP_table_sec = np.zeros((self.num_factors,2+1,self.num_attributes+1))
        self.grad_DP_table_thi = np.zeros((self.num_factors,num_order+1,self.num_attributes+1))
        self.grad_DP_table_sec[:,2,num_attributes]  = 1
        self.grad_DP_table_thi[:,num_order,num_attributes] = 1


        self.count = 0
        self.n_iter = n_iter
    def _predict_instance(self,x):
        # x is a CSR format one-dimension array
        result = 0.0
        result += self.w0
        w = self.w
        v_p = self.v_p
        v_q = self.v_q
        #STEP 1: 计算一阶的response
        for i in range(x.nnz):
            feature  = x.indices[i]
            result += w[feature]*x.data[i]
        #STEP 2: 计算二阶的response,这里还可以化简
        for i in range(self.num_factors):
            DPT = self.DP_table_sec[i,:,:]
            for t in range(1,2+1):
                lastnnz = 0
                for k in range(x.nnz):
                    feature = x.indices[k] + 1
                    DPT[t,lastnnz+1:feature] = DPT[t,lastnnz]
                    DPT[t,feature] = DPT[t,feature-1] + v_p[feature,i]*x.data[k]*DPT[t-1,feature-1]
                    lastnnz = feature

            self.DP_table_sec[i,:,:] = DPT
            result += DPT[2,self.num_attributes]
            
        #STEP 3：计算三阶的response
        for i in range(self.num_factors):
            DPT = self.DP_table_thi[i,:,:]
            for t in range(1,3+1):
                lastnnz = 0
                for k in range(x.nnz):
                    feature = x.indices[k] + 1
                    DPT[t,lastnnz+1:feature] = DPT[t,lastnnz]
                    DPT[t,feature] = DPT[t,feature-1] + v_q[feature,i]*x.data[k]*DPT[t-1,feature-1]
                    lastnnz = feature

            self.DP_table_thi[i,:,:] = DPT
            result += DPT[3,self.num_attributes]
        return result

    def _predict(self,data):
        n_samples = data.shape[0]
        ret_pred = np.zeros(n_samples)
        for i in range(n_samples):
            ret_pred[i] = self._predict_instance(data[i,:])
        return ret_pred

    def _grad_DP(self,x):
        d = self.num_attributes
        num_attributes = d
        num_factors = self.num_factors
        #we need to calculate the specific dp table for the new input x
        DP_table_sec  = self.DP_table_sec
        DP_table_thi = self.DP_table_thi
        #grad_v_p 也需要重新初始化
        grad_v_p = np.zeros((num_attributes+1,num_factors+1))
        grad_v_q = np.zeros((num_attributes+1,num_factors+1))
        #必须重新初始化
        grad_DP_table_sec = np.zeros((self.num_factors,2+1,self.num_attributes+1))
        grad_DP_table_thi = np.zeros((self.num_factors,self.num_order+1,self.num_attributes+1))
        grad_DP_table_sec[:,2,num_attributes]  = 1
        grad_DP_table_thi[:,self.num_order,num_attributes] = 1
        v_p = self.v_p
        v_q = self.v_q
        # step 1 : 二阶
        #x.toarray()
        array_x = x.toarray()
        for i in range(self.num_factors):
            for t in range(2,0,-1):
                '''for j in range(d,t-1,-1):
                    if( j < d and t < 2):
                        grad_DP_table_sec[i,t,j] = grad_DP_table_sec[i,t+1,j+1]*v_p[j+1,i]*array_x[0][j]
                    if(j < d):
                        grad_DP_table_sec[i,t,j] = grad_DP_table_sec[i,t,j] + grad_DP_table_sec[i,t,j+1]'''

                lastnnz = d
                if(t == 2):
                        grad_DP_table_sec[i,t,t:d] = 1
                        continue
                for k in range(x.nnz-1,-1,-1):
                    feature = x.indices[k] + 1
                    if(feature >= t):
                        #feature must <= d
                        if(feature == d):
                            continue
                        # walk in two steps
                        #step1, feature:lastnnz
                        #step2, calculate feature-1
                        grad_DP_table_sec[i,t,feature:lastnnz]=0
                        grad_DP_table_sec[i,t,feature:lastnnz] = grad_DP_table_sec[i,t,lastnnz]
                        
                        grad_DP_table_sec[i,t,feature-1]=grad_DP_table_sec[i,t+1,feature]*v_p[feature,i]*x.data[k]
                        grad_DP_table_sec[i,t,feature-1] += grad_DP_table_sec[i,t,feature]
                        lastnnz = feature - 1
#                        if(feature < d and t < 2):
#                            grad_DP_table_sec[i,t,feature:lastnnz] = 0
#                            grad_DP_table_sec[i,t,feature-1] = grad_DP_table_sec[i,t+1,feature]*v_p[feature,i]*x.data[k]
#                        if(feature < d and t!=2):
#                            grad_DP_table_sec[i,t,feature:lastnnz] = grad_DP_table_sec[i,t,lastnnz]
#                            grad_DP_table_sec[i,t,feature-1] = grad_DP_table_sec[i,t,feature-1] + grad_DP_table_sec[i,t,feature]
#                            lastnnz = feature-1
                grad_DP_table_sec[i,t,t:lastnnz] = grad_DP_table_sec[i,t,lastnnz]

        for i in range(self.num_factors):
            '''for j in range(1,d+1):
                for t in range(1,2+1):
                    grad_v_p[j,i] += grad_DP_table_sec[i,t,j] * DP_table_sec[i,t-1,j-1]*array_x[0][j-1]'''
            for k in range(x.nnz):
                feature = x.indices[k]+1
                for t in range(1,2+1):
                    grad_v_p[feature,i] += grad_DP_table_sec[i,t,feature] *DP_table_sec[i,t-1,feature-1]*x.data[k]
        #step 2:三阶
        for i in range(self.num_factors):
            for t in range(3,0,-1):
                lastnnz = d
                if(t==3):
                    grad_DP_table_thi[i,t,t:d] = 1
                    continue
                for k in range(x.nnz-1,-1,-1):
                    feature = x.indices[k] + 1
                    if(feature >= t):
                        if(feature == d):
                            continue
                        
                        grad_DP_table_thi[i,t,feature:lastnnz] = 0
                        grad_DP_table_thi[i,t,feature:lastnnz] = grad_DP_table_thi[i,t,lastnnz]

                        grad_DP_table_thi[i,t,feature-1]=grad_DP_table_thi[i,t+1,feature]*v_q[feature,i]*x.data[k]
                        grad_DP_table_thi[i,t,feature-1] += grad_DP_table_thi[i,t,feature]
                        lastnnz = feature - 1
                grad_DP_table_thi[i,t,t:lastnnz] = grad_DP_table_thi[i,t,lastnnz]
        for i in range(self.num_factors):
            for k in range(x.nnz):
                feature = x.indices[k] + 1
                for t in range(1,3+1):
                    grad_v_q[feature,i] += grad_DP_table_thi[i,t,feature] *DP_table_thi[i,t-1,feature-1]*x.data[k]

        self.grad_v_p = grad_v_p
        self.grad_v_q = grad_v_q



    def _sgd_theta_step(self,x,y):
        w0 = self.w0
        w = self.w
        v_p = self.v_p
        v_q = self.v_q
        grad_w = self.grad_w
        num_factors = self.num_factors
        #计算对应于这个x的DPtable中的数据
        p = self._predict_instance(x)

        mult = 2*(p-y)

        #set learning schedule
        #self.learning_rate = 1.0/((self.t + self.t0))
        self.learning_rate = 0.01
        self.sum_loss += _squared_loss(y,p)

        #bias
        grad_0 = mult
        # 没有对 w0 添加惩罚项
        w0 -= self.learning_rate*grad_0

        #更新一阶系数
        for i in range(x.nnz):
            feature = x.indices[i]
            grad_w[feature] = mult*x.data[i]
            w[feature] -= self.learning_rate*(grad_w[feature] + 2*self.reg_1*w[feature])
        #更新二阶系数,暂时将三阶的情况与二阶合并
        self._grad_DP(x)
        grad_v_p = self.grad_v_p
        grad_v_q = self.grad_v_q
        for i in range(1,num_factors+1):
            for j in range(x.nnz):
                feature = x.indices[j] + 1
                v_p[feature,i] = v_p[feature,i] - self.learning_rate*(mult*grad_v_p[feature,i]+2*self.reg_2*v_p[feature,i])
                v_q[feature,i] = v_q[feature,i] - self.learning_rate*(mult*grad_v_q[feature,i]+2*self.reg_2*v_q[feature,i])

        self.w0 = w0
        self.w = w
        self.v_p = v_p
        self.v_q = v_q
        self.t +=1
        self.count +=1



    #no validation data yet
    def fit(self,train_data, train_y, test_data,test_y):

        n_samples = train_data.shape[0]
        itercount = 0
        training_errors = []
        testing_errors = []
        if self.verbose == True:
            train_ret_name = './results/train_error_'+self.dataname+'.txt'
            test_ret_name = './results/test_error_'+self.dataname+'.txt'
        for epoch in range(self.n_iter):
            if self.verbose == True:
                print("-----EPOCH----:"+str(epoch))
            self.sum_loss = 0.0
            self.count = 0
            n_sample_sele = 10

            my_shuffle = np.random.permutation(n_samples)  
            for item in tqdm(my_shuffle):
#            for item in my_shuffle:
                self._sgd_theta_step(train_data[item,:],train_y[item])
            training_errors.append(np.sqrt(self.sum_loss/self.count))
            print(np.sqrt(self.sum_loss/self.count))
            if(self.verbose==True and itercount%10==0):
                pre_y = self._predict(test_data)
                test_error = np.sqrt(np.sum((pre_y-test_y)**2)/test_y.shape[0])
                testing_errors.append(test_error)
                print('testing error %f'%test_error)
            itercount +=1
        if self.verbose == True:
            #draw_line(training_errors,testing_errors,self.dataname)
            np.savetxt(train_ret_name,np.array(training_errors),fmt = '%10.5f')
            np.savetxt(test_ret_name,np.array(testing_errors),fmt = '%10.5f')

def draw_line(training_errors,testing_errors,dataname):
    lentrain = len(training_errors)
    lentest  = len(testing_errors)
    a,subp = plt.subplots(2)
    subp[0].plot(range(lentrain),training_errors)
    subp[1].plot(range(lentest),testing_errors)
    dataname = './results/figures/'+dataname
    plt.savefig(dataname+'.png')
    plt.show()



def _squared_loss(a,b):
    return (a-b)*(a-b)
