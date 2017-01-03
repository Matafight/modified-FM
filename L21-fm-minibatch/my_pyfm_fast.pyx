#_*_ coding:utf-8 _*_

import numpy as np
import sys
from libc.math cimport exp,log,pow,sqrt
import time
import random
cimport numpy as np
cimport cython
import matplotlib.pyplot as plt
from libc.stdlib cimport malloc, free
np.import_array()

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INTEGER
# MODEL constants

#control learning_rate
DEF OPTIMAL = 0
DEF INVERSE_SCALING = 1

cdef class FM_fast(object):
    """
    parameters:
    w : np.ndarray[DOUBLE, ndim=1, mode='c']
    v : np.ndarray[DOUBLE, ndim= 2, mode='c']
    num_factors : INTEGER
    num_attributes : INTEGER
    n_iter = INTEGER
    k0 : INTEGER
    k1 : INTEGER
    w0 : double
    t : double
    t0 : double
    #what's this used for?
    l : double
    power_t :double
    min_target : double
    max_target : double
    eta0 : double
    learning_rate_schedule : INTEGER
    shuffle_training : INTEGER
    task : INTEGER
    seed : INTEGER
    verbose: INTEGER
    """
    cdef double w0
    cdef np.ndarray w
    cdef np.ndarray v
    cdef double early_stop_w0
    cdef np.ndarray early_stop_w
    cdef np.ndarray early_stop_v
    cdef INTEGER num_factors
    cdef INTEGER num_attributes
    cdef INTEGER n_iter
    cdef INTEGER  k0
    cdef INTEGER k1

    #why use the different DOUBLE type from w0
    cdef DOUBLE t
    cdef DOUBLE t0
    cdef DOUBLE l
    cdef DOUBLE power_t
    cdef DOUBLE min_target
    cdef DOUBLE max_target
    cdef np.ndarray sum_
    cdef np.ndarray sum_sqr
    cdef INTEGER task
    cdef INTEGER learning_rate_schedule
    cdef double learning_rate
    cdef double init_learning_rate 
        
    cdef INTEGER shuffle_training
    cdef INTEGER seed
    cdef INTEGER verbose
    
    cdef INTEGER ifall
    cdef DOUBLE  reg_0
    cdef DOUBLE reg_1
    cdef DOUBLE reg_2

    cdef DOUBLE grad_w0
    cdef np.ndarray grad_w
    cdef np.ndarray grad_v
    cdef np.ndarray U_v
    cdef np.ndarray U_w
    cdef DOUBLE U_w0
    cdef DOUBLE T_rda # global T for RDA algorithm
    cdef str dataname
    cdef DOUBLE sumloss
    cdef INTEGER count # what for?
    cdef CSRDataset x_test
    cdef np.ndarray y_test
    cdef DOUBLE gamma
    def __init__(self,
                  np.ndarray[DOUBLE,ndim=1,mode='c'] w,
                  np.ndarray[DOUBLE, ndim=2,mode='c'] v,
                  INTEGER num_factors,
                  INTEGER num_attributes,
                  INTEGER n_iter,
                  INTEGER k0,
                  INTEGER k1,
                  double w0,
                  double t,
                  double t0,
                  double power_t,
                  double min_target,
                  double max_target,
                  double eta0,
                  INTEGER learning_rate_schedule,
                  INTEGER shuffle_training,
                  INTEGER task,
                  INTEGER seed,
                  INTEGER verbose,
                  dataname,
                  double reg_1,
                  double reg_2,
                  double gamma,
                  CSRDataset x_test,
                  np.ndarray[DOUBLE,ndim=1, mode  = 'c'] y_test,
                  INTEGER ifall):
        self.w0 = w0
        self.w = w
        self.v = v
        self.early_stop_w0 = w0
        self.early_stop_w = w
        self.early_stop_v = v
        self.num_factors = num_factors
        self.num_attributes = num_attributes
        self.n_iter = n_iter
        self.k0 = k0
        self.k1 = k1
        self.t = 1
        self.t0  = 1
        self.learning_rate = eta0
        self.init_learning_rate = eta0
        self.power_t  = power_t
        self.min_target = min_target
        self.max_target = max_target
        self.sum_ = np.zeros(self.num_factors)
        self.sum_sqr= np.zeros(self.num_factors)
        self.task = task
        self.learning_rate_schedule = learning_rate_schedule
        self.shuffle_training = shuffle_training
        self.seed = seed
        self.verbose = verbose
        self.reg_0 = 0.0
        self.reg_1 = reg_1
        self.reg_2 = reg_2
        self.sumloss=0.0
        self.count = 0
        self.dataname = dataname
        self.grad_w0 = 0.0
        self.grad_w = np.zeros(self.num_attributes)
        self.grad_v = np.zeros((self.num_factors,self.num_attributes))
        self.U_w = np.zeros(self.num_attributes)
        self.U_v = np.zeros((self.num_factors,self.num_attributes))
        self.U_w0 = 1
        self.T_rda = 1.0
        self.gamma = gamma
        self.x_test = x_test
        self.y_test = y_test
        self.ifall = ifall

    cdef _predict_instance(self, DOUBLE * x_data_ptr, INTEGER * x_ind_ptr,INTEGER xnnz):
        #helper variable
        cdef DOUBLE result = 0.0
        cdef INTEGER feature
        cdef INTEGER i = 0
        cdef INTEGER f = 0
        cdef DOUBLE  d

        #map instance variables to local variables
        cdef DOUBLE  w0 = self.w0
        cdef np.ndarray[DOUBLE, ndim = 1,mode='c'] w = self.w
        cdef np.ndarray[DOUBLE, ndim = 2,mode='c'] v = self.v
        cdef np.ndarray[DOUBLE,ndim = 1,mode='c'] sum_ = np.zeros(self.num_factors)
        cdef np.ndarray[DOUBLE,ndim = 1,mode='c'] sum_sqr = np.zeros(self.num_factors)

        if self.k0 > 0:
            result +=w0
        if self.k1 > 0:
            for i in range(xnnz):
                #x is stored in CSR format
                feature = x_ind_ptr[i]
                result += w[feature]*x_data_ptr[i]

        for f in range(self.num_factors):
            sum_[f] = 0.0
            sum_sqr[f] = 0.0
            for i in range(xnnz):
                feature = x_ind_ptr[i]
                d = v[f,feature]*x_data_ptr[i]
                sum_[f] += d
                sum_sqr[f] += d*d
            result += 0.5*(sum_[f]*sum_[f]-sum_sqr[f])

        #pass sum to sgd_theta
        self.sum_ = sum_
        return result

    def _predict(self, CSRDataset dataset):
        cdef INTEGER i =0
        cdef Py_ssize_t n_samples = dataset.n_samples
        cdef DOUBLE * x_data_ptr = NULL
        cdef INTEGER * x_ind_ptr = NULL
        cdef INTEGER xnnz
        cdef DOUBLE y_placeholder
        cdef DOUBLE p = 0.0

        cdef np.ndarray[DOUBLE, ndim = 1,mode='c'] return_preds = np.zeros(n_samples)
        for i in range(n_samples):
            dataset.data_index(&x_data_ptr,&x_ind_ptr,&xnnz,&y_placeholder,i)
            p = self._predict_instance(x_data_ptr,x_ind_ptr,xnnz)
            return_preds[i] = p
        x_data_ptr = NULL
        x_ind_ptr = NULL
        return return_preds

    cdef _update_grad_minibatch(self,DOUBLE * x_data_ptr, INTEGER * x_ind_ptr, INTEGER xnnz, DOUBLE y):
        cdef DOUBLE grad_w0 = self.grad_w0
        cdef np.ndarray[DOUBLE, ndim = 1, mode = 'c'] grad_w = self.grad_w
        cdef np.ndarray[DOUBLE, ndim = 2, mode = 'c'] grad_v = self.grad_v
        cdef np.ndarray[DOUBLE, ndim = 2, mode = 'c'] v = self.v
        cdef DOUBLE learning_rate = self.learning_rate
        cdef DOUBLE reg_1 = self.reg_1
        cdef DOUBLE reg_2 = self.reg_2
        p = self._predict_instance(x_data_ptr, x_ind_ptr, xnnz)

        #regression task
        p = min(self.max_target, p)
        p = max(self.min_target, p)
        mult = 2*(p-y)
        
        #set learning_rate
        self.sumloss += _squared_loss(p,y)
        grad_w0 += mult
        for i in range(xnnz):
            feature = x_ind_ptr[i]
            grad_w[feature] += mult*x_data_ptr[i]
        
        for f in range(self.num_factors):
            for i in range(xnnz):
                feature = x_ind_ptr[i]
                grad_v[f,feature] += mult*x_data_ptr[i]*(self.sum_[f]-x_data_ptr[i]*v[f,feature]) 
        self.grad_w0 = grad_w0
        self.grad_w = grad_w
        self.grad_v = grad_v
        
    cdef _average_and_update(self,INTEGER num_samples):
        #update w0,w,v,learning_rate, set grad_w0,grad_w,grad_v to be zeros
        cdef DOUBLE w0 = self.w0
        cdef np.ndarray[DOUBLE, ndim = 1, mode = 'c'] w = self.w
        cdef np.ndarray[DOUBLE, ndim = 2, mode = 'c'] v = self.v
        cdef DOUBLE grad_w0 = self.grad_w0
        cdef np.ndarray[DOUBLE, ndim = 1, mode = 'c'] grad_w = self.grad_w
        cdef np.ndarray[DOUBLE, ndim = 2, mode = 'c'] grad_v = self.grad_v
        cdef DOUBLE learning_rate = self.learning_rate
        cdef DOUBLE mynum_samples = num_samples
        cdef DOUBLE reg_1 = self.reg_1
        cdef DOUBLE reg_2 = self.reg_2
        grad_w0 = grad_w0/mynum_samples
        grad_w = grad_w/mynum_samples
        grad_v = grad_v/mynum_samples
        learning_rate = 1.0/(self.t + self.t0)

        w0 = w0 - learning_rate*grad_w0
        w = w - learning_rate*grad_w
        v = v - learning_rate*grad_v

        U = np.concatenate((w.reshape(1,self.num_attributes),v),axis = 0)
        # step 1 
        '''absU = abs(U)
        U[absU <= reg_1] = 0
        ind = absU > reg_1
        U[ind] = (absU[ind] - reg_1)/absU[ind] * U[ind]'''

        #step 2, L2 norm on each column of U
        normU = np.linalg.norm(U,axis = 0)
        normU[normU <= reg_2] = 0
        ind = normU > reg_2
        normU[ind] = (normU[ind] - reg_2)/normU[ind]
        alpha = np.tile(normU,(self.num_factors+1,1))
        U = U*alpha
        w = U[0,:]
        #num_zero = np.sum(w==0)
        #zero_rato = float(num_zero)/self.num_attributes
        v = U[1:,:]

        self.grad_w0 = 0.0
        self.grad_w = np.zeros((self.num_attributes))
        self.grad_v = np.zeros((self.num_factors,self.num_attributes))
        self.w0 = w0
        self.w = w
        self.v = v
        self.t += 1
        self.count +=1


        
    def fit(self, CSRDataset dataset):

        cdef Py_ssize_t n_samples = dataset.n_samples
        cdef DOUBLE * x_data_ptr = NULL
        cdef INTEGER * x_ind_ptr = NULL

        #helper variables
        cdef INTEGER itercount=0
        cdef INTEGER xnnz
        cdef DOUBLE y = 0.0

        cdef   INTEGER count =0
        cdef   INTEGER epoch = 0
        cdef   INTEGER i =0
        cdef DOUBLE sample_weight = 1.0
        cdef DOUBLE min_early_stop = sys.maxint
        cdef   INTEGER count_early_stop = 0
        if self.ifall > 0:
            num_sample_iter = n_samples
        else:
            num_sample_iter  = 100
        cur_time = time.strftime('%m-%d-%H-%M',time.localtime(time.time()))
        if(self.verbose > 0):
            fh = open('./results/'+self.dataname+'/train_'+cur_time+'_'+str(self.reg_1)+'__'+str(self.reg_2)+'_'+'k_'+str(self.num_factors)+'_.txt','w')
            fhtest = open('./results/'+self.dataname+'/test_'+cur_time+'_'+str(self.reg_1)+'__'+str(self.reg_2)+'_'+'k_'+str(self.num_factors)+'_.txt','w')
            #在文件的开头简单介绍一下参数设置
            fhtest.write('reg_1:'+str(self.reg_1)+'\n')
            fhtest.write('reg_2:'+str(self.reg_2)+'\n')
            fhtest.write('num_factors:'+str(self.num_factors)+'\n')
            fhtest.write('init_learning_rate:'+str(self.init_learning_rate)+'\n')
            fhtest.write('num_sample_iter:'+str(num_sample_iter)+'\n')
            training_errors = []
            testing_errors = []
        for epoch in range(self.n_iter):
            if self.verbose >0 :
                pre_test = self._predict(self.x_test)
                pre_error = 0.5*np.sum((pre_test-self.y_test)**2)/self.y_test.shape[0]
                testing_errors.append(pre_error)

            self.count = 0
            self.sumloss = 0
            if self.shuffle_training:
                dataset.shuffle(self.seed)

            selected_list = random.sample(range(n_samples),num_sample_iter)
          
            for i in selected_list:
                dataset.data_index(&x_data_ptr, &x_ind_ptr,&xnnz,&y,i)
                #mini batch
                self._update_grad_minibatch(x_data_ptr,x_ind_ptr,xnnz,y)
            #average gradient
            #set self.w,w0,v to zeros
            self._average_and_update(num_sample_iter)
               
            if self.verbose > 0:
                if(itercount % 10 ==0):
                    strtemp = "Training MSE--"+str(self.sumloss/(self.count*num_sample_iter))+"\n"
                    print(strtemp)
                    fh.write(str(self.sumloss/(self.count*num_sample_iter))+'\n')
                    training_errors.append(self.sumloss/(self.count*num_sample_iter))
                    iter_error = 0.0
                    pre_test = self._predict(self.x_test)
                    iter_error = 0.5*np.sum((pre_test-self.y_test)**2)/self.y_test.shape[0]
                    print("=======test_error===="+str(iter_error))
                    testing_errors.append(iter_error)
                    fhtest.write(str(iter_error)+'\n')
            else:
                iter_error = 0.0
                pre_test = self._predict(self.x_test)
                iter_error = 0.5*np.sum((pre_test-self.y_test)**2)/self.y_test.shape[0]
                count_early_stop += 1
                if(iter_error < min_early_stop):
                    min_early_stop = iter_error
                    self.early_stop_w0 = self.w0
                    self.early_stop_w = self.w
                    self.early_stop_v = self.v
                    count_early_stop = 0
                if(count_early_stop == 50):
                    print('----EARLY-STOPPING-')
                    self.w0 = self.early_stop_w0
                    self.w = self.early_stop_w
                    self.v = self.early_stop_v
                    break

            itercount +=1
        
        if(self.verbose>0):
            fh.close()
            fhtest.close()
            self.draw_line(training_errors,testing_errors,cur_time)

    def draw_line(self,training_errors,testing_errors,cur_time):
        lentrain = len(training_errors)
        lentest  = len(testing_errors)
        a,subp = plt.subplots(2)
        subp[0].plot(range(lentrain),training_errors)
        subp[1].plot(range(lentest),testing_errors)
        dataname = './results/'+self.dataname+'/figures/'+cur_time+'_reg_1_'+str(self.reg_1)+'_reg_2_'+str(self.reg_2)+'_k_'+str(self.num_factors)
        plt.savefig(dataname+'.png')
        plt.show()


cdef _squareq(np.ndarray a, INTEGER b):

    cdef DOUBLE ret = 0.0
    for i in range(b):
        ret += a[i]**2
    return ret



cdef _squared_loss(DOUBLE  p,DOUBLE y):
    return 0.5*(p-y)*(p-y)


cdef inline double max(double a, double b):
    return a if a >= b else b

cdef inline double min(double a, double b):
    return a if a <= b else b



cdef class CSRDataset:
    """An sklearn ``SequentialDataset`` backed by a scipy sparse CSR matrix. This is an ugly hack for the moment until I find the best way to link to sklearn. """

    cdef Py_ssize_t n_samples
    cdef INTEGER current_index
    cdef INTEGER stride
    cdef DOUBLE *X_data_ptr
    cdef INTEGER *X_indptr_ptr
    cdef INTEGER *X_indices_ptr
    cdef DOUBLE *Y_data_ptr
    cdef np.ndarray feature_indices
    cdef INTEGER *feature_indices_ptr
    cdef np.ndarray index
    cdef INTEGER *index_data_ptr
    cdef DOUBLE *sample_weight_data

    def __cinit__(self, np.ndarray[DOUBLE, ndim=1, mode='c'] X_data,
                  np.ndarray[INTEGER, ndim=1, mode='c'] X_indptr,
                  np.ndarray[INTEGER, ndim=1, mode='c'] X_indices,
                  np.ndarray[DOUBLE, ndim=1, mode='c'] Y,
                  np.ndarray[DOUBLE, ndim=1, mode='c'] sample_weight):
        """Dataset backed by a scipy sparse CSR matrix.

        The feature indices of ``x`` are given by x_ind_ptr[0:nnz].
        The corresponding feature values are given by
        x_data_ptr[0:nnz].

        Parameters
        ----------
        X_data : ndarray, dtype=np.float64, ndim=1, mode='c'
            The data array of the CSR matrix; a one-dimensional c-continuous
            numpy array of dtype np.float64.
        X_indptr : ndarray, dtype=np.INTEGER32, ndim=1, mode='c'
            The index poINTEGERer array of the CSR matrix; a one-dimensional
            c-continuous numpy array of dtype np.INTEGER32.
        X_indices : ndarray, dtype=np.INTEGER32, ndim=1, mode='c'
            The column indices array of the CSR matrix; a one-dimensional
            c-continuous numpy array of dtype np.INTEGER32.
        Y : ndarray, dtype=np.float64, ndim=1, mode='c'
            The target values; a one-dimensional c-continuous numpy array of
            dtype np.float64.
        sample_weights : ndarray, dtype=np.float64, ndim=1, mode='c'
            The weight of each sample; a one-dimensional c-continuous numpy
            array of dtype np.float64.
        """
        self.n_samples = Y.shape[0]
        self.current_index = -1
        self.X_data_ptr = <DOUBLE *>X_data.data
        self.X_indptr_ptr = <INTEGER *>X_indptr.data
        self.X_indices_ptr = <INTEGER *>X_indices.data
        self.Y_data_ptr = <DOUBLE *>Y.data
        self.sample_weight_data = <DOUBLE *> sample_weight.data
        # Use index array for fast shuffling
        cdef np.ndarray[INTEGER, ndim=1,
                        mode='c'] index = np.arange(0, self.n_samples,
                                                    dtype=np.int32)
        self.index = index
        self.index_data_ptr = <INTEGER *> index.data

    cdef void next(self, DOUBLE **x_data_ptr, INTEGER **x_ind_ptr,
                   INTEGER *nnz, DOUBLE *y, DOUBLE *sample_weight):
        cdef INTEGER current_index = self.current_index
        if current_index >= (self.n_samples - 1):
            current_index = -1

        current_index += 1
        
        cdef INTEGER sample_idx = self.index_data_ptr[current_index]
        cdef INTEGER offset = self.X_indptr_ptr[sample_idx]
        y[0] = self.Y_data_ptr[sample_idx]
        x_data_ptr[0] = self.X_data_ptr + offset
        x_ind_ptr[0] = self.X_indices_ptr + offset
        nnz[0] = self.X_indptr_ptr[sample_idx + 1] - offset
        sample_weight[0] = 0
        self.current_index = current_index

    cdef void data_index(self,DOUBLE **x_data_ptr,INTEGER ** x_ind_ptr, INTEGER * nnz, DOUBLE *y,INTEGER new_index):
        cdef INTEGER sample_idx = self.index_data_ptr[new_index]
        cdef INTEGER offset = self.X_indptr_ptr[sample_idx]
        y[0] = self.Y_data_ptr[sample_idx]
        x_data_ptr[0] = self.X_data_ptr + offset
        x_ind_ptr[0] = self.X_indices_ptr + offset
        nnz[0] = self.X_indptr_ptr[sample_idx + 1] - offset
    cdef void shuffle(self, seed):
        np.random.RandomState(seed).shuffle(self.index)