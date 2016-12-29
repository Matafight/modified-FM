#_*_ coding:utf-8 _*_

import numpy as np
import sys
from libc.math cimport exp,log,pow,sqrt
import time
import random
import matplotlib.pyplot as plt
cimport numpy as np
cimport cython

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
    num_factors : int
    num_attributes : int
    n_iter = int
    k0 : int
    k1 : int
    w0 : double
    t : double
    t0 : double
    #what's this used for?
    l : double
    power_t :double
    min_target : double
    max_target : double
    eta0 : double
    learning_rate_schedule : int
    shuffle_training : int
    task : int
    seed : int
    verbose: int
    """
    cdef double w0
    cdef np.ndarray w
    cdef np.ndarray v
    cdef double early_stop_w0
    cdef np.ndarray early_stop_w
    cdef np.ndarray early_stop_v
    cdef int num_factors
    cdef int num_attributes
    cdef int n_iter
    cdef int k0
    cdef int k1

    #why use the different DOUBLE type from w0
    cdef DOUBLE t
    cdef DOUBLE t0
    cdef DOUBLE l
    cdef DOUBLE power_t
    cdef DOUBLE min_target
    cdef DOUBLE max_target
    cdef np.ndarray sum_
    cdef np.ndarray sum_sqr
    cdef int task
    cdef int learning_rate_schedule
    cdef double learning_rate
    cdef double init_learning_rate
    cdef int shuffle_training
    cdef int seed
    cdef int verbose

    cdef DOUBLE  reg_0
    cdef DOUBLE reg_1
    cdef DOUBLE reg_2

    cdef np.ndarray grad_w
    cdef np.ndarray grad_v
    cdef str dataname
    cdef DOUBLE sumloss
    cdef int count # what for?
    cdef CSRDataset x_test
    cdef np.ndarray y_test
    def __init__(self,
                  np.ndarray[DOUBLE,ndim=1,mode='c'] w,
                  np.ndarray[DOUBLE, ndim=2,mode='c'] v,
                  int num_factors,
                  int num_attributes,
                  int n_iter,
                  int k0,
                  int k1,
                  double w0,
                  double t,
                  double t0,
                  double power_t,
                  double min_target,
                  double max_target,
                  double eta0,
                  int learning_rate_schedule,
                  int shuffle_training,
                  int task,
                  int seed,
                  int verbose,
                  dataname,
                  double reg_1,
                  double reg_2,
                  CSRDataset x_test,
                  np.ndarray[DOUBLE,ndim=1, mode  = 'c'] y_test):
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
        self.grad_w = np.zeros(self.num_attributes)
        self.grad_v = np.zeros((self.num_factors,self.num_attributes))
        #if(verbose==False):

        self.x_test = x_test
        self.y_test = y_test

    cdef _predict_instance(self, DOUBLE * x_data_ptr, INTEGER * x_ind_ptr,int xnnz):
        #helper variable
        cdef DOUBLE result = 0.0
        cdef int feature
        cdef unsigned int i = 0
        cdef unsigned int f = 0
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
        cdef unsigned int i =0
        cdef Py_ssize_t n_samples = dataset.n_samples
        cdef DOUBLE * x_data_ptr = NULL
        cdef INTEGER * x_ind_ptr = NULL
        cdef int xnnz
        cdef DOUBLE sample_weight = 1.0
        cdef DOUBLE y_placeholder
        cdef DOUBLE p = 0.0

        cdef np.ndarray[DOUBLE, ndim = 1,mode='c'] return_preds = np.zeros(n_samples)

        for i in range(n_samples):
            dataset.next(&x_data_ptr,&x_ind_ptr,&xnnz,&y_placeholder,&sample_weight)
            p = self._predict_instance(x_data_ptr,x_ind_ptr,xnnz)

            return_preds[i] = p
        return return_preds

    cdef _sgd_theta_step(self,DOUBLE * x_data_ptr, INTEGER * x_ind_ptr,int xnnz,DOUBLE y):
        cdef DOUBLE mult = 0.0
        cdef DOUBLE p
        cdef int feature
        cdef unsigned int i=0
        cdef unsigned int f=0
        cdef DOUBLE d
        cdef DOUBLE grad_0

        cdef DOUBLE w0 = self.w0
        cdef np.ndarray[DOUBLE, ndim =1 ,mode='c']  w = self.w
        cdef np.ndarray[DOUBLE, ndim = 2,mode ='c'] v = self.v
        cdef np.ndarray[DOUBLE, ndim = 1,mode='c'] grad_w = self.grad_w
        cdef np.ndarray[DOUBLE,ndim = 2,mode='c'] grad_v = self.grad_v
        cdef DOUBLE learning_rate = self.learning_rate
        cdef DOUBLE reg_0 = self.reg_0
        cdef DOUBLE reg_1 = self.reg_1
        cdef DOUBLE reg_2 = self.reg_2

        # have already calculate sum_
        p = self._predict_instance(x_data_ptr,x_ind_ptr,xnnz)

        # regression task
        p = min(self.max_target,p)
        p = max(self.min_target,p)
        mult = 2*(p-y)

        #set learning schedule
        self.learning_rate = 1.0/(self.t + self.t0)

        self.sumloss += _squared_loss(p,y)
        #update global bias
        if self.k0 > 0:
            grad_0 = mult
            w0 -= learning_rate*(grad_0 + 2*reg_0*w0)

        if self.k1 > 0:
            for i in range(xnnz):
                feature = x_ind_ptr[i]
                grad_w[feature]= mult*x_data_ptr[i]

                w[feature] -= learning_rate*(grad_w[feature]+ 2*reg_1*w[feature])

        for f in range(self.num_factors):
            for i in range(xnnz):
                feature = x_ind_ptr[i]
                grad_v[f,feature] = mult*x_data_ptr[i]*(self.sum_[f]-x_data_ptr[i]*v[f,feature])
                v[f,feature] -= learning_rate*(grad_v[f,feature] + 2*reg_2*v[f,feature])

        self.learning_rate = learning_rate
        self.w0 = w0
        self.w = w
        self.v = v
        self.grad_w = grad_w
        self.grad_v = grad_v
        self.t +=1
        self.count +=1

    def fit(self, CSRDataset dataset):
        cdef Py_ssize_t n_samples = dataset.n_samples
        cdef DOUBLE * x_data_ptr = NULL
        cdef INTEGER * x_ind_ptr = NULL

        #helper variables
        cdef int itercount=0
        cdef int xnnz
        cdef DOUBLE y = 0.0
        cdef unsigned int count =0
        cdef unsigned int epoch = 0
        cdef unsigned int i =0
        cdef DOUBLE sample_weight = 1.0
        cdef DOUBLE min_early_stop = sys.maxint
        cdef unsigned int count_early_stop = 0

        num_sample_iter = 100
        if self.verbose > 0:
            cur_time = time.strftime('%m-%d-%H-%M',time.localtime(time.time()))
            fh = open('./results/'+self.dataname+'/train_'+cur_time+'_'+str(self.reg_1)+'__'+str(self.reg_2)+'_k_'+str(self.num_factors)+'_.txt','w')
            fhtest = open('./results/'+self.dataname+'/test_'+cur_time+'_'+str(self.reg_1)+'__'+str(self.reg_2)+'_k_'+str(self.num_factors)+'_.txt','w')
            fhtest.write('reg_1:'+str(self.reg_1)+'\n')
            fhtest.write('reg_2:'+str(self.reg_2)+'\n')
            fhtest.write('num_factors:'+str(self.num_factors)+'\n')
            fhtest.write('init_learning_rate:'+str(self.init_learning_rate)+'\n')
            fhtest.write('num_sample_iter:'+str(num_sample_iter)+'\n')
            training_errors = []
            testing_errors = []
        for epoch in range(self.n_iter):
            self.count = 0
            self.sumloss = 0

            if self.shuffle_training:
                dataset.shuffle(self.seed)

            selected_list = random.sample(range(n_samples),num_sample_iter)

            for i in selected_list:
                dataset.data_index(&x_data_ptr, &x_ind_ptr,&xnnz,&y,&sample_weight,i)
                self._sgd_theta_step(x_data_ptr,x_ind_ptr,xnnz,y)
            if self.verbose > 0:
                if(itercount % 10 ==0):
                    strtemp = "Training MSE--"+str(self.sumloss/self.count)+"\n"
                    print(strtemp)
                    fh.write(str(self.sumloss/self.count)+'\n')
                    training_errors.append(self.sumloss/self.count)
                    iter_error = 0.0
                    pre_test = self._predict(self.x_test)
                    iter_error = 0.5*np.sum((pre_test-self.y_test)**2)/self.y_test.shape[0]
                    print("=======test_error===="+str(iter_error))
                    fhtest.write(str(iter_error)+'\n')
                    testing_errors.append(iter_error)
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
                    print('-----EARLY-STOPPING---')
                    self.w0 = self.early_stop_w0
                    self.w = self.early_stop_w
                    self.v = self.early_stop_v
                    break

            itercount +=1
        if(self.verbose>0):
            self.draw_line(training_errors,testing_errors,cur_time)
            fh.close()
            fhtest.close()

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
    cdef int current_index
    cdef int stride
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
        X_indptr : ndarray, dtype=np.int32, ndim=1, mode='c'
            The index pointer array of the CSR matrix; a one-dimensional
            c-continuous numpy array of dtype np.int32.
        X_indices : ndarray, dtype=np.int32, ndim=1, mode='c'
            The column indices array of the CSR matrix; a one-dimensional
            c-continuous numpy array of dtype np.int32.
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
                   int *nnz, DOUBLE *y, DOUBLE *sample_weight):
        cdef int current_index = self.current_index
        if current_index >= (self.n_samples - 1):
            current_index = -1

        current_index += 1
        cdef int sample_idx = self.index_data_ptr[current_index]
        cdef int offset = self.X_indptr_ptr[sample_idx]
        y[0] = self.Y_data_ptr[sample_idx]
        x_data_ptr[0] = self.X_data_ptr + offset
        x_ind_ptr[0] = self.X_indices_ptr + offset
        nnz[0] = self.X_indptr_ptr[sample_idx + 1] - offset
        #this is wrong when the num of data increases to 1 million
        #sample_weight[0] = self.sample_weight_data[sample_idx]

        self.current_index = current_index
    cdef void data_index(self,DOUBLE **x_data_ptr,INTEGER ** x_ind_ptr, int * nnz, DOUBLE *y, DOUBLE * sample_weight,INTEGER new_index):
        cdef int sample_idx = self.index_data_ptr[new_index]
        cdef int offset = self.X_indptr_ptr[sample_idx]
        y[0] = self.Y_data_ptr[sample_idx]
        x_data_ptr[0] = self.X_data_ptr + offset
        x_ind_ptr[0] = self.X_indices_ptr + offset
        nnz[0] = self.X_indptr_ptr[sample_idx + 1] - offset

        #sample_weight[0] = self.sample_weight_data[sample_idx]
    cdef void shuffle(self, seed):
        np.random.RandomState(seed).shuffle(self.index)
