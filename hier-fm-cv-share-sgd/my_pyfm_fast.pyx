#_*_ coding:utf-8 _*_

import numpy as np
import sys
from time import time
from libc.math cimport exp,log,pow,sqrt
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

    cdef int shuffle_training
    cdef int seed
    cdef int verbose

    cdef DOUBLE  reg_0
    cdef DOUBLE reg_1
    cdef DOUBLE reg_2

    cdef np.ndarray grad_w
    cdef np.ndarray grad_v
    
    cdef DOUBLE sumloss
    cdef int count # what for?

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
                  int verbose):
        self.w0 = w0
        self.w = w
        self.v = v
        self.num_factors = num_factors
        self.num_attributes = num_attributes
        self.n_iter = n_iter
        self.k0 = k0
        self.k1 = k1
        self.t = 1
        self.t0  = 1
        self.learning_rate = eta0
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
        self.reg_1 = 0.0
        self.reg_2 = 0.0
        self.sumloss=0.0
        self.count = 0
        
        self.grad_w = np.zeros(self.num_attributes)
        self.grad_v = np.zeros((self.num_factors,self.num_attributes))
    
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

        cdef np.ndarray[DOUBLE, ndim=1,mode='c'] normq = np.zeros(self.num_attributes)
        cdef np.ndarray[DOUBLE, ndim = 1,mode='c'] normv = np.zeros(self.num_attributes)
        cdef np.ndarray[DOUBLE,ndim = 1,mode='c'] squareq = np.zeros(self.num_attributes)
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

                squareq[feature] = _squareq(v[:,feature],self.num_factors)
                normq[feature]=sqrt(squareq[feature] + w[feature]**2)
                w[feature] -= learning_rate*(grad_w[feature]+reg_1*w[feature]/normq[feature])

        #update feature factor vectors

        for f in range(self.num_factors):
            for i in range(xnnz):
                feature = x_ind_ptr[i]
                grad_v[f,feature] = mult*x_data_ptr[i]*(self.sum_[f]-x_data_ptr[i]*v[f,feature])
                normv[feature]= sqrt(squareq[feature])
                v[f,feature] -= learning_rate*(grad_v[f,feature] + reg_1*v[f,feature]/normq[feature] + reg_2*v[f,feature]/normv[feature])



        #pass updated vars to other functions

        self.learning_rate = learning_rate
        self.w0 = w0
        self.w = w
        self.v = v
        self.grad_w = grad_w
        self.grad_v = grad_v
        self.t +=1
        self.count +=1
                

    def fit(self, CSRDataset dataset, CSRDataset validation_dataset):
        cdef Py_ssize_t n_samples = dataset.n_samples
        cdef Py_ssize_t n_validation_samples = validation_dataset.n_samples
        cdef DOUBLE * x_data_ptr = NULL
        cdef INTEGER * x_ind_ptr = NULL
        cdef DOUBLE * validation_x_data_ptr = NULL
        cdef DOUBLE * validation_x_ind_ptr =NULL

        #helper variables
        cdef int xnnz
        cdef DOUBLE y = 0.0
        cdef DOUBLE validation_y = 0.0
        cdef int validation_xnnz
        cdef unsigned int count =0
        cdef unsigned int epoch = 0
        cdef unsigned int i =0
        cdef DOUBLE sample_weight = 1.0
        cdef DOUBLE validation_sample_weight=1.0

        for epoch in range(self.n_iter):
            if self.verbose >0 :
                print("--Epoch %d" %(epoch + 1))
            self.count = 0
            self.sumloss = 0

            if self.shuffle_training:
                dataset.shuffle(self.seed)

            for i in range(n_samples):
                dataset.next(&x_data_ptr, & x_ind_ptr, &xnnz,&y,&sample_weight)
                self._sgd_theta_step(x_data_ptr,x_ind_ptr,xnnz,y)

                if(epoch > 0):
                    # lambda step 
                    pass
            if self.verbose > 0:
                print ("Training %s:%.5f"%("MSE",(self.sumloss/self.count)))


                

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
        #这个next 函数是用来干嘛的?
        #就是让下一个数据 指向引用参数
        #offset? 
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
        sample_weight[0] = self.sample_weight_data[sample_idx]

        self.current_index = current_index

    cdef void shuffle(self, seed):
        np.random.RandomState(seed).shuffle(self.index)