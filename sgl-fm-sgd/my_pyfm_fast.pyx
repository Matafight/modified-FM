#_*_ coding:utf-8 _*_
 
import numpy as np
import sys
from libc.math cimport exp,log,pow,sqrt
import time
import random
cimport numpy as np
cimport cython
import matplotlib.pyplot as plt

np.import_array()

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INTEGER



cdef class FM_fast(object):
    cdef double w0
    cdef np.ndarray w
    cdef np.ndarray v
    cdef double early_stop_w0
    cdef np.ndarray early_stop_w
    cdef np.ndarray early_stop_v
    cdef int num_factors
    cdef int num_attributes
    cdef int n_iter

    cdef DOUBLE t
    cdef DOUBLE t0
    cdef DOUBLE min_target
    cdef DOUBLE max_target
    cdef np.ndarray sum_
    cdef np.ndarray sum_sqr
    cdef int task
    cdef double learning_rate
    cdef double init_learning_rate 
        
    cdef int shuffle_training
    cdef int seed
    cdef int verbose
    cdef int L_1
    cdef int L_21
    cdef int if_pd
    cdef int mini_batch

    cdef DOUBLE  reg_0
    cdef DOUBLE reg_1
    cdef DOUBLE reg_2
    cdef DOUBLE grad_w0
    cdef np.ndarray grad_w
    cdef np.ndarray grad_v
    cdef str path_detail
    cdef str method_name
    cdef DOUBLE sumloss
    cdef int count 
    cdef CSRDataset x_test
    cdef np.ndarray y_test
    cdef CSRDataset x_valid
    cdef np.ndarray y_valid
    def __init__(self,
                  np.ndarray[DOUBLE,ndim=1,mode='c'] w,
                  np.ndarray[DOUBLE, ndim=2,mode='c'] v,
                  int num_factors,
                  int num_attributes,
                  int n_iter,
                  double w0,
                  double t,
                  double t0,
                  double min_target,
                  double max_target,
                  double eta0,
                  int task,
                  int seed,
                  int verbose,
                  int L_1,
                  int L_21,
                  path_detail,
                  method_name,
                  double reg_1,
                  double reg_2,
                  CSRDataset x_test,
                  np.ndarray[DOUBLE,ndim=1, mode = 'c'] y_test,
                  CSRDataset x_valid,
                  np.ndarray[DOUBLE,ndim = 1, mode = 'c'] y_valid,
                  int if_pd,
                  int mini_batch):
        self.w0 = w0
        self.w = w
        self.v = v
        self.early_stop_w0 = w0
        self.early_stop_w = w
        self.early_stop_v = v
        self.num_factors = num_factors
        self.num_attributes = num_attributes
        self.n_iter = n_iter
        self.t = 1
        self.t0  = 1
        self.learning_rate = eta0
        self.init_learning_rate = eta0
        self.min_target = min_target
        self.max_target = max_target
        self.sum_ = np.zeros(self.num_factors)
        self.sum_sqr= np.zeros(self.num_factors)
        self.task = task
        self.seed = seed
        self.verbose = verbose
        self.L_1 = L_1
        self.L_21 = L_21
        self.reg_0 = 0.0
        if(if_pd > 0):
            self.reg_1 = reg_1
            self.reg_2 = reg_2
        else:
            self.reg_1 = reg_1*reg_2
            self.reg_2 = (1-reg_1)*reg_2*np.sqrt(num_factors)
        self.sumloss=0.0
        self.count = 0
        self.path_detail = path_detail
        self.method_name = method_name
        self.grad_w = np.zeros(self.num_attributes)
        self.grad_v = np.zeros((self.num_factors,self.num_attributes))
        self.x_test = x_test
        self.y_test = y_test
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.if_pd = if_pd
        self.mini_batch = mini_batch

    cdef _predict_instance(self, DOUBLE * x_data_ptr, INTEGER * x_ind_ptr,int xnnz):
        cdef DOUBLE result = 0.0
        cdef int feature
        cdef unsigned int i = 0
        cdef unsigned int f = 0
        cdef DOUBLE  d
        cdef DOUBLE  w0 = self.w0
        cdef np.ndarray[DOUBLE, ndim = 1,mode='c'] w = self.w
        cdef np.ndarray[DOUBLE, ndim = 2,mode='c'] v = self.v
        cdef np.ndarray[DOUBLE,ndim = 1,mode='c'] sum_ = np.zeros(self.num_factors)
        cdef np.ndarray[DOUBLE,ndim = 1,mode='c'] sum_sqr = np.zeros(self.num_factors)

        result +=w0
        for i in range(xnnz):
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
            dataset.next(&x_data_ptr,&x_ind_ptr,&xnnz,&y_placeholder)
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
        grad_0 = mult
        w0 -= learning_rate*(grad_0 + 2*reg_0*w0)
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

    cdef _update_grad_minibatch(self,DOUBLE * x_data_ptr, INTEGER * x_ind_ptr, INTEGER xnnz, DOUBLE y):
        cdef DOUBLE grad_w0 = self.grad_w0
        cdef np.ndarray[DOUBLE, ndim = 1, mode = 'c'] grad_w = self.grad_w
        cdef np.ndarray[DOUBLE, ndim = 2, mode = 'c'] grad_v = self.grad_v
        cdef np.ndarray[DOUBLE, ndim = 2, mode = 'c'] v = self.v
        cdef DOUBLE learning_rate = self.learning_rate
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



    cdef _average_and_update_ord(self,int num_samples):
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
        w = w - learning_rate*(grad_w+2*reg_1*w)
        v = v - learning_rate*(grad_v+2*reg_2*v)
       
        self.grad_w0 = 0.0
        self.grad_w = np.zeros((self.num_attributes))
        self.grad_v = np.zeros((self.num_factors,self.num_attributes))
        self.w0 = w0
        self.w = w
        self.v = v
        self.t += 1
        self.count +=1        

    cdef _average_and_update_sgl(self,INTEGER num_samples):
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
        if(self.L_1 > 0):
            absU = abs(U)
            U[absU <= reg_1] = 0
            ind = absU > reg_1
            U[ind] = (absU[ind] - reg_1)/absU[ind] * U[ind]

        #step 2, L2 norm on each column of U
        if(self.L_21 > 0):
            normU = np.linalg.norm(U,axis = 0)
            normU[normU <= reg_2] = 0
            ind = normU > reg_2
            normU[ind] = (normU[ind] - reg_2)/normU[ind]
            alpha = np.tile(normU,(self.num_factors+1,1))
            U = U*alpha
        w = U[0,:]
        v = U[1:,:]

        self.grad_w0 = 0.0
        self.grad_w = np.zeros((self.num_attributes))
        self.grad_v = np.zeros((self.num_factors,self.num_attributes))
        self.w0 = w0
        self.w = w
        self.v = v
        self.t += 1
        self.count +=1
    #using another optimization technology: FOBO and  Moreau-Yosida Regularization method to update the parameter
    cdef _sgd_FOBO_MYR_step(self,DOUBLE * x_data_ptr, INTEGER * x_ind_ptr, int xnnz, DOUBLE y):
        cdef DOUBLE w0 = self.w0
        cdef np.ndarray[DOUBLE, ndim = 1, mode = 'c'] w = self.w
        cdef np.ndarray[DOUBLE, ndim = 2, mode = 'c'] v = self.v
        cdef np.ndarray[DOUBLE, ndim = 1, mode = 'c'] grad_w = self.grad_w
        cdef np.ndarray[DOUBLE, ndim = 2, mode = 'c'] grad_v = self.grad_v
        cdef DOUBLE learning_rate = self.learning_rate
        cdef DOUBLE reg_1 = self.reg_1
        cdef DOUBLE reg_2 = self.reg_2

        p = self._predict_instance(x_data_ptr, x_ind_ptr, xnnz)
        
        #regression task
        p = min(self.max_target, p)
        p = max(self.min_target, p)
        mult = 2*(p-y)
        
        #set learning_rate
        self.learning_rate = 1.0/(self.t + self.t0)
        self.sumloss += _squared_loss(p,y)

        #update bias
        grad_0 = mult
        w0 -= learning_rate*(grad_0)
        for i in range(xnnz):
            feature = x_ind_ptr[i]
            grad_w[feature] = mult*x_data_ptr[i]
            w[feature] -= learning_rate*(grad_w[feature])
            
        for f in range(self.num_factors):
            for i in range(xnnz):
                feature = x_ind_ptr[i]
                grad_v[f,feature] = mult*x_data_ptr[i]*(self.sum_[f]-x_data_ptr[i]*v[f,feature]) 
                v[f,feature] -= learning_rate*(grad_v[f,feature])
                
        U = np.concatenate((w.reshape(1,self.num_attributes),v),axis = 0)
        # step 1 
        if(self.L_1 > 0):
            absU = abs(U)
            U[absU <= reg_1] = 0
            ind = absU > reg_1
            U[ind] = (absU[ind] - reg_1)/absU[ind] * U[ind]
        
        #step 2, L2 norm on each column of U
        if(self.L_21 > 0):
            normU = np.linalg.norm(U,axis = 0)
            normU[normU <= reg_2] = 0
            ind = normU > reg_2
            normU[ind] = (normU[ind] - reg_2)/normU[ind]
            alpha = np.tile(normU,(self.num_factors+1,1))
            U = U*alpha

        w = U[0,:]
        v = U[1:,:]

        self.learning_rate = learning_rate
        self.w0 = w0
        self.w = w
        self.v = v
        self.grad_w = grad_w
        self.grad_v = grad_v
        self.t += 1
        self.count += 1


        
    def return_sparsity(self):
        cdef np.ndarray[DOUBLE, ndim =1 ,mode='c']  w = self.w
        cdef np.ndarray[DOUBLE, ndim = 2,mode ='c'] v = self.v
        cdef np.ndarray[DOUBLE,ndim = 2,mode = 'c'] U = np.concatenate((np.reshape(w,(1,self.num_attributes)),v),axis = 0)
        zeros_ind_total = U==0
        total = (self.num_factors+1)*self.num_attributes
        per_total = np.sum(zeros_ind_total)/float(total)
        per_w = np.sum(w==0)/self.num_attributes
        return per_w,per_total
        
        

    def fit(self, CSRDataset dataset):
        cdef Py_ssize_t n_samples = dataset.n_samples
        cdef DOUBLE * x_data_ptr = NULL
        cdef INTEGER * x_ind_ptr = NULL

        cdef int itercount=0
        cdef int xnnz
        cdef DOUBLE y = 0.0

        cdef unsigned int count =0
        cdef unsigned int epoch = 0
        cdef unsigned int i =0
        cdef DOUBLE sample_weight = 1.0
        cdef DOUBLE min_early_stop = sys.maxint
        cdef unsigned int count_early_stop = 0
        #judge if ord or sgl is used
        if(self.L_1 <= 0 and self.L_21 <= 0):
            if_ord = True
        else:
            if_ord = False

        cur_time = time.strftime('%m-%d-%H-%M',time.localtime(time.time()))
        if(self.verbose > 0):
            num_sample_iter = n_samples
            fh = open(self.path_detail+'/Convergence_train_'+cur_time+'_'+str(self.reg_1)+'__'+str(self.reg_2)+'_'+'k_'+str(self.num_factors)+'_.txt','w')
            fhtest = open(self.path_detail+'/Convergence_test_'+cur_time+'_'+str(self.reg_1)+'__'+str(self.reg_2)+'_'+'k_'+str(self.num_factors)+'_.txt','w')
            fhvalid = open(self.path_detail+'/Convergence_valid_'+cur_time+'_'+str(self.reg_1)+'__'+str(self.reg_2)+'_'+'k_'+str(self.num_factors)+'_.txt','w')
            fhtest.write('reg_1:'+str(self.reg_1)+'\n')
            fhtest.write('reg_2:'+str(self.reg_2)+'\n')
            fhtest.write('num_factors:'+str(self.num_factors)+'\n')
            fhtest.write('init_learning_rate:'+str(self.init_learning_rate)+'\n')
            fhtest.write('num_sample_iter:'+str(num_sample_iter)+'\n')
            training_errors = []
            testing_errors = []
        else:
            num_sample_iter = 100
        for epoch in range(self.n_iter):
            if self.verbose >0 :
                pre_test = self._predict(self.x_test)
                pre_error = 0.5*np.sum((pre_test-self.y_test)**2)/self.y_test.shape[0]
                testing_errors.append(pre_error)
                fhtest.write(str(iter_error)+'\n')

            self.count = 0
            self.sumloss = 0
            selected_list = random.sample(range(n_samples),num_sample_iter)
            # 选择更新方式
            if(self.mini_batch > 0):
                for i in selected_list:
                    dataset.data_index(&x_data_ptr, &x_ind_ptr,&xnnz,&y,i)
                    self._update_grad_minibatch(x_data_ptr,x_ind_ptr,xnnz,y)
                if(if_ord == False):
                    self._average_and_update_sgl(num_sample_iter)
                else:
                    self._average_and_update_ord(num_sample_iter)
            else:
                for i in selected_list:
                    dataset.data_index(&x_data_ptr, &x_ind_ptr,&xnnz,&y,i)
                    if(if_ord == False):
                        self._sgd_FOBO_MYR_step(x_data_ptr,x_ind_ptr,xnnz,y)
                    else:
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
                    testing_errors.append(iter_error)
                    fhtest.write(str(iter_error)+'\n')
                    pre_valid = self._predict(self.x_valid)
                    valid_error = 0.5*np.sum((pre_valid-self.y_valid)**2)/self.y_valid.shape[0]
                    fhvalid.write(str(valid_error)+'\n')
            else:
                valid_error = 0.0
                pre_valid = self._predict(self.x_valid)
                valid_error = 0.5*np.sum((pre_valid-self.y_valid)**2)/self.y_valid.shape[0]
                count_early_stop += 1
                if(valid_error < min_early_stop):
                    min_early_stop = valid_error
                    self.early_stop_w0 = self.w0
                    self.early_stop_w = self.w
                    self.early_stop_v = self.v
                    count_early_stop = 0
                if(count_early_stop == 20):
                    print('----EARLY-STOPPING-')
                    self.w0 = self.early_stop_w0
                    self.w = self.early_stop_w
                    self.v = self.early_stop_v
                    break
            itercount +=1
        if(self.verbose <= 0 ):
            self.w0 = self.early_stop_w0
            self.w = self.early_stop_w
            self.v = self.early_stop_v
        if(self.verbose > 0):
            fh.close()
            fhtest.close()
            fhvalid.close()

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
                   int *nnz, DOUBLE *y):
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
        self.current_index = current_index
    cdef void data_index(self,DOUBLE **x_data_ptr,INTEGER ** x_ind_ptr, int * nnz, DOUBLE *y,INTEGER new_index):
        cdef int sample_idx = self.index_data_ptr[new_index]
        cdef int offset = self.X_indptr_ptr[sample_idx]
        y[0] = self.Y_data_ptr[sample_idx]
        x_data_ptr[0] = self.X_data_ptr + offset
        x_ind_ptr[0] = self.X_indices_ptr + offset
        nnz[0] = self.X_indptr_ptr[sample_idx + 1] - offset
    cdef void shuffle(self, seed):
        np.random.RandomState(seed).shuffle(self.index)
