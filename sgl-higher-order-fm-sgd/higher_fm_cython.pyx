#_*_ coding:utf-8 _*_
import numpy as np
from libc.math cimport exp,log,pow,sqrt
cimport numpy as np
cimport cython
np.import_array()
from tqdm import tqdm
ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INTEGER


cdef class FM:
    cdef double w0
    cdef np.ndarray w
    cdef np.ndarray v_p
    cdef np.ndarray v_q
    cdef int num_factors
    cdef int num_attributes
    cdef int num_order
    cdef str dataname
    cdef DOUBLE reg_1
    cdef DOUBLE reg_2
    cdef double learning_rate
    cdef DOUBLE t
    cdef double t0
    cdef DOUBLE sum_loss

    cdef np.ndarray prev_grad_w
    cdef np.ndarray grad_w
    cdef np.ndarray grad_v_p
    cdef np.ndarray grad_v_q
    cdef np.ndarray prev_grad_v_p
    cdef np.ndarray prev_grad_v_q

    cdef np.ndarray DP_table_sec
    cdef np.ndarray DP_table_thi
    
    cdef np.ndarray grad_DP_table_sec
    cdef np.ndarray grad_DP_table_thi

    #only for adadelta algorithm
    cdef np.ndarray delta_grad_w
    cdef np.ndarray delta_grad_v_p
    cdef np.ndarray delta_squ_grad_w
    cdef np.ndarray delta_squ_grad_v_p

    #only for adam
    cdef np.ndarray adam_grad_w 
    cdef np.ndarray adam_grad_v_p 
    cdef np.ndarray adam_grad_v_q 

    cdef int count
    cdef int n_iter
    cdef str method
    cdef str path
    cdef CSRDataset x_test
    cdef np.ndarray y_test

    #for sparse group lasso 
    cdef DOUBLE total_l1
    cdef DOUBLE total_l21
    cdef np.ndarray cur_l1
    cdef np.ndarray cur_l21

    def __init__(self,
                 int num_order,
                 int n_iter,
                 int num_factors,
                 int num_attributes,
                 double min_target,
                 double max_target,
                 double reg_1,
                 double reg_2,
                 double learning_rate,
                 str method,
                 str path,
                 CSRDataset x_test,
                 np.ndarray[DOUBLE,ndim = 1,mode = 'c'] y_test):
        self.num_order = num_order
        self.n_iter = n_iter
        self.num_factors = num_factors
        self.num_attributes = num_attributes
        self.reg_1 = reg_1
        self.reg_2 = reg_2
        self.t = 1
        self.t0 =1
        self.w0 = 1

        self.w = np.zeros(self.num_attributes)
        self.v_p = np.zeros((self.num_attributes+1,self.num_factors+1))
        self.v_q = np.zeros((self.num_attributes+1,self.num_factors+1))

        self.learning_rate = learning_rate
        self.sum_loss = 0.0

        # all previous and current gradents
        self.prev_grad_w = np.zeros(self.num_attributes)
        self.grad_w = np.zeros(self.num_attributes)
        self.grad_v_p = np.zeros((self.num_attributes+1,self.num_factors+1))
        self.grad_v_q = np.zeros((self.num_attributes+1,self.num_factors+1))
        self.prev_grad_v_p = np.zeros((self.num_attributes+1,self.num_factors+1))
        self.prev_grad_v_q = np.zeros((self.num_attributes+1,self.num_factors+1))

        self.DP_table_sec = np.zeros((self.num_factors,2+1,self.num_attributes+1))
        self.DP_table_thi = np.zeros((self.num_factors,3+1,self.num_attributes+1))
        self.DP_table_sec[:,0,:] = np.ones(num_attributes + 1)
        self.DP_table_thi[:,0,:] = np.ones(num_attributes + 1)

        self.grad_DP_table_sec = np.zeros((self.num_factors,2+1,self.num_attributes+1))
        self.grad_DP_table_thi = np.zeros((self.num_factors,num_order+1,self.num_attributes+1))
        self.grad_DP_table_sec[:,2,num_attributes]  = 1
        self.grad_DP_table_thi[:,num_order,num_attributes] = 1

        #only for adadelta algorithm
        self.delta_grad_w = np.zeros(self.num_attributes)
        self.delta_grad_v_p = np.zeros((self.num_attributes+1,self.num_factors+1))
        self.delta_squ_grad_w = np.zeros(self.num_attributes)
        self.delta_squ_grad_v_p = np.zeros((self.num_attributes+1,self.num_factors+1))

        #only for adam algorithm
        # use the prev_grad_w,prev_grad_v_p
        self.adam_grad_w = np.zeros(self.num_attributes)
        self.adam_grad_v_p = np.zeros((self.num_attributes+1,self.num_factors+1))
        self.adam_grad_v_q = np.zeros((self.num_attributes+1,self.num_factors+1))

        self.method = method
        self.path = path
        self.count = 0
        self.x_test = x_test
        self.y_test = y_test

        #for sparse group lasso regularization
        self.total_l1 = 0
        self.total_l21 = 0
        self.cur_l1 = np.zeros(self.num_attributes)
        self.cur_l21 = np.zeros(self.num_attributes)


    cdef _predict_instance(self,DOUBLE *x_data_ptr,INTEGER * x_ind_ptr,int xnnz):
        cdef DOUBLE result = 0.0
        cdef DOUBLE w0 = self.w0
        cdef np.ndarray[DOUBLE,ndim = 1,mode = 'c'] w = self.w
        cdef np.ndarray[DOUBLE,ndim = 2,mode = 'c'] v_p = self.v_p
        cdef np.ndarray[DOUBLE,ndim = 2,mode = 'c'] v_q = self.v_q
        cdef int i = 0
        cdef int t = 0
        cdef int k = 0
        cdef np.ndarray[DOUBLE,ndim = 2,mode = 'c'] DPT
        result += w0

        #step1 compute 1-st order response
        for i in range(xnnz):
            feature = x_ind_ptr[i]
            result += w[feature]*x_data_ptr[i]

        cdef int lastnnz = 0
        for i in range(self.num_factors):
            DPT = self.DP_table_sec[i,:,:]
            for t in range(1,2+1):
                lastnnz = 0
                for k in range(xnnz):
                    feature = x_ind_ptr[k] + 1
                    DPT[t,lastnnz+1:feature] = DPT[t,lastnnz]
                    DPT[t,feature] = DPT[t,feature-1] + v_p[feature,i]*x_data_ptr[k]
                    lastnnz = feature
            self.DP_table_sec[i,:,:] = DPT
            result += DPT[2,self.num_attributes]
        #third order
        if self.num_order == 3:
            for i in range(self.num_factors):
                DPT = self.DP_table_thi[i,:,:]
                for t in range(1,3+1):
                    lastnnz = 0
                    for k in range(xnnz):
                        feature = x_ind_ptr[k] + 1
                        DPT[t,lastnnz+1:feature] = DPT[t,lastnnz]
                        DPT[t,feature] = DPT[t,feature-1] + v_q[feature,i]*x_data_ptr[k]
                        lastnnz = feature
                self.DP_table_thi[i,:,:] = DPT
                result += DPT[3,self.num_attributes]

        return result

    def _predict(self,CSRDataset dataset):
        cdef unsigned int i=0
        cdef Py_ssize_t n_samples = dataset.n_samples
        cdef DOUBLE *x_data_ptr  = NULL
        cdef INTEGER *x_ind_ptr = NULL
        cdef int xnnz
        cdef DOUBLE y_placeholder
        cdef DOUBLE p = 0.0

        cdef np.ndarray[DOUBLE,ndim = 1,mode = 'c'] return_preds = np.zeros(n_samples)
        for i in range(n_samples):
            dataset.next(&x_data_ptr,&x_ind_ptr,&xnnz,&y_placeholder)
            p = self._predict_instance(x_data_ptr,x_ind_ptr,xnnz)
            return_preds[i] = p
        return return_preds

    cdef _grad_DP(self,DOUBLE *x_data_ptr,INTEGER *x_ind_ptr,int xnnz):

        cdef int d = self.num_attributes
        cdef int num_factors = self.num_factors
        cdef np.ndarray[DOUBLE,ndim =3,mode ='c'] DP_table_sec= self.DP_table_sec
        cdef np.ndarray[DOUBLE,ndim =2,mode = 'c'] grad_v_p = np.zeros((self.num_attributes+1,self.num_factors+1))
        cdef np.ndarray[DOUBLE,ndim =3,mode = 'c'] grad_DP_table_sec = np.zeros((self.num_factors,2+1,self.num_attributes+1))
        cdef np.ndarray[DOUBLE,ndim =2,mode = 'c'] v_p = self.v_p
        grad_DP_table_sec[:,2,self.num_attributes] = 1

       
        cdef np.ndarray[DOUBLE,ndim =3,mode ='c'] DP_table_thi= self.DP_table_thi 
        cdef np.ndarray[DOUBLE,ndim =2,mode = 'c'] grad_v_q = np.zeros((self.num_attributes+1,self.num_factors+1))
        cdef np.ndarray[DOUBLE,ndim =3,mode = 'c'] grad_DP_table_thi = np.zeros((self.num_factors,3+1,self.num_attributes+1))
        cdef np.ndarray[DOUBLE,ndim =2,mode = 'c'] v_q = self.v_q
        grad_DP_table_thi[:,3,self.num_attributes] = 1

        
        
        cdef int feature
        cdef int lastnnz = d
        cdef int i = 0
        cdef int t = 0
        cdef int k = 0
        #second degree
        for i in range(self.num_factors):
            for t in range(2,0,-1):
                lastnnz = d
                if(t ==2):
                    grad_DP_table_sec[i,t,t:d] = 1
                    continue
                for k in range(xnnz-1,-1,-1):
                    feature = x_ind_ptr[k] + 1
                    if(feature >= t):
                        if(feature == d):
                            continue
                        grad_DP_table_sec[i,t,feature:lastnnz] = 0
                        grad_DP_table_sec[i,t,feature:lastnnz] = grad_DP_table_sec[i,t,lastnnz]
                        grad_DP_table_sec[i,t,feature-1]=grad_DP_table_sec[i,t+1,feature]*v_p[feature,i]*x_data_ptr[k]
                        grad_DP_table_sec[i,t,feature-1] += grad_DP_table_sec[i,t,feature]
                        lastnnz = feature-1
                grad_DP_table_sec[i,t,t:lastnnz] = grad_DP_table_sec[i,t,lastnnz]
        
        for i in range(self.num_factors):
            for k in range(xnnz):
                feature = x_ind_ptr[k] +1
                for t in range(1,2+1):
                    grad_v_p[feature,i] += grad_DP_table_sec[i,t,feature] *DP_table_sec[i,t-1,feature-1]*x_data_ptr[k]
        
        #third order
        if self.num_order == 3:
            for i in range(self.num_factors):
                for t in range(3,0,-1):
                    lastnnz = d
                    if(t == 3):
                        grad_DP_table_thi[i,t,t:d] = 1
                        continue
                    for k in range(xnnz-1,-1,-1):
                        feature = x_ind_ptr[k] + 1
                        if(feature >= t):
                            if(feature == d):
                                continue
                            grad_DP_table_thi[i,t,feature:lastnnz] = 0
                            grad_DP_table_thi[i,t,feature:lastnnz] = grad_DP_table_thi[i,t,lastnnz]
                            grad_DP_table_thi[i,t,feature-1]=grad_DP_table_thi[i,t+1,feature]*v_q[feature,i]*x_data_ptr[k]
                            grad_DP_table_thi[i,t,feature-1] += grad_DP_table_thi[i,t,feature]
                            lastnnz = feature-1
                    grad_DP_table_thi[i,t,t:lastnnz] = grad_DP_table_thi[i,t,lastnnz]
        
            for i in range(self.num_factors):
                for k in range(xnnz):
                    feature = x_ind_ptr[k] +1
                    for t in range(1,3+1):
                        grad_v_q[feature,i] += grad_DP_table_thi[i,t,feature] *DP_table_thi[i,t-1,feature-1]*x_data_ptr[k]
        

        self.grad_v_p = grad_v_p
        self.grad_v_q = grad_v_q

   
    

    
    cdef _sgd_theta_FOBOS(self,DOUBLE *x_data_ptr,INTEGER * x_ind_ptr, int xnnz, DOUBLE y):
 
        cdef DOUBLE w0 = self.w0
        cdef np.ndarray[DOUBLE,ndim =1,mode='c'] w = self.w
        cdef np.ndarray[DOUBLE,ndim = 2,mode='c'] v_p = self.v_p
        cdef np.ndarray[DOUBLE,ndim = 2,mode='c'] v_q = self.v_q
        cdef np.ndarray[DOUBLE,ndim=1,mode='c'] grad_w = self.grad_w
        cdef int num_factors = self.num_factors
        cdef DOUBLE p = self._predict_instance(x_data_ptr,x_ind_ptr,xnnz)
        cdef DOUBLE mult = 2*(p-y)

        
        self.sum_loss += self._squared_loss(y,p)

        cdef unsigned int i=0
        cdef unsigned int j=0
        cdef int feature =0
        
        cdef DOUBLE eta = 0.01
        cdef DOUBLE beta1 = 0.9
        cdef DOUBLE beta2 = 0.999

        cdef DOUBLE hat_m
        cdef DOUBLE hat_v
        #bias
        cdef DOUBLE grad_0 = mult
        w0 -= self.learning_rate*grad_0

        #update 1-st order coefficient
        for i in range(xnnz):
            feature = x_ind_ptr[i]
            grad_w[feature] = mult*x_data_ptr[i] 
            self.prev_grad_w[feature] = beta2*self.prev_grad_w[feature] + (1-beta2)*grad_w[feature]**2
            self.adam_grad_w[feature] = beta1*self.adam_grad_w[feature] + (1-beta1)*grad_w[feature]
            hat_m  = self.adam_grad_w[feature]/(1-beta1)
            hat_v = self.prev_grad_w[feature]/(1-beta2)
            w[feature] -= (eta/(np.sqrt(hat_v)+1e-8))*hat_m
            

        

        #update second-order coefficients
        # _grad_DP return the incomplete gradients
        self._grad_DP(x_data_ptr,x_ind_ptr,xnnz)
        cdef np.ndarray[DOUBLE,ndim = 2,mode = 'c'] grad_v_p = self.grad_v_p
        cdef np.ndarray[DOUBLE,ndim = 2,mode = 'c'] grad_v_q = self.grad_v_q

        for i in range(1,self.num_factors+1):
            for j in range(xnnz):
                feature = x_ind_ptr[j]+1
                grad_v_p[feature,i] = mult*grad_v_p[feature,i]
                self.prev_grad_v_p[feature,i] = beta2*self.prev_grad_v_p[feature,i] + (1-beta2)* grad_v_p[feature,i]**2
                self.adam_grad_v_p[feature,i] = beta1*self.adam_grad_v_p[feature,i] + (1-beta1)* grad_v_p[feature,i]
                hat_m = self.adam_grad_v_p[feature,i]/(1-beta1)
                hat_v = self.prev_grad_v_p[feature,i]/(1-beta2)
                v_p[feature,i] -= (eta/(np.sqrt(hat_v)+1e-8))*hat_m
        
        #third -order
        if self.num_order == 3:
            for i in range(1,self.num_factors+1):
                for j in range(xnnz):
                    feature = x_ind_ptr[j]+1
                    grad_v_q[feature,i] = mult*grad_v_q[feature,i]
                    self.prev_grad_v_q[feature,i] = beta2*self.prev_grad_v_q[feature,i] + (1-beta2)* grad_v_q[feature,i]**2
                    self.adam_grad_v_q[feature,i] = beta1*self.adam_grad_v_q[feature,i] + (1-beta1)* grad_v_q[feature,i]
                    hat_m = self.adam_grad_v_q[feature,i]/(1-beta1)
                    hat_v = self.prev_grad_v_q[feature,i]/(1-beta2)
                    v_q[feature,i] -= (eta/(np.sqrt(hat_v)+1e-8))*hat_m   
        
        # L1 regularization
        self.total_l1 += self.learning_rate*self.reg_1
        
        for i in range(xnnz):
            feature = x_ind_ptr[i]
            lambda_1 = self.total_l1 - self.cur_l1[feature]
            self.cur_l1[feature] = self.total_l1
           
            comb = np.append(w[feature],[v_p[feature+1,1:],v_q[feature+1,1:]])
            abscomb = abs(comb)
            comb[abscomb <= lambda_1] = 0
            ind = abscomb>lambda_1
            comb[ind] = ((abscomb[ind]-lambda_1)/abscomb[ind])*comb[ind]
            w[feature] = comb[0]
            v_p[feature+1,1:] = comb[1:self.num_factors+1]
            v_q[feature+1,1:] = comb[self.num_factors+1:]
        
        #L2 on each column of U
        self.total_l21 += self.learning_rate*self.reg_2
        for i in range(xnnz):
            feature = x_ind_ptr[i]
            lambda_2 = self.total_l21 - self.cur_l21[feature]
            self.cur_l21[feature] = self.total_l21
            comb  = np.append(w[feature],[v_p[feature+1,1:],v_q[feature+1,1:]])
            norm_comb = np.linalg.norm(comb)
            if(norm_comb < lambda_2):
                mult = 0
            else:
                mult = (norm_comb-lambda_2)/norm_comb
            comb = mult*comb
            w[feature] = comb[0]
            v_p[feature+1,1:] = comb[1:self.num_factors+1]
            v_q[feature+1,1:] = comb[self.num_factors+1:]

        self.w0 = w0
        self.w = w
        self.v_p = v_p
        self.v_q = v_q
        self.t +=1
        self.count +=1    
    
    
    
    def fit(self,CSRDataset dataset):
        cdef Py_ssize_t n_samples  = dataset.n_samples
        cdef DOUBLE * x_data_ptr = NULL
        cdef INTEGER * x_ind_ptr = NULL
        cdef int xnnz
        cdef DOUBLE y = 0.0
        cdef int itercount = 0
        cdef unsigned int epoch
        cdef unsigned int i

        #add early stopping support 
        cdef int early_stopping = 0
        cdef double minist_loss = 1000
        
        fh_train = open(self.path+'train_'+str(self.reg_1)+'_'+str(self.reg_2)+'_order_'+str(self.num_order)+'.txt','w')
        fh_test = open(self.path +'test_'+str(self.reg_1)+'_'+str(self.reg_2)+'_order_'+str(self.num_order)+'.txt','w')
        for epoch in range(self.n_iter):
            self.count = 0 
            self.sum_loss = 0
            dataset.shuffle()
            
            for i in tqdm(range(n_samples)):
                dataset.next(&x_data_ptr,&x_ind_ptr,&xnnz,&y)
                #self._sgd_theta_step_adam(x_data_ptr,x_ind_ptr,xnnz,y)
                self._sgd_theta_FOBOS(x_data_ptr,x_ind_ptr,xnnz,y)
            training_error = np.sqrt(self.sum_loss/self.count)
            print('training_error:%f'%training_error)

            fh_train.write(str(training_error)+'\n')
            if itercount % 1==0:
                pred = self._predict(self.x_test)
                test_error = np.sqrt(np.sum((pred-self.y_test)**2)/pred.shape[0])
                print('Testing error %f'%test_error)
                fh_test.write(str(test_error)+'\n')

                if minist_loss > test_error:
                    early_stopping  = 0
                    minist_loss = test_error
                early_stopping +=1 
                if early_stopping == 20:
                    print('early_stopping......')
                    fh_train.close()
                    fh_test.close()
                    break
        
            #itercount +=1
        fh_train.close()
        fh_test.close()
    cdef _squared_loss(self,DOUBLE y, DOUBLE p):
        return (y-p)*(y-p)

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
        cdef np.ndarray[INTEGER, ndim=1,mode='c'] index = np.arange(0, self.n_samples,
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
    cdef void shuffle(self):
        np.random.shuffle(self.index)
