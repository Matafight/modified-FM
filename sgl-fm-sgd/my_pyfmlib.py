import numpy as np
from sklearn import cross_validation
import random
from my_pyfm_fast import FM_fast, CSRDataset

TASKS = {"regression":0, "classification":1}

class FM:

    def __init__(self,
                 num_factors = 10,
                 num_iter = 1,
                 init_stdev = 0.1,
                 validation_size = 0.01,
                 initial_learning_rate = 0.01,
                 t0=0.001,
                 task = 'regression',
                 verbose = True,
                 L_1 = True,
                 L_21 = True,
                 seed = 28,
                 path_detail = "unknown",
                 method_name = 'sgl',
                 reg_1 = 0.01,
                 reg_2 = 0.01,
                 if_pd = False,
                 mini_batch = False):
        self.num_factors=num_factors
        self.num_iter = num_iter
        self.sum = np.zeros(self.num_factors)
        self.sum_sqr = np.zeros(self.num_factors)
        self.init_stdev=init_stdev
        self.validation_size = validation_size
        self.task = task
        self.verbose = verbose
        self.seed = seed
        self.L_1 = L_1
        self.L_21 = L_21
        self.if_pd = if_pd

        #learning rate parameter
        self.eta0 = initial_learning_rate
        self.t = 1.0
        self.learning_rate = initial_learning_rate
        self.t0 = t0
        #regularization paramter ,start with no regularization
        self.reg_0 = 0.01
        self.reg_1 = reg_1
        self.reg_2 = reg_2
        self.path_detail = path_detail
        self.method_name = method_name
        self.mini_batch = mini_batch


  
    def _bool_to_int(self,bool_arg):
        if bool_arg == True:
            return 1
        else:
            return 0


    def fit(self,X,y,x_test,y_test):
        if type(y)!= np.ndarray:
            y = np.array(y)
        self.max_target = max(y)
        self.min_target = min(y)
        verbose = self._bool_to_int(self.verbose)

        L_1 = self._bool_to_int(self.L_1)
        L_21 = self._bool_to_int(self.L_21)
        if_pd = self._bool_to_int(self.if_pd)
        mini_batch = self._bool_to_int(self.mini_batch)
        self.num_attribute = X.shape[1]
        X_train_dataset = _make_dataset(X,y)
        x_test_data = _make_dataset(x_test,y_test)
        #setup params
        self.w0 = 0.0
        self.w = np.zeros(self.num_attribute)
        np.random.seed(seed=self.seed)
        self.v = np.random.normal(scale = self.init_stdev,size=(self.num_factors,self.num_attribute))
        task = 0
        self.fm_fast = FM_fast(self.w,
                                   self.v,
                                   self.num_factors,
                                   self.num_attribute,
                                   self.num_iter,
                                   self.w0,
                                   self.t,
                                   self.t0,
                                   self.min_target,
                                   self.max_target,
                                   self.eta0,
                                   task,
                                   self.seed,
                                   verbose,
                                   L_1,
                                   L_21,
                                   self.path_detail,
                                   self.method_name,
                                   self.reg_1,
                                   self.reg_2,
                                   x_test_data,
                                   y_test,
                                   if_pd,
                                   mini_batch)
        self.fm_fast.fit(X_train_dataset)

    def predict(self,X,y):
        sparse_X = _make_dataset(X,y)
        #sparse_X = _make_dataset(X,np.ones(X.shape[0]))
        return self.fm_fast._predict(sparse_X)
    def return_sparsity(self):
        return self.fm_fast.return_sparsity()






def _make_dataset(X, y_i):
    """Create ``Dataset`` abstraction for sparse and dense inputs."""
    sample_weight = np.ones(X.shape[0], dtype=np.float64, order='C') # ignore sample weight for the moment
    dataset = CSRDataset(X.data, X.indptr, X.indices, y_i, sample_weight)
    return dataset
