import numpy as np
from sklearn import cross_validation
import random
from higher_fm_cython import FM, CSRDataset


class HOFM:

    def __init__(self,
                 num_order = 2,
                 num_factors = 10,
                 num_iter = 10,
                 path = './',
                 reg_1 = 0.01,
                 reg_2 = 0.01,
                 learning_rate = 1):

        self.num_order = num_order
        self.num_factors=num_factors
        self.num_iter = num_iter

        self.reg_0 = 0.01
        self.reg_1 = reg_1
        self.reg_2 = reg_2
        self.learning_rate = learning_rate
        self.path = path
  
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


        self.num_attributes = X.shape[1]
        X_train_dataset = _make_dataset(X,y)
        x_test_data = _make_dataset(x_test,y_test)

        self.fm_fast = FM(num_order = self.num_order,
                          n_iter = self.num_iter,
                          num_factors = self.num_factors,
                          num_attributes = self.num_attributes,
                          min_target = self.min_target,
                          max_target = self.max_target,
                          reg_1 = self.reg_1,
                          reg_2 = self.reg_2,
                          learning_rate = self.learning_rate,
                          method = 'adam',
                          path = self.path,
                          x_test = x_test_data,
                          y_test = y_test)
        self.fm_fast.fit(X_train_dataset)

    def predict(self,X,y):
        sparse_X = _make_dataset(X,y)
        #sparse_X = _make_dataset(X,np.ones(X.shape[0]))
        return self.fm_fast._predict(sparse_X)


def _make_dataset(X, y_i):
    """Create ``Dataset`` abstraction for sparse and dense inputs."""
    sample_weight = np.ones(X.shape[0], dtype=np.float64, order='C') # ignore sample weight for the moment
    dataset = CSRDataset(X.data, X.indptr, X.indices, y_i, sample_weight)
    return dataset
