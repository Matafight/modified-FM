#_*_ coding:utf-8 _*_
import numpy as np
from libc.math cimport exp,log,pow,sqrt
cimport numpy as np
cimport cython
np.import_array()

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INTEGER

cdef class FM(object):
    cdef double w0
    cdef np.ndarray w
    cdef np.ndarray v_p
    cdef np.ndarray v_q
    cdef int num_factors
    cdef int num_attributes
    cdef int num_order = num_order
    cdef str dataname
    cdef DOUBLE reg_1
    cdef DOUBLE reg_2
    cdef double learning_rate
    cdef DOUBLE t
    cdef double t0
    cdef DOUBLE sum_loss
    cdef np.ndarray grad_w
    cdef np.ndarray grad_v_p
    cdef np.ndarray grad_v_q
    cdef np.ndarray DP_table_sec
    cdef np.ndarray DP_table_thi
    
    
