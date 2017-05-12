# modified-FM

 The code of the modified-FM (Factorization Machine), we use different regularizer for the Factorization Machine,and also implement the higher order FM（>=3）.
 - [higher order FM](https://github.com/Matafight/modified-FM/tree/dev/higher-order-fm)
    Implement higher order factorization machine using sgd
 - [adaptive-FM](https://github.com/Matafight/modified-FM/tree/dev/ord-fm-adaptive-sgd)
    Implement second order factorization machine with adaptive regularizer
 - [sgl-fm-sgd](https://github.com/Matafight/modified-FM/tree/dev/sgl-fm-sgd)
    There are four kinds of FM in this sub directory,including sparse group lasso based FM , L21 based FM ,L1 based FM and the square of L2 norm based FM.
 - [sgl-higher-order-fm-sgd](https://github.com/Matafight/modified-FM/tree/dev/sgl-higher-order-fm-sgd)
    Implement the higher order factorization machine with sparse group lasso regularization(some bugs to be fixed)
    
## Reference
 1. Rendle S. Factorization machines[C]//Data Mining (ICDM), 2010 IEEE 10th International Conference on. IEEE, 2010: 995-1000.
 2. Blondel M, Fujino A, Ueda N, et al. Higher-Order Factorization Machines[C]//Advances in Neural Information Processing Systems. 2016: 3351-3359.
 3. Rendle S. Learning recommender systems with adaptive regularization[C]//Proceedings of the fifth ACM international conference on Web search and data mining. ACM, 2012: 133-142.
 4. Friedman J, Hastie T, Tibshirani R. A note on the group lasso and a sparse group lasso[J]. arXiv preprint arXiv:1001.0736, 2010.