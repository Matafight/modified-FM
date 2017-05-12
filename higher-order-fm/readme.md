# Implement the Higher-Order FM using SGD

## includeing the following files
1. **higher_fm.py**

   the python version of "higher order factorization machine" using sgd.
   Currently, we support adam, adadelta and RMSprop algorithm. I introduced the several algorithms here http://matafight.github.io/2017/01/25/%E9%9A%8F%E6%9C%BA%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E7%AE%97%E6%B3%95/

2. **higher_fm_cython.pyx**

   since the python version code is pertty slow, i also implement the cython version of the code, and cython is a compiler specifically designed for accelerating the execution of python code. 

3. **hofmlib.py**

   this is an interface for main.py to call the cython extension codes.

4. **main.py**

    the file to load data, preprocess data and call the training algorithm.

5. **setup.py**

    basic file for building cython code.

6. **mymultiprocess_crossvalidation.py**

    this is originly used for crossvalidation, but it hasn't updated for quite long time, so it may not compatible to the current functions above, but when you want to use it , just modify some lines of it will works.

