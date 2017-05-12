# Implement the Higher-Order FM using SGD

## includeing the following files
1. **higher_fm.py**

   the python version of "higher order factorization machine" using sgd.

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

