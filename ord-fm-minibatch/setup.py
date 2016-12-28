from distutils.core import setup
from Cython.Build import cythonize
import numpy
setup(
  name = 'mypyfast',
  ext_modules=cythonize("my_pyfm_fast.pyx"),
    include_dirs=[numpy.get_include()]
)
