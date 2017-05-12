
from distutils.core import setup
from Cython.Build import cythonize
import numpy
setup(
    name = 'higher-order-fm',
    ext_modules = cythonize('higher_fm_cython.pyx'),
    include_dirs=[numpy.get_include()]
)