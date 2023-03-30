from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='convolve2d_cython',
    ext_modules=cythonize("convolve2d_cython.pyx"),
    include_dirs=[numpy.get_include()]
)