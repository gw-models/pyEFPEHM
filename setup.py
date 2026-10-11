from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

#the extension needs numpy's headers, whose path is only known at build time and so
#cannot be declared statically in pyproject.toml; all other metadata lives there
setup(
    ext_modules = cythonize([
        Extension("pyEFPEHM.utils.cython_utils",
                  ["pyEFPEHM/utils/cython_utils.pyx"],
                  include_dirs=[numpy.get_include()]),
    ]),
)
