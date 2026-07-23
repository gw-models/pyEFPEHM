import sys, os, re
from setuptools import setup, find_packages, Extension
import numpy

#ensure Cython is installed before importing it
try:
    from Cython.Build import cythonize
except ImportError:
    print("Cython not found, installing it now...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "Cython"], check=True, capture_output=True)
    from Cython.Build import cythonize

#read the version from the package __init__ so there is a single source of truth
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "pyEFPEHM", "__init__.py")) as f:
    version = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", f.read()).group(1)

#read the long description from the README
with open(os.path.join(here, "README.md")) as f:
    long_description = f.read()

setup(
    name             = 'pyEFPEHM',
    description      = 'Python package for an efficient fully precessing eccentric inspiral model.',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    version          = version,
    author           = 'Gonzalo Morrás, Geraint Pratten, Patricia Schmidt',
    author_email     = 'gonzalo.morras@ligo.org',
    license          = 'Apache-2.0',
    packages         = find_packages(),
    package_dir      = {'pyEFPEHM': 'pyEFPEHM'},
    ext_modules      = cythonize([Extension("pyEFPEHM.utils.cython_utils",["pyEFPEHM/utils/cython_utils.pyx"],include_dirs=[numpy.get_include()]),]),
    url              = 'https://github.com/gw-models/pyEFPEHM',
    download_url     = 'https://github.com/gw-models/pyEFPEHM',
    python_requires  = '>=3.8',
    install_requires = ['numpy','scipy>=1.9.3','Cython'],
    entry_points     = {
        #summed [h_+, h_x] polarizations, for pycbc.waveform.get_fd_waveform/get_td_waveform
        'pycbc.waveform.fd': ['pyEFPEHM = pyEFPEHM.waveform.plugins:pycbc_fd_plugin'],
        'pycbc.waveform.td': ['pyEFPEHM = pyEFPEHM.waveform.plugins:pycbc_td_plugin'],
        #un-summed (l, mp, n, m) modes, for pycbc.waveform.get_fd_waveform_modes/get_td_waveform_modes
        #(requires a pycbc version with the fd_modes/td_modes plugin entry points, see plugin.py)
        'pycbc.waveform.fd_modes': ['pyEFPEHM = pyEFPEHM.waveform.plugins:pycbc_fd_modes_plugin'],
        'pycbc.waveform.td_modes': ['pyEFPEHM = pyEFPEHM.waveform.plugins:pycbc_td_modes_plugin'],
    },
    classifiers      = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)

