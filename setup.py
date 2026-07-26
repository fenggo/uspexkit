"""Build script for the Cython fingerprint extension.

Compiles uspexkit/uspex_fast_core.pyx into a shared object during
``pip install .``.  Falls back gracefully if Cython is not available
(the generated .c file is used instead).
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

import os
import sys

try:
    import numpy as np
    numpy_include = np.get_include()
except ImportError:
    numpy_include = os.path.join(sys.prefix, "include")

try:
    from Cython.Build import cythonize
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

_pyx = "uspexkit/uspex_fast_core.pyx"
_c = "uspexkit/uspex_fast_core.c"

if HAS_CYTHON and os.path.exists(_pyx):
    source = _pyx
    ext_modules = cythonize(
        [Extension(
            "uspexkit.uspex_fast_core",
            sources=[source],
            include_dirs=[numpy_include],
            extra_compile_args=["-O3"],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
        },
    )
elif os.path.exists(_c):
    ext_modules = [Extension(
        "uspexkit.uspex_fast_core",
        sources=[_c],
        include_dirs=[numpy_include],
        extra_compile_args=["-O3"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )]
else:
    ext_modules = []

setup(ext_modules=ext_modules)
