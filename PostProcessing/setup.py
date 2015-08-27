from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extensions = []
extra_compile_args = []
# ,'-Rpass=loop-vectorize']
extra_compile_args += ['-O3', '-march=native', '-Wno-unused', '-Wno-#warnings']

_ext = Extension("combine3d", ["combine3d.pyx"],
                 extra_compile_args=extra_compile_args)
extensions.append(_ext)


setup(
    ext_modules=cythonize(extensions, verbose=1)
)
