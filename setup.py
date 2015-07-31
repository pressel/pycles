from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np
import mpi4py as mpi4py
import petsc4py

#Now get include paths from relevant python modules
include_path = [mpi4py.get_include()]
include_path += [np.get_include()]
include_path += [petsc4py.get_include()]
include_path += ['./Csrc']
include_path += ['/opt/local/lib/petsc/include']


#library_dirs =['/opt/local/lib/petsc/lib/']
library_dirs = []
libraries = []
#libraries.append('petsc')


extensions = []
extra_compile_args=[]
extra_compile_args+=['-O3','-march=native','-Wno-unused','-Wno-#warnings']#,'-Rpass=loop-vectorize']

_ext = Extension("Grid",["Grid.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)

_ext = Extension("Initialization",["Initialization.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)

_ext = Extension("Microphysics",["Microphysics.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)

_ext = Extension("thermodynamic_functions",["thermodynamic_functions.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)

_ext = Extension("Thermodynamics",["Thermodynamics.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)

_ext = Extension("ThermodynamicsDry",["ThermodynamicsDry.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)

_ext = Extension("ThermodynamicsSA",["ThermodynamicsSA.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)

_ext = Extension("ReferenceState",["ReferenceState.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)

_ext = Extension("Simulation3d",["Simulation3d.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)

_ext = Extension("ParallelMPI",["ParallelMPI.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)

_ext = Extension("PrognosticVariables",["PrognosticVariables.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)

_ext = Extension("DiagnosticVariables",["DiagnosticVariables.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)

_ext = Extension("ScalarAdvection",["ScalarAdvection.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)

_ext = Extension("MomentumAdvection",["MomentumAdvection.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)

_ext = Extension("ScalarDiffusion",["ScalarDiffusion.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)

_ext = Extension("MomentumDiffusion",["MomentumDiffusion.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)

_ext = Extension("FluxDivergence",["FluxDivergence.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)

_ext = Extension("PressureSolver",["PressureSolver.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)

_ext = Extension("PressureFFTSerial",["PressureFFTSerial.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)

_ext = Extension("PressureFFTParallel",["PressureFFTParallel.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)

_ext = Extension("SparseSolvers",["SparseSolvers.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)

_ext = Extension("SGS",["SGS.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)

_ext = Extension("TimeStepping",["TimeStepping.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)


_ext = Extension("Kinematics",["Kinematics.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)

_ext = Extension("Lookup",["Lookup.pyx"],include_dirs = include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries , library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs  )
extensions.append(_ext)



setup(
    ext_modules = cythonize(extensions,verbose=1,include_path=include_path)
)
