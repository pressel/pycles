from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np
import mpi4py as mpi4py
import sys
import platform
import subprocess as sp
import os.path
import string
import shutil

if not os.environ.get("CC"):
    os.environ["CC"] = "mpicc"

# Now get include paths from relevant python modules
include_path = [mpi4py.get_include()]
include_path += [np.get_include()]
include_path += ['./Csrc']

def get_netcdf_include():
    return sp.check_output(['nc-config', '--includedir']).strip().decode()

def get_netcdf_prefix():
    return sp.check_output(['nc-config', '--prefix']).strip().decode()

conda_root = os.environ.get('CONDA_DEFAULT_ENV')
print(f'[setup.py] {conda_root=}')
if conda_root:
    # Compile flags for conda env (implemented on ch4 at IAC-ETHZ)
    assert conda_root, "no active conda env"  # SR/TMP
    conda_lib = conda_root + '/lib'
    conda_include = conda_root + '/include'
    library_dirs = [conda_lib]
    #+library_dirs = os.environ['LD_LIBRARY_PATH'].split(':')
    libraries = []
    libraries.append('mpi')
    libraries.append('gfortran')
    extensions = []
    extra_compile_args=[]
    extra_compile_args+=['-std=c99', '-O3', '-march=native', '-Wno-unused',
                         '-Wno-#warnings', '-Wno-maybe-uninitialized', '-Wno-cpp', '-Wno-array-bounds','-fPIC']
    extra_objects=['./RRTMG/rrtmg_build/rrtmg_combined.o']
    netcdf_include = conda_include
    netcdf_lib = conda_lib
    f_compiler = 'gfortran'
elif sys.platform == 'darwin':
    #Compile flags for MacOSX
    library_dirs = []
    libraries = []
    extensions = []
    extra_compile_args = []
    extra_compile_args += ['-O3', '-march=native', '-Wno-unused', '-Wno-#warnings','-fPIC']
    extra_objects=['./RRTMG/rrtmg_build/rrtmg_combined.o']
    netcdf_include = get_netcdf_include()
    netcdf_lib = os.path.join(get_netcdf_prefix(), 'lib')
    f_compiler = 'gfortran'
elif 'eu' in platform.node():
    #Compile flags for euler @ ETHZ (original PyCLES etup from 2016)
    # library_dirs = ['/cluster/apps/openmpi/1.6.5/x86_64/gcc_4.8.2/lib/']
    library_dirs = ['/cluster/apps/gcc-4.8.5/openmpi-4.1.4-pu2smponvdeu574nqolsw4rynnagngch/lib']
    libraries = []
    libraries.append('mpi')
    libraries.append('gfortran')
    extensions = []
    extra_compile_args=[]
    extra_compile_args+=['-std=c99', '-O3', '-march=native', '-Wno-unused',
                         '-Wno-#warnings', '-Wno-maybe-uninitialized', '-Wno-cpp', '-Wno-array-bounds','-fPIC']
    extra_objects=['./RRTMG/rrtmg_build/rrtmg_combined.o']
    netcdf_include = '/cluster/apps/gcc-4.8.5/netcdf-4.9.0-f2lwkfrx5o6b67cl2sv634ba5vmhg3qg/include'
    # netcdf_include = '/cluster/apps/netcdf/4.3.1/x86_64/gcc_4.8.2/openmpi_1.6.5/include'
    netcdf_lib = '/cluster/apps/gcc-4.8.5/netcdf-4.9.0-f2lwkfrx5o6b67cl2sv634ba5vmhg3qg/lib'
    # netcdf_lib = '/cluster/apps/netcdf/4.3.1/x86_64/gcc_4.8.2/openmpi_1.6.5/lib'
    f_compiler = 'gfortran'
elif (platform.machine()  == 'x86_64') and ('LD_LIBRARY_PATH' in os.environ):
    #Compile flags for Central @ Caltech
    library_dirs = os.environ['LD_LIBRARY_PATH'].split(':')
    libraries = []
    libraries.append('mpi')
    libraries.append('gfortran')
    extensions = []
    extra_compile_args=[]
    extra_compile_args+=['-std=c99', '-O3', '-march=native', '-Wno-unused',
                         '-Wno-#warnings', '-Wno-maybe-uninitialized', '-Wno-cpp', '-Wno-array-bounds','-fPIC']
    extra_objects=['./RRTMG/rrtmg_build/rrtmg_combined.o']
    netcdf_include = '/central/software/netcdf-c/4.6.1/include'
    netcdf_lib = '/central/software/netcdf-c/4.6.1/lib'
    # Comment the above two lines and uncomment below to use Fram@Caltech)
    #netcdf_include = '/share/apps/software/rhel6/software/netCDF/4.4.0-foss-2016a/include'
    #netcdf_lib = '/share/apps/software/rhel6/software/netCDF/4.4.0-foss-2016a/lib'
    f_compiler = 'gfortran'

else:
    if platform.system()=='Linux':
        #Best guess at compile flags for a Linux computer
        library_dirs = os.environ['PATH'].split(':')
        libraries = []
        libraries.append('mpi')
        libraries.append('gfortran')
        extensions = []
        extra_compile_args=[]
        extra_compile_args+=['-std=c99', '-O3', '-march=native', '-Wno-unused',
                             '-Wno-#warnings', '-Wno-maybe-uninitialized', '-Wno-cpp', '-Wno-array-bounds','-fPIC']
        extra_objects=['./RRTMG/rrtmg_build/rrtmg_combined.o']
        netcdf_include = '/share/apps/software/rhel6/software/netCDF/4.4.0-foss-2016a/include'
        netcdf_lib = '/share/apps/software/rhel6/software/netCDF/4.4.0-foss-2016a/lib'
        f_compiler = 'gfortran'

    else: 
        print('Unknown system platform: ' + sys.platform  + 'or unknown system name: ' + platform.node())
        sys.exit()

_ext = Extension('Grid', ['Grid.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('Initialization', ['Initialization.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('Microphysics', ['Microphysics.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('Microphysics_Arctic_1M', ['Microphysics_Arctic_1M.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('thermodynamic_functions', ['thermodynamic_functions.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('Thermodynamics', ['Thermodynamics.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('ThermodynamicsDry', ['ThermodynamicsDry.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('ThermodynamicsSA', ['ThermodynamicsSA.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('ReferenceState', ['ReferenceState.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('Simulation3d', ['Simulation3d.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('ParallelMPI', ['ParallelMPI.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('PrognosticVariables', ['PrognosticVariables.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('DiagnosticVariables', ['DiagnosticVariables.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('ScalarAdvection', ['ScalarAdvection.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('MomentumAdvection', ['MomentumAdvection.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('ScalarDiffusion', ['ScalarDiffusion.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('MomentumDiffusion', ['MomentumDiffusion.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('FluxDivergence', ['FluxDivergence.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('PressureSolver', ['PressureSolver.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('PressureFFTSerial', ['PressureFFTSerial.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('PressureFFTParallel', ['PressureFFTParallel.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('SparseSolvers', ['SparseSolvers.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('SGS', ['SGS.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('TimeStepping', ['TimeStepping.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('Kinematics', ['Kinematics.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('Lookup', ['Lookup.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('NetCDFIO', ['NetCDFIO.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('Surface', ['Surface.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)


_ext = Extension('SurfaceBudget', ['SurfaceBudget.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('Damping', ['Damping.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('Forcing', ['Forcing.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('entropies', ['entropies.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('Radiation', ['Radiation.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs, extra_objects=extra_objects)
extensions.append(_ext)

_ext = Extension('AuxiliaryStatistics', ['AuxiliaryStatistics.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('ConditionalStatistics', ['ConditionalStatistics.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)
_ext = Extension('Tracers', ['Tracers.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('Restart', ['Restart.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('VisualizationOutput', ['VisualizationOutput.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('PostProcessing', ['PostProcessing.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

#Build RRTMG

if shutil.which("csh"):
    CSH = "csh"
elif shutil.which("tcsh"):
    CSH = "tcsh"
else:
    raise Exception("found neither csh nor tcsh")

rrtmg_compiled = os.path.exists('./RRTMG/rrtmg_build/rrtmg_combined.o')
if not rrtmg_compiled:
    run_str = 'cd ./RRTMG; '
    run_str += ('FC='+ f_compiler + ' LIB_NETCDF=' + netcdf_lib + ' INC_NETCDF='+
               netcdf_include + ' ' + CSH + ' ./compile_RRTMG_combined.csh')
    print(run_str)
    sp.call([run_str], shell=True)
else:
    print("RRTMG Seems to be already compiled.")




setup(
    ext_modules=cythonize(extensions, verbose=1, include_path=include_path,
                          language_level=sys.version_info.major)
)
