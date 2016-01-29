from scipy.fftpack import fft, ifft
cimport Grid
cimport ParallelMPI
import cython
cimport numpy as np
import numpy as np
from libc.math cimport sqrt, ceil
include "parameters.pxi"



cdef class Filter:
    cdef:
        Py_ssize_t nwave
        double dk
        double [:] wavenumbers
        double [:] kx
        double [:] ky
        ParallelMPI.Pencil X_Pencil, Y_Pencil

    cdef double[:] spectral_2d(self, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa, double *data, double filter_factor)

