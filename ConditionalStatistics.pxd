#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

from scipy.fftpack import fft, ifft
cimport Grid
cimport ReferenceState
cimport DiagnosticVariables
cimport PrognosticVariables
cimport ParallelMPI

from NetCDFIO cimport NetCDFIO_CondStats

include "parameters.pxi"

cdef class ConditionalStatistics:
    cdef list CondStatsClasses
    cpdef initialize(self, namelist, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                               DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_CondStats NC, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_CondStats NC, ParallelMPI.ParallelMPI Pa)


cdef class NullCondStats:

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,  NetCDFIO_CondStats NC, ParallelMPI.ParallelMPI Pa)


cdef class SpectraStatistics:
    cdef:
        Py_ssize_t nwave
        double dk
        double [:] wavenumbers
        double [:] kx
        double [:] ky
        cdef ParallelMPI.Pencil X_Pencil, Y_Pencil, Z_Pencil


    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,  NetCDFIO_CondStats NC, ParallelMPI.ParallelMPI Pa)

    cpdef forward_transform(self, Grid.Grid Gr,ParallelMPI.ParallelMPI Pa, double [:] data, complex [:] data_fft)
    cpdef fluctuation_forward_transform(self, Grid.Grid Gr,ParallelMPI.ParallelMPI Pa, double [:] data, complex [:] data_fft)
    cpdef compute_spectrum(self, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa, complex [:] data_fft )
    cpdef compute_cospectrum(self, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa, complex [:] data_fft_1, complex [:] data_fft_2 )

