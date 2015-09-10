#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

from scipy.fftpack import fft2, ifft2
cimport DiagnosticVariables

cimport ParallelMPI
cimport Grid
cimport ReferenceState
cimport SparseSolvers

import numpy as np
cimport numpy as np
from libc.math cimport cos

import cython
include 'parameters.pxi'

cdef class PressureFFTSerial:

    def __init__(self):
        pass

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, ParallelMPI.ParallelMPI Pa):
        self.compute_modified_wave_numbers(Gr)
        self.compute_off_diagonals(Gr,RS)
        self.b = np.zeros(Gr.dims.nl[2],dtype=np.double,order='c')

        self.TDMA_Solver = SparseSolvers.TDMA()
        self.TDMA_Solver.initialize(Gr.dims.nl[2])

        print('Initializing variables')
        return

    cpdef compute_modified_wave_numbers(self,Grid.Grid Gr):
        self.kx2 = np.zeros(Gr.dims.nl[0],dtype=np.double,order='c')
        self.ky2 = np.zeros(Gr.dims.nl[1],dtype=np.double,order='c')
        cdef:
            double xi, yi
            Py_ssize_t i,j
        for i in xrange(Gr.dims.nl[0]):
            if i <= Gr.dims.nl[0]/2:
                xi = np.double(i)
            else:
                xi = np.double(i - Gr.dims.nl[0])
            self.kx2[i] = (2.0 * cos((2.0 * pi/Gr.dims.nl[0]) * xi)-2.0)/Gr.dims.dx[0]/Gr.dims.dx[0]

        for j in xrange(Gr.dims.nl[1]):
            if j <= Gr.dims.nl[1]/2:
                yi = np.double(j)
            else:
                yi = np.double(j-Gr.dims.nl[1])
            self.ky2[j] = (2.0 * cos((2.0 * pi/Gr.dims.nl[1]) * yi)-2.0)/Gr.dims.dx[1]/Gr.dims.dx[1]

        #Remove the odd-ball
        self.kx2[0] = 0.0
        self.ky2[0] = 0.0

        return

    cpdef compute_off_diagonals(self,Grid.Grid Gr, ReferenceState.ReferenceState RS):

        cdef:
            Py_ssize_t  k

        #self.a is the lower diagonal
        self.a = np.zeros(Gr.dims.nl[2],dtype=np.double,order='c')
        #self.c is the upper diagonal
        self.c = np.zeros(Gr.dims.nl[2],dtype=np.double,order='c')

        #Set boundary conditions at the surface
        self.a[0] =  0.0
        self.c[0] = Gr.dims.dxi[2] * Gr.dims.dxi[2] * RS.rho0[ Gr.dims.gw]

        #Fill Matrix Values
        for k in xrange(1,Gr.dims.nl[2]-1):
            self.a[k] = Gr.dims.dxi[2] * Gr.dims.dxi[2] * RS.rho0[k + Gr.dims.gw-1]
            self.c[k] = Gr.dims.dxi[2] * Gr.dims.dxi[2] * RS.rho0[k + Gr.dims.gw]

        #Now set surface boundary conditions
        k = Gr.dims.nl[2]-1
        self.a[k] = Gr.dims.dxi[2] * Gr.dims.dxi[2] * RS.rho0[k + Gr.dims.gw-1]
        self.c[k] = 0.0

        return

    cdef inline void compute_diagonal(self,Grid.Grid Gr,ReferenceState.ReferenceState RS,Py_ssize_t i, Py_ssize_t j) nogil:

        cdef:
            Py_ssize_t k
            double kx2 = self.kx2[i]
            double ky2 = self.ky2[j]

        #Set the matrix rows for the interior point
        self.b[0] = (RS.rho0_half[ Gr.dims.gw] * (kx2 + ky2)
                         - (RS.rho0[ Gr.dims.gw] )*Gr.dims.dxi[2]*Gr.dims.dxi[2])

        for k in xrange(1,Gr.dims.nl[2]-1):
            self.b[k] = (RS.rho0_half[k + Gr.dims.gw] * (kx2 + ky2)
                         - (RS.rho0[k + Gr.dims.gw] + RS.rho0[k + Gr.dims.gw -1])*Gr.dims.dxi[2]*Gr.dims.dxi[2])
        k = Gr.dims.nl[2]-1
        self.b[k] = (RS.rho0_half[k + Gr.dims.gw] * (kx2 + ky2)
                         - (RS.rho0[k + Gr.dims.gw -1])*Gr.dims.dxi[2]*Gr.dims.dxi[2])


        return


    cpdef solve(self,Grid.Grid Gr, ReferenceState.ReferenceState RS,DiagnosticVariables.DiagnosticVariables DV,
                 ParallelMPI.ParallelMPI PM):

        cdef:
            complex [:,:,:] div_fft
            double [:,:,:] div = np.empty((Gr.dims.nl[0],Gr.dims.nl[1],Gr.dims.nl[2]),dtype=np.double,order='c')

            Py_ssize_t i, j,k, count

        #This is a relatively sloppy way to change the dimension of the array but for now we will use it
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw

            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw

            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]

            Py_ssize_t ishift, jshift, ijk

            Py_ssize_t div_shift = DV.get_varshift(Gr,'divergence')


        with nogil:
            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        div[i-Gr.dims.gw,j-Gr.dims.gw,k-Gr.dims.gw] = DV.values[div_shift + ijk ]


        #Compute the 2D horizontal fft of the dataset
        div_fft  =  fft2(div,axes=(0,1))

        #This is one of the few places in the code where we will use a 3d array this is to avoid a memory copy
        cdef double [:] dkr = np.empty((Gr.dims.nl[2]),dtype=np.double,order='c')
        cdef double [:] dki = np.empty((Gr.dims.nl[2]),dtype=np.double,order='c')


        #with nogil:
        for i in range(Gr.dims.nl[0]):
            for j in range(Gr.dims.nl[1]):
                for k in range(Gr.dims.nl[2]):
                    dkr[k] =  div_fft[i,j,k].real
                    dki[k] =  div_fft[i,j,k].imag

                self.compute_diagonal(Gr,RS,i,j)
                self.TDMA_Solver.solve(&dkr[0],&self.a[0],&self.b[0],&self.c[0])
                self.TDMA_Solver.solve(&dki[0],&self.a[0],&self.b[0],&self.c[0])

                for k in range(Gr.dims.nl[2]):
                    if i != 0 or j != 0:
                        div_fft[i,j,k] = dkr[k] + dki[k] * 1j
                    else:
                        div_fft[i,j,k] = 0.0 + 0j

        cdef:
            double [:,:,:] p = ifft2(div_fft,axes=(0,1)).real
            Py_ssize_t pres_shift = DV.get_varshift(Gr,'dynamic_pressure')


        with nogil:
            for i in xrange(Gr.dims.nl[0]):
                ishift = (i + Gr.dims.gw) * istride
                for j in xrange(Gr.dims.nl[1]):
                    jshift = (j + Gr.dims.gw) * jstride
                    for k in xrange(Gr.dims.nl[2]):
                        DV.values[pres_shift + ishift + jshift + k + Gr.dims.gw] = p[i,j,k]

        return

