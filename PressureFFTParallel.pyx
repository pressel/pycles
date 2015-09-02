from scipy.fftpack import fft, ifft
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


cdef class PressureFFTParallel:
    def __init__(self):
        pass

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, ParallelMPI.ParallelMPI Pa):

        self.b = np.zeros(Gr.dims.nl[2],dtype=np.double,order='c')
        self.compute_modified_wave_numbers(Gr)
        self.compute_off_diagonals(Gr,RS)

        self.TDMA_Solver = SparseSolvers.TDMA()
        self.TDMA_Solver.initialize(Gr.dims.nl[2])

        self.X_Pencil = ParallelMPI.Pencil()
        self.Y_Pencil = ParallelMPI.Pencil()
        self.Z_Pencil = ParallelMPI.Pencil()

        self.X_Pencil.initialize(Gr,Pa,dim=0)
        self.Y_Pencil.initialize(Gr,Pa,dim=1)
        self.Z_Pencil.initialize(Gr,Pa,dim=2)

        return

    @cython.boundscheck(False)  #Turn off numpy array index bounds checking
    @cython.wraparound(False)   #Turn off numpy array wrap around indexing
    @cython.cdivision(True)
    cpdef compute_modified_wave_numbers(self,Grid.Grid Gr):
        self.kx2 = np.zeros(Gr.dims.nl[0],dtype=np.double,order='c')
        self.ky2 = np.zeros(Gr.dims.nl[1],dtype=np.double,order='c')
        cdef:
            double xi, yi
            long i,j,ii,jj
        for ii in xrange(Gr.dims.nl[0]):
            i = Gr.dims.indx_lo[0] + ii
            if i <= (Gr.dims.n[0] )/2:
                xi = np.double(i)
            else:
                xi = np.double(i - Gr.dims.n[0])
            self.kx2[ii] = (2.0 * cos((2.0 * pi/Gr.dims.n[0]) * xi)-2.0)/Gr.dims.dx[0]/Gr.dims.dx[0]

        for jj in xrange(Gr.dims.nl[1]):
            j = Gr.dims.indx_lo[1] + jj
            if j <= Gr.dims.n[1]/2:
                yi = np.double(j)
            else:
                yi = np.double(j-Gr.dims.n[1])
            self.ky2[jj] = (2.0 * cos((2.0 * pi/Gr.dims.n[1]) * yi)-2.0)/Gr.dims.dx[1]/Gr.dims.dx[1]

        #Remove the odd-ball
        if Gr.dims.indx_lo[0] == 0:
            self.kx2[0] = 0.0
        if Gr.dims.indx_lo[1] == 0:
            self.ky2[0] = 0.0

        return

    @cython.boundscheck(False)  #Turn off numpy array index bounds checking
    @cython.wraparound(False)   #Turn off numpy array wrap around indexing
    @cython.cdivision(True)
    cpdef compute_off_diagonals(self,Grid.Grid Gr, ReferenceState.ReferenceState RS):

        cdef:
            Py_ssize_t  k

        #self.a is the lower diagonal
        self.a = np.zeros(Gr.dims.n[2],dtype=np.double,order='c')
        #self.c is the upper diagonal
        self.c = np.zeros(Gr.dims.n[2],dtype=np.double,order='c')

        #Set boundary conditions at the surface
        self.a[0] =  0.0
        self.c[0] = Gr.dims.dxi[2] * Gr.dims.dxi[2] * RS.rho0[ Gr.dims.gw]

        #Fill Matrix Values
        for k in xrange(1,Gr.dims.n[2]-1):
            self.a[k] = Gr.dims.dxi[2] * Gr.dims.dxi[2] * RS.rho0[k + Gr.dims.gw-1]
            self.c[k] = Gr.dims.dxi[2] * Gr.dims.dxi[2] * RS.rho0[k + Gr.dims.gw]

        #Now set surface boundary conditions
        k = Gr.dims.n[2]-1
        self.a[k] = Gr.dims.dxi[2] * Gr.dims.dxi[2] * RS.rho0[k + Gr.dims.gw-1]
        self.c[k] = 0.0


    @cython.boundscheck(False)  #Turn off numpy array index bounds checking
    @cython.wraparound(False)   #Turn off numpy array wrap around indexing
    @cython.cdivision(True)
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

    @cython.boundscheck(False)  #Turn off numpy array index bounds checking
    @cython.wraparound(False)   #Turn off numpy array wrap around indexing
    @cython.cdivision(True)
    cpdef solve(self,Grid.Grid Gr, ReferenceState.ReferenceState RS,DiagnosticVariables.DiagnosticVariables DV
                , ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t i,j,k,ijk
            Py_ssize_t istride = Gr.dims.nl[1] * Gr.dims.nl[2]
            Py_ssize_t jstride = Gr.dims.nl[1]
            Py_ssize_t ishift, jshift
            double [:] dkr = np.empty((Gr.dims.nl[2]),dtype=np.double,order='c')
            double [:] dki = np.empty((Gr.dims.nl[2]),dtype=np.double,order='c')
            Py_ssize_t div_shift = DV.get_varshift(Gr,'divergence')
            Py_ssize_t pres_shift = DV.get_varshift(Gr,'dynamic_pressure')
            Py_ssize_t p, pencil_i, pencil_j
            Py_ssize_t count = 0
            Py_ssize_t pencil_shift = 0 #self.Z_Pencil.n_pencil_map[self.Z_Pencil.rank - 1]
            double [:,:] x_pencil,
            complex [:,:] x_pencil_fft, x_pencil_ifft, x_pencil_complex
            complex [:,:] y_pencil, z_pencil
            complex [:] div_fft= np.zeros(Gr.dims.npg,dtype=np.complex,order='c')
            complex [:] pres = np.zeros(Gr.dims.npg,dtype=np.complex,order='c')

        #Do fft in x direction
        x_pencil = self.X_Pencil.forward_double(&Gr.dims, Pa, &DV.values[div_shift])
        x_pencil_fft = fft(x_pencil,axis=1)
        self.X_Pencil.reverse_complex(&Gr.dims, Pa, x_pencil_fft, &div_fft[0] )

        #Do fft in y direction
        y_pencil = self.Y_Pencil.forward_complex(&Gr.dims, Pa, &div_fft[0])
        y_pencil_fft = fft(y_pencil,axis=1)
        self.Y_Pencil.reverse_complex(&Gr.dims, Pa, y_pencil_fft, &div_fft[0])

        #Transpose in z
        z_pencil = self.Z_Pencil.forward_complex(&Gr.dims, Pa, &div_fft[0])

        #At this point the data is in the correct pencils so we may do the TDMA solve
        for p in xrange(self.Z_Pencil.n_local_pencils):
            pencil_i = (pencil_shift + p) // Gr.dims.nl[1]
            pencil_j = (pencil_shift + p) % Gr.dims.nl[1]
            for k in xrange(Gr.dims.nl[2]):
                dkr[k] =  z_pencil[p,k].real
                dki[k] =  z_pencil[p,k].imag

            self.compute_diagonal(Gr,RS,pencil_i,pencil_j)
            self.TDMA_Solver.solve(&dkr[0],&self.a[0],&self.b[0],&self.c[0])
            self.TDMA_Solver.solve(&dki[0],&self.a[0],&self.b[0],&self.c[0])

            for k in xrange(Gr.dims.nl[2]):
               if pencil_i + Gr.dims.indx_lo[0] !=0 or pencil_j + Gr.dims.indx_lo[1] !=0:
                   z_pencil[p,k] = dkr[k] + dki[k] * 1j
               else:
                   z_pencil[p,k] = 0.0

        #Inverse transpose in z
        self.Z_Pencil.reverse_complex(&Gr.dims, Pa, z_pencil, &div_fft[0])

        #Do ifft in y direction
        y_pencil = self.Y_Pencil.forward_complex(&Gr.dims, Pa, &div_fft[0])
        y_pencil_fft = ifft(y_pencil,axis=1)
        self.Y_Pencil.reverse_complex(&Gr.dims, Pa, y_pencil_fft, &div_fft[0])

        #Do ifft in x direction
        x_pencil_complex = self.X_Pencil.forward_complex(&Gr.dims, Pa, &div_fft[0])
        x_pencil_ifft =ifft(x_pencil_complex,axis=1)
        self.X_Pencil.reverse_complex(&Gr.dims, Pa, x_pencil_ifft, &pres[0] )

        count = 0
        with nogil:
            for i in xrange(Gr.dims.npg):
                DV.values[pres_shift + i ] = pres[i].real

        return