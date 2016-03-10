from scipy.fftpack import fft, ifft
cimport Grid
cimport ParallelMPI
import cython
cimport numpy as np
import numpy as np
from libc.math cimport sqrt, ceil
include "parameters.pxi"



cdef class Filter:

    def __init__(self, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t ii, i,  jj, j
            double xi, yj

        # Set up the wavenumber vectors
        self.nwave = int( np.ceil(np.sqrt(2.0) * (Gr.dims.n[0] + 1.0) * 0.5 ) + 1.0)
        self.dk = 2.0 * pi/(Gr.dims.n[0]*Gr.dims.dx[0])
        self.wavenumbers = np.arange(self.nwave, dtype=np.double) * self.dk

        self.kx = np.zeros(Gr.dims.nl[0],dtype=np.double,order='c')
        self.ky = np.zeros(Gr.dims.nl[1],dtype=np.double,order='c')

        for ii in xrange(Gr.dims.nl[0]):
            i = Gr.dims.indx_lo[0] + ii
            if i <= (Gr.dims.n[0])/2:
                xi = np.double(i)
            else:
                xi = np.double(i - Gr.dims.n[0])
            self.kx[ii] = xi * self.dk
        for jj in xrange(Gr.dims.nl[1]):
            j = Gr.dims.indx_lo[1] + jj
            if j <= Gr.dims.n[1]/2:
                yj = np.double(j)
            else:
                yj = np.double(j-Gr.dims.n[1])
            self.ky[jj] = yj * self.dk

        #Instantiate classes used for Pencil communication/transposes
        self.X_Pencil = ParallelMPI.Pencil()
        self.Y_Pencil = ParallelMPI.Pencil()

        #Initialize classes used for Pencil communication/tranposes (here dim corresponds to the pencil direction)
        self.X_Pencil.initialize(Gr,Pa,dim=0)
        self.Y_Pencil.initialize(Gr,Pa,dim=1)

        return

    cdef double[:] spectral_2d(self, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa, double *data, double filter_factor):
        cdef:
            double [:] filtered_data = np.zeros(Gr.dims.npg,dtype=np.double,order='c')
            complex [:] filtered_data_complex = np.zeros(Gr.dims.npg,dtype=np.complex,order='c')
            complex [:] data_fft= np.zeros(Gr.dims.npg,dtype=np.complex,order='c')
            double [:,:] x_pencil
            complex [:,:] x_pencil_fft, x_pencil_ifft, x_pencil_complex
            complex [:,:] y_pencil, y_pencil_fft
            double k_cutoff = self.wavenumbers[self.nwave-1]/filter_factor
            Py_ssize_t i, j, k, ijk, kg, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t gw = Gr.dims.gw
            double [:] kx = self.kx
            double [:] ky = self.ky
            double kmag



        # Forward transform in X & Y
        #Do fft in x direction
        x_pencil = self.X_Pencil.forward_double(&Gr.dims, Pa, &data[0])
        x_pencil_fft = fft(x_pencil,axis=1)
        self.X_Pencil.reverse_complex(&Gr.dims, Pa, x_pencil_fft, &data_fft[0])

        #Do fft in y direction
        y_pencil = self.Y_Pencil.forward_complex(&Gr.dims, Pa, &data_fft[0])
        y_pencil_fft = fft(y_pencil,axis=1)
        self.Y_Pencil.reverse_complex(&Gr.dims, Pa, y_pencil_fft, &data_fft[0])

        # Do the cutoff
        with nogil:
            for i in xrange(Gr.dims.nl[0]):
                ishift = (i + gw) * istride
                for j in xrange(Gr.dims.nl[1]):
                    jshift = (j + gw) * jstride
                    kmag = sqrt(kx[i]*kx[i] + ky[j]*ky[j])
                    if kmag > k_cutoff:
                        for k in xrange(Gr.dims.nl[2]):
                            kg = k + gw
                            ijk = ishift + jshift + kg
                            data_fft[ijk] = 0 + 0j


        # Now the backwards transform
        #Do ifft in y direction
        y_pencil = self.Y_Pencil.forward_complex(&Gr.dims, Pa, &data_fft[0])
        y_pencil_fft = ifft(y_pencil,axis=1)
        self.Y_Pencil.reverse_complex(&Gr.dims, Pa, y_pencil_fft, &data_fft[0])

        #Do ifft in x direction
        x_pencil_complex = self.X_Pencil.forward_complex(&Gr.dims, Pa, &data_fft[0])
        x_pencil_ifft =ifft(x_pencil_complex,axis=1)
        self.X_Pencil.reverse_complex(&Gr.dims, Pa, x_pencil_ifft, &filtered_data_complex[0] )

        with nogil:
            for i in xrange(Gr.dims.npg):
                filtered_data[i] = filtered_data_complex[i].real

        return  filtered_data

