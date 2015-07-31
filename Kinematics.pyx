cimport Grid
cimport PrognosticVariables

import numpy as np
cimport numpy as np


cdef extern from "kinematics.h":
    void compute_velocity_gradient(Grid.DimStruct *dims, double *v, double *vgrad, long d)
    void compute_strain_rate(Grid.DimStruct *dims, double *vgrad, double *strain_rate)
    void compute_strain_rate_mag(Grid.DimStruct *dims, double *vgrad, double *strain_rate)

cdef class Kinematics:
    def __init__(self):
        pass

    cpdef initialize(self, Grid.Grid Gr):
        self.vgrad = np.zeros(Gr.dims.npg * Gr.dims.dims * Gr.dims.dims,dtype=np.double, order='c')
        self.strain_rate = np.zeros(Gr.dims.npg * Gr.dims.dims * Gr.dims.dims, dtype=np.double, order = 'c')
        self.strain_rate_mag = np.zeros(Gr.dims.npg,dtype=np.double, order='c')
        return

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV):

        cdef:
            int vi1, d, count=0
            int shift_v1
            int shift_v_grad, shift_strain

        import time
        t1 = time.time()
        for vi1 in xrange(Gr.dims.dims):
            shift_v1 = PV.velocity_directions[vi1] * Gr.dims.npg
            for d in xrange(Gr.dims.dims):
                shift_v_grad = Gr.dims.npg * count
                compute_velocity_gradient(&Gr.dims,&PV.values[shift_v1],&self.vgrad[shift_v_grad],d)
                count += 1



        compute_strain_rate(&Gr.dims, &self.vgrad[0],&self.strain_rate[0])
        compute_strain_rate_mag(&Gr.dims, &self.strain_rate[0],&self.strain_rate_mag[0])
        t2 = time.time()
        print t2  - t1

        return

    cdef int get_grad_shift(self, Grid.Grid Gr, int vel_i, int dx_i):
        return 3 * Gr.dims.npg * vel_i + Gr.dims.npg * dx_i