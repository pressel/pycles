import numpy as np
cimport numpy as np

# cdef extern from "momentum_advection.h":
#     void compute_advective_tendencies_m(Grid.DimStruct *dims, double *rho0, double *rho0_half,
#                                     double *alpha0, double *alpha0_half, double *vel_advected,
#                                     double *vel_advecting, double *tendency, Py_ssize_t d_advected,
#                                     Py_ssize_t d_advecting, Py_ssize_t scheme)
cimport Grid
cimport ParallelMPI
cimport PrognosticVariables

cdef extern from "cc_statistics.h":
    void horizontal_mean(Grid.DimStruct *dims, double *values)

cdef class TestArray:
    def __init__(self,namelist):
        print('initialising TestArray_c')
        return

    # def initialize(self):
    def initialize(self, namelist):
        # self.Pa = ParallelMPI.ParallelMPI(namelist)
        # self.Gr = Grid.Grid(self.Pa)
        # self.Gr = Grid.Grid(namelist, self.Pa)
        return

    cpdef array_c(self):
        k = 2
        m = 10
        b = np.empty((k,m))
        # self.array_mean()
        return b

    cpdef array_mean(self, PrognosticVariables.PrognosticVariables PV, Grid.Grid Gr):
        print('calling TestArray_c.array_mean')
        # # horizontal_mean()
        # cdef dims = Gr.dims
        # print(dims)
        # cdef nxg = dims.ng[0]
        # cdef nyg = dims.ng[1]
        # cdef nzg = dims.ng[2]

        # # cdef double [:,:,:,:] b = self.array_c

        cdef Py_ssize_t shift_u = PV.velocity_directions[0] * Gr.dims.npg
        horizontal_mean(&Gr.dims, &PV.values[shift_u])

        cdef int i
        for i in range(Gr.dims.npg):
            PV.values[i] = 2

        return

# compute_advective_tendencies_m(&Gr.dims, &Rs.rho0[0], &Rs.rho0_half[0], &Rs.alpha0[0], &Rs.alpha0_half[0],
#                                             &PV.values[shift_advected], &PV.values[shift_advecting],
#                                            &PV.tendencies[shift_advected], i_advected, i_advecting, self.order)
