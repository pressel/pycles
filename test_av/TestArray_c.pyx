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
    void horizontal_mean_return(Grid.DimStruct *dims, double *values, double *mean)

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
        u_val = PV.get_variable_array('u', Gr)

        cdef Py_ssize_t shift_u = PV.velocity_directions[0] * Gr.dims.npg
        horizontal_mean(&Gr.dims, &PV.values[shift_u])

        return


    cpdef array_mean_return(self, PrognosticVariables.PrognosticVariables PV, Grid.Grid Gr):
        print('calling TestArray_c.array_mean_return')

        u_val = PV.get_variable_array('u', Gr)
        # cdef double [:] u_mean = np.empty(shape = u_val.shape[2])
        cdef double [:] u_mean = np.zeros(Gr.dims.ng[2])
        print('!!!', u_mean.shape, Gr.dims.ng[2])
        print('!!! before !!!')
        print(np.array(u_mean))

        cdef Py_ssize_t shift_u = PV.velocity_directions[0] * Gr.dims.npg
        horizontal_mean_return(&Gr.dims, &PV.values[shift_u], &u_mean[0])

        print('!!! after !!!')
        print(np.array(u_mean))

        return





    cpdef set_PV_values_const(self, PrognosticVariables.PrognosticVariables PV, Grid.Grid Gr):
        cdef int i
        for i in range(Gr.dims.npg):
            PV.values[i] = 2.0

        return


    cpdef set_PV_values(self, PrognosticVariables.PrognosticVariables PV, Grid.Grid Gr):
        cdef:
            # Py_ssize_t shift_flux = i_advected * Gr.dims.dims * Gr.dims.npg + i_advecting * Gr.dims.npg
            Py_ssize_t i, j, k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]

            Py_ssize_t imin = 0
            Py_ssize_t jmin = 0
            Py_ssize_t kmin = 0

            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t kmax = Gr.dims.nlg[2]

        for i in range(Gr.dims.npg):
            PV.values[i] = 0.0

        for i in xrange(imin, imax):
                # print(i)
                ishift = i * istride
                for j in xrange(jmin, jmax):
                    jshift = j * jstride
                    for k in xrange(kmin, kmax):
                        PV.values[0+ishift+jshift+k] = i-Gr.dims.gw
                        # print(PV.values[0+ishift+jshift+k])
        return

