import numpy as np
cimport numpy as np

cimport TestArray_c

cimport Grid
cimport ParallelMPI
cimport PrognosticVariables

# cdef extern from "momentum_advection.h":
#     void compute_advective_tendencies_m(Grid.DimStruct *dims, double *rho0, double *rho0_half,
#                                     double *alpha0, double *alpha0_half, double *vel_advected,
#                                     double *vel_advecting, double *tendency, Py_ssize_t d_advected,
#                                     Py_ssize_t d_advecting, Py_ssize_t scheme) nogil
cdef extern from "cc_statistics.h":
    void horizontal_mean(Grid.DimStruct *dims, double *values)

class TestRun:
    def __init__(self,namelist):
        print('initialising TestArray')
        return

    # def initialize(self):
    def initialize(self, namelist):
        # a = ParallelMPI.ParallelMPI(namelist)
        self.Pa = ParallelMPI.ParallelMPI(namelist)
        self.Gr = Grid.Grid(namelist, self.Pa)
        self.PV = PrognosticVariables.PrognosticVariables(self.Gr)

        return

    def array(self):        # does NOT work with cdef
        k = 2
        m = 10
        b = np.empty((k,m))

        aux = np.linspace(0,m-1,m)
        b[0,:] = aux
        b[1,:] = aux

        # print(b[0,:])
        return b


    def array_mean(self, namelist):
        k = 2
        m = 10
        cdef double [:,:] b = np.empty((k,m))
        cdef double [:] aux = np.linspace(0,m-1,m)
        b[0,:] = aux
        b[1,:] = aux
        # print(*b[0,:])

        # cdef dims = self.Gr.dims
        # cdef nxg = dims.ng[0]
        # cdef nyg = dims.ng[1]
        # cdef nzg = dims.ng[2]
        self.PV.add_variable('u', 'm/s', "sym", "velocity", self.Pa)
        self.PV.set_velocity_direction('u', 0, self.Pa)
        self.PV.initialize(self.Gr, self.Pa)
        u_val = self.PV.get_variable_array('u', self.Gr)
        print('u shape: ', u_val.shape)

        # d = self.PV.velocity_directions[0]

        mean = TestArray_c.TestArray(namelist)
        mean.array_mean(self.PV, self.Gr)

        # cdef Py_ssize_t shift_u = self.PV.velocity_directions[0] * self.Gr.dims.npg     # cannot take the address of a Python variable
        # horizontal_mean(&self.Gr.dims, &self.PV.values[shift_u])

        # print(np.shape(self.PV.values))
        # cdef dims = Grid.Grid.dims
        # dims = self.Gr.dims
        # horizontal_mean(&dims, &b)
        # compute_advective_tendencies_m(&Gr.dims, &Rs.rho0[0], &Rs.rho0_half[0], &Rs.alpha0[0], &Rs.alpha0_half[0],
        #                                     &PV.values[shift_advected], &PV.values[shift_advecting],
        #                                    &PV.tendencies[shift_advected], i_advected, i_advecting, self.order)
        return

    def hor_mean(self,namelist):
        print('hor mean')
        mean = TestArray_c.TestArray(namelist)
        b = mean.array_c()
        mean.array_mean(self.PV, self.Gr)

        print('finished TestRun')
        return