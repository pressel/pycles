import numpy as np
cimport numpy as np

cimport TestArray

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
    void horizontal_mean_return(Grid.DimStruct *dims, double *values, double *mean)

class TestRun:
    def __init__(self,namelist):
        print('initialising TestArray')
        return

    def initialize(self, namelist):
        self.Pa = ParallelMPI.ParallelMPI(namelist)
        self.Gr = Grid.Grid(namelist, self.Pa)
        self.PV = PrognosticVariables.PrognosticVariables(self.Gr)

        return


    # def array(self):        # does NOT work with cdef
    #     k = 2
    #     m = 10
    #     b = np.empty((k,m))
    #
    #     aux = np.linspace(0,m-1,m)
    #     b[0,:] = aux
    #     b[1,:] = aux
    #
    #     # print(b[0,:])
    #     return b


    def array_mean(self, namelist):
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
        # mean = TestArray.TestArray(namelist)
        # mean.array_mean(self.PV, self.Gr)

        # cdef Py_ssize_t shift_u = self.PV.velocity_directions[0] * self.Gr.dims.npg     # cannot take the address of a Python variable
        # horizontal_mean(&self.Gr.dims, &self.PV.values[shift_u])

        return


    def hor_mean(self,namelist):
        print('TestRun.hor_mean')

        mean = TestArray.TestArray(namelist)

        self.Pa.root_print('(1) const PV values')
        mean.set_PV_values_const(self.PV, self.Gr)
        mean.array_mean_return(self.PV, self.Gr, self.Pa)

        self.Pa.root_print('')
        self.Pa.root_print('(2) varying PV values')
        self.Pa.root_print('(2a) array_mean_return')
        mean.set_PV_values(self.PV, self.Gr, self.Pa)
        u_val = self.PV.get_variable_array('u', self.Gr)
        mean.array_mean_return(self.PV, self.Gr, self.Pa)

        # self.Pa.root_print('(2b) array_mean_const')
        # # mean.set_PV_values(self.PV, self.Gr, self.Pa)
        # # u_val = self.PV.get_variable_array('u', self.Gr)
        # mean.array_mean_const(self.PV, self.Gr, self.Pa)

        print('finished TestRun')



        return