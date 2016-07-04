cimport ParallelMPI
cimport Grid
cimport Restart
cimport PrognosticVariables

# cdef extern from "grid.h":
#     struct DimStruct:
#
#         int dims
#
#         int [3] n
#         int [3] ng
#         int [3] nl
#         int [3] nlg
#         int [3] indx_lo_g
#         int [3] indx_lo
#
#         int npd
#         int npl
#         int npg
#         int gw
#
#         int [3] nbuffer
#         int [3] ghosted_stride
#
#         double [3] dx
#         double [3] dxi
#
cdef class TestArray:
    cdef:
        int k
        int m
        double [:] b

        # double array(self)

    cpdef array_mean(self, PrognosticVariables.PrognosticVariables PV, Grid.Grid Gr)
    cpdef array_mean_return(self, PrognosticVariables.PrognosticVariables PV, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa)
    cpdef array_mean_return_3d(self, PrognosticVariables.PrognosticVariables PV, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa)
    cpdef array_mean_const(self, PrognosticVariables.PrognosticVariables PV, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa)


    cpdef set_PV_values_const(self, PrognosticVariables.PrognosticVariables PV, Grid.Grid Gr)
    cpdef set_PV_values(self, PrognosticVariables.PrognosticVariables PV, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa)



