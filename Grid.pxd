cimport ParallelMPI
cimport Restart

cdef extern from "grid.h":
    struct DimStruct:

        int dims

        int [3] n
        int [3] ng
        int [3] nl
        int [3] nlg
        int [3] indx_lo_g
        int [3] indx_lo

        int npd
        int npl
        int npg
        int gw

        int [3] nbuffer
        int [3] ghosted_stride

        float [3] dx
        float [3] dxi

cdef class Grid:
    cdef:
        DimStruct dims

        float [:] x
        float [:] x_half
        float [:] y
        float [:] y_half
        float [:] z
        float [:] z_half

        float [:] xl
        float [:] xl_half
        float [:] yl
        float [:] yl_half
        float [:] zl
        float [:] zl_half

        void compute_global_dims(self)
        void compute_local_dims(self,ParallelMPI.ParallelMPI Parallel)
        void compute_coordinates(self)

    cpdef extract_local(self, float [:] global_array, int dim)

    cpdef extract_local_ghosted(self, float [:] global_array, int dim)

    cpdef restart(self, Restart.Restart Re)



