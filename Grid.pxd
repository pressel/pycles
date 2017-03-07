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

        double [3] dx
        double [3] dxi


        double zp_half_0
        double zp_0

        double * jac;
        double * jac_half;
        double * ijac;
        double * ijac_half;

        double * jacl;
        double * jacl_half;
        double * ijacl;
        double * ijacl_half




cdef class Grid:
    cdef:
        DimStruct dims

        double [:] x
        double [:] x_half
        double [:] y
        double [:] y_half
        double [:] z
        double [:] z_half

        double [:] xl
        double [:] xl_half
        double [:] yl
        double [:] yl_half
        double [:] zl
        double [:] zl_half

        double [:] zp
        double [:] zp_half
        double [:] zpl
        double [:] zpl_half

        double [:] dzp
        double [:] dzp_half

        double [:] jac;
        double [:] jac_half;
        double [:] ijac;
        double [:] ijac_half;

        double [:] jacl;
        double [:] jacl_half;
        double [:] ijacl;
        double [:] ijacl_half;

        void compute_global_dims(self)
        void compute_local_dims(self,ParallelMPI.ParallelMPI Parallel)
        void compute_coordinates(self)

    cpdef extract_local(self, double [:] global_array, int dim)

    cpdef extract_local_ghosted(self, double [:] global_array, int dim)

    cpdef restart(self, Restart.Restart Re)



