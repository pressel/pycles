cimport mpi4py.libmpi as mpi
cimport Grid

cdef class ParallelMPI:
    cdef:
        mpi.MPI_Comm  comm_world
        mpi.MPI_Comm  cart_comm_world
        mpi.MPI_Comm  cart_comm_sub_x
        mpi.MPI_Comm  cart_comm_sub_y
        mpi.MPI_Comm  cart_comm_sub_z
        mpi.MPI_Comm  cart_comm_sub_xy

        public int rank
        int size

        int sub_x_size
        int sub_y_size
        int sub_z_size

        int sub_x_rank
        int sub_y_rank
        int sub_z_rank

        void barrier(self)

        void create_sub_communicators(self)

        double domain_scalar_sum(self, double local_value)
        double domain_scalar_max(self, double local_value)
        double domain_scalar_min(self, double local_value)
        double domain_integral(self, Grid.Grid Gr, double* values, double* rho)
        double [:] domain_vector_sum(self, double [:] local_vector, Py_ssize_t n)
        double [:] HorizontalMean(self,Grid.Grid Gr, double* values)
        double [:] HorizontalMeanofSquares(self,Grid.Grid Gr, double* values1, double* values2)
        double [:] HorizontalMeanofCubes(self, Grid.Grid Gr, double* values1, double* values2, double* values3)
        double [:] HorizontalMaximum(self, Grid.Grid Gr, double* values)
        double [:] HorizontalMinimum(self, Grid.Grid Gr, double* values)
        double HorizontalMeanSurface(self, Grid.Grid Gr, double* values)
        double [:] HorizontalMeanConditional(self,Grid.Grid Gr, double* values, double* mask)
        double [:] HorizontalMeanofSquaresConditional(self,Grid.Grid Gr, double* values1, double* values2, double* mask)

    cpdef root_print(self, txt_output)
    cpdef kill(self)

cdef class Pencil:

    cdef:
        long n_total_pencils
        long n_local_pencils
        long pencil_length
        long [:] n_pencil_map
        long  [:] nl_map
        long n_local_values
        int [:] send_counts
        int [:] recv_counts
        int [:] sdispls
        int [:] rdispls
        int dim
        int size
        int rank

    cpdef initialize(self, Grid.Grid Gr, ParallelMPI Pa, int dim)
    cdef double [:,:] forward_double(self, Grid.DimStruct *dims, ParallelMPI Pa, double *data)
    cdef void build_buffer_double(self, Grid.DimStruct *dims, double *data, double *local_transpose)
    cdef void unpack_buffer_double(self,Grid.DimStruct *dims, double *recv_buffer, double [:,:] pencils)
    cdef void reverse_double(self, Grid.DimStruct *dims, ParallelMPI Pa, double [:,:] pencils, double *data)
    cdef void reverse_build_buffer_double(self, Grid.DimStruct *dims, double [:,:] pencils, double *send_buffer)
    cdef void reverse_unpack_buffer_double(self, Grid.DimStruct *dims, double *recv_buffer, double *data )
    cdef complex [:,:] forward_complex(self, Grid.DimStruct *dims, ParallelMPI Pa, complex *data)
    cdef void build_buffer_complex(self, Grid.DimStruct *dims, complex *data, complex *local_transpose)
    cdef void unpack_buffer_complex(self,Grid.DimStruct *dims, complex *recv_buffer, complex [:,:] pencils)
    cdef void reverse_complex(self, Grid.DimStruct *dims, ParallelMPI Pa, complex [:,:] pencils, complex *data)
    cdef void reverse_build_buffer_complex(self, Grid.DimStruct *dims, complex [:,:] pencils, complex *send_buffer)
    cdef void reverse_unpack_buffer_complex(self, Grid.DimStruct *dims, complex *recv_buffer, complex *data )