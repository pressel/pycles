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

        float domain_scalar_sum(self, float local_value)
        float domain_scalar_max(self, float local_value)
        float domain_scalar_min(self, float local_value)
        float [:] domain_vector_sum(self, float [:] local_vector, Py_ssize_t n)
        float [:] HorizontalMean(self,Grid.Grid Gr, float* values)
        float [:] HorizontalMeanofSquares(self,Grid.Grid Gr, float* values1, float* values2)
        float [:] HorizontalMeanofCubes(self, Grid.Grid Gr, float* values1, float* values2, float* values3)
        float [:] HorizontalMaximum(self, Grid.Grid Gr, float* values)
        float [:] HorizontalMinimum(self, Grid.Grid Gr, float* values)
        float HorizontalMeanSurface(self, Grid.Grid Gr, float* values)
        float [:] HorizontalMeanConditional(self,Grid.Grid Gr, float* values, float* mask)
        float [:] HorizontalMeanofSquaresConditional(self,Grid.Grid Gr, float* values1, float* values2, float* mask)

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
    cdef float [:,:] forward_double(self, Grid.DimStruct *dims, ParallelMPI Pa, float *data)
    cdef void build_buffer_double(self, Grid.DimStruct *dims, float *data, float *local_transpose)
    cdef void unpack_buffer_double(self,Grid.DimStruct *dims, float *recv_buffer, float [:,:] pencils)
    cdef void reverse_double(self, Grid.DimStruct *dims, ParallelMPI Pa, float [:,:] pencils, float *data)
    cdef void reverse_build_buffer_double(self, Grid.DimStruct *dims, float [:,:] pencils, float *send_buffer)
    cdef void reverse_unpack_buffer_double(self, Grid.DimStruct *dims, float *recv_buffer, float *data )
    cdef float complex [:,:] forward_complex(self, Grid.DimStruct *dims, ParallelMPI Pa, float complex *data)
    cdef void build_buffer_complex(self, Grid.DimStruct *dims, float complex *data, float complex *local_transpose)
    cdef void unpack_buffer_complex(self,Grid.DimStruct *dims, float complex *recv_buffer, float complex [:,:] pencils)
    cdef void reverse_complex(self, Grid.DimStruct *dims, ParallelMPI Pa, float complex [:,:] pencils, float complex *data)
    cdef void reverse_build_buffer_complex(self, Grid.DimStruct *dims, float complex [:,:] pencils, float complex *send_buffer)
    cdef void reverse_unpack_buffer_complex(self, Grid.DimStruct *dims, float complex *recv_buffer, float complex *data )