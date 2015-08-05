cimport ParallelMPI
cdef class NetCDFIO:
    cdef:
        str stats_file_name
        str stats_path
        str uuid

    cpdef initialize(self,dict namelist, ParallelMPI.ParallelMPI Pa)
    cpdef setup_stats_file(self, ParallelMPI.ParallelMPI Pa)
