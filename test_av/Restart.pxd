cimport ParallelMPI

cdef class Restart:
    cdef:
        public dict restart_data
        str restart_path
        str input_path
        public bint is_restart_run
        str uuid
        public double last_restart_time
        public double frequency

    cpdef initialize(self)
    cpdef write(self, ParallelMPI.ParallelMPI Pa)
    cpdef read(self, ParallelMPI.ParallelMPI Pa)
    cpdef free_memory(self)