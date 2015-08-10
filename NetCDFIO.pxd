cimport ParallelMPI
cimport TimeStepping
cimport PrognosticVariables
cimport Grid
cdef class NetCDFIO_Stats:
    cdef:
        str stats_file_name
        str stats_path
        str output_path
        str path_plus_file
        str uuid

        public double last_output_time
        public double frequency

    cpdef initialize(self,dict namelist, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa)
    cpdef setup_stats_file(self, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa)

cdef class NetCDFIO_Fields:
    cdef:
        str fields_file_name
        str fields_path
        str output_path
        str path_plus_file
        str uuid

        public double last_output_time
        public double frequency

    cpdef initialize(self, dict namelist, ParallelMPI.ParallelMPI Pa)

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)

    cpdef create_fields_file(self,Grid.Grid Gr, ParallelMPI.ParallelMPI Pa)

    cpdef dump_prognostic_variables(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)

    cpdef add_field(self, name)
    cpdef write_field(self,name,double [:] data)