cimport ParallelMPI
cimport TimeStepping
cimport PrognosticVariables
cimport DiagnosticVariables
cimport Grid
cdef class NetCDFIO_Stats:
    cdef:
        object root_grp
        object profiles_grp
        object ts_grp

        str stats_file_name
        str stats_path
        str output_path
        str path_plus_file
        str uuid

        public double last_output_time
        public double frequency
        public bint do_output

    cpdef initialize(self, dict namelist, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa)
    cpdef setup_stats_file(self, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa)
    cpdef add_profile(self, var_name, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa, units=*, nice_name=*, desc=*)
    cpdef add_reference_profile(self, var_name, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa, units=*, nice_name=*, desc=*)
    cpdef add_ts(self, var_name, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa, units=*, nice_name=*, desc=*)
    cpdef open_files(self, ParallelMPI.ParallelMPI Pa)
    cpdef close_files(self, ParallelMPI.ParallelMPI Pa)
    cpdef write_profile(self, var_name, double[:] data, ParallelMPI.ParallelMPI Pa)
    cpdef write_reference_profile(self, var_name, double[:] data, ParallelMPI.ParallelMPI Pa)
    cpdef write_ts(self, var_name, double data, ParallelMPI.ParallelMPI Pa)
    cpdef write_simulation_time(self, double t, ParallelMPI.ParallelMPI Pa)

cdef class NetCDFIO_Fields:
    cdef:
        str fields_file_name
        str fields_path
        str output_path
        str path_plus_file
        str uuid
        list diagnostic_fields

        public double last_output_time
        public double frequency
        public bint do_output

    cpdef initialize(self, dict namelist, ParallelMPI.ParallelMPI Pa)

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)

    cpdef create_fields_file(self, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa)

    cpdef dump_prognostic_variables(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)
    cpdef dump_diagnostic_variables(self, Grid.Grid Gr, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)

    cpdef add_field(self, name)
    cpdef write_field(self, name, double[:] data)

cdef class NetCDFIO_CondStats:
    cdef:
        str stats_file_name
        str stats_path
        str output_path
        str path_plus_file
        str uuid

        public double last_output_time
        public double frequency
        public bint do_output

    cpdef initialize(self, dict namelist, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa)
    cpdef create_condstats_group(self, str groupname, str dimname, double[:] dimval, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa)
    cpdef add_condstat(self, str varname, str groupname, str dimname, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa)
    cpdef write_condstat(self, varname, groupname, double[:,:] data, ParallelMPI.ParallelMPI Pa)
    cpdef write_condstat_time(self, double t, ParallelMPI.ParallelMPI Pa)
