cimport ParallelMPI as ParallelMPI
cimport PrognosticVariables as PrognosticVariables
cimport DiagnosticVariables as DiagnosticVariables
cimport Grid as Grid
cimport Restart

cdef class TimeStepping:
    cdef:
        public double dt
        public double t
        public double cfl_max
        public double cfl_limit
        public double dt_max
        public double dt_initial
        public double t_max
        public double acceleration_factor
        double [:,:] value_copies
        double [:,:] tendency_copies
        public Py_ssize_t rk_step
        public Py_ssize_t n_rk_steps
        public Py_ssize_t ts_type
        void initialize_second(self,PrognosticVariables.PrognosticVariables PV)
        void initialize_third(self,PrognosticVariables.PrognosticVariables PV)
        void initialize_fourth(self,PrognosticVariables.PrognosticVariables PV)


    cpdef initialize(self, namelist, PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI Pa)
    cpdef update(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI Pa)
    cpdef accelerate_tendencies(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI Pa)
    cpdef update_second(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)
    cpdef update_third(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)
    cpdef update_fourth(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)

    cpdef adjust_timestep(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
    cdef void compute_cfl_max(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa)
    cpdef restart(self, Restart.Restart Re)

    cdef inline double cfl_time_step(self)