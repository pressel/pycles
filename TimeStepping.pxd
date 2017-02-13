cimport ParallelMPI as ParallelMPI
cimport PrognosticVariables as PrognosticVariables
cimport DiagnosticVariables as DiagnosticVariables
cimport Grid as Grid
cimport Restart

cdef class TimeStepping:
    cdef:
        public float dt
        public float t
        public float cfl_max
        public float cfl_limit
        public float dt_max
        public float dt_initial
        public float t_max
        float [:,:] value_copies
        float [:,:] tendency_copies
        public Py_ssize_t rk_step
        public Py_ssize_t n_rk_steps
        public Py_ssize_t ts_type
        void initialize_second(self,PrognosticVariables.PrognosticVariables PV)
        void initialize_third(self,PrognosticVariables.PrognosticVariables PV)
        void initialize_fourth(self,PrognosticVariables.PrognosticVariables PV)


    cpdef initialize(self, namelist, PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI Pa)
    cpdef update(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI Pa)
    cpdef update_second(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)
    cpdef update_third(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)
    cpdef update_fourth(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)

    cpdef adjust_timestep(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
    cdef void compute_cfl_max(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa)
    cpdef restart(self, Restart.Restart Re)

    cdef inline float cfl_time_step(self)