cimport ParallelMPI as ParallelMPI
cimport PrognosticVariables as PrognosticVariables
cimport Grid as Grid

cdef class TimeStepping:
    cdef:
        public double dt
        public double t

        double [:,:] value_copies
        double [:,:] tendency_copies

        public int rk_step
        public int n_rk_steps
        public int ts_type

        void initialize_second(self,PrognosticVariables.PrognosticVariables PV)
        void initialize_third(self,PrognosticVariables.PrognosticVariables PV)

    cpdef initialize(self, namelist, PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI Pa)

    cpdef update(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)

    cpdef update_second(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)
    cpdef update_third(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)

