cimport Grid
cimport ReferenceState
cimport ParallelMPI
cimport DiagnosticVariables
cimport PrognosticVariables

cdef class VisualizationOutput:

    cdef:
        str vis_path
        str uuid
        public double last_vis_time
        public double frequency

    cpdef initialize(self)
    cpdef write(self, Grid.Grid Gr,  ReferenceState.ReferenceState RS,
                PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                ParallelMPI.ParallelMPI Pa)