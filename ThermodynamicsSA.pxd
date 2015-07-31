cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport ParallelMPI

from Thermodynamics cimport ClausiusClapeyron

cdef class ThermodynamicsSA:
    cdef:
        double (*L_fp)(double T, double Lambda) nogil
        double (*Lambda_fp)(double T) nogil
        ClausiusClapeyron CC

    cpdef initialize(self,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,  DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Par)

    cpdef entropy(self, double p0, double T, double qt, double ql, double qi)

    cpdef alpha(self, double p0, double T, double qt, double qv)

    cpdef eos(self, double p0, double s, double qt)

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
              PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV)

    cpdef get_pv_star(self, t)