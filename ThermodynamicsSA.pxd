cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport ParallelMPI

from Thermodynamics cimport ClausiusClapeyron
from NetCDFIO cimport NetCDFIO_Fields, NetCDFIO_Stats

cdef class ThermodynamicsSA:
    cdef:
        bint do_qt_clipping
        double (*L_fp)(double T, double Lambda) nogil
        double (*Lambda_fp)(double T) nogil
        ClausiusClapeyron CC
        s_prognostic

    cpdef initialize(self,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef entropy(self, double p0, double T, double qt, double ql, double qi)
    cpdef alpha(self, double p0, double T, double qt, double qv)
    cpdef eos(self, double p0, double s, double qt)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
              PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV)
    cpdef get_pv_star(self, t)
    cpdef get_lh(self,t)
    cpdef write_fields(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                       NetCDFIO_Fields NF, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                   DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef liquid_stats(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                    DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)