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
        float (*L_fp)(float T, float Lambda) nogil
        float (*Lambda_fp)(float T) nogil
        ClausiusClapeyron CC

    cpdef initialize(self,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef entropy(self, float p0, float T, float qt, float ql, float qi)
    cpdef alpha(self, float p0, float T, float qt, float qv)
    cpdef eos(self, float p0, float s, float qt)
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