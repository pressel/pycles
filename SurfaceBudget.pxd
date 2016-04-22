cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport ParallelMPI
cimport TimeStepping
from Thermodynamics cimport  LatentHeat, ClausiusClapeyron
from NetCDFIO cimport NetCDFIO_Stats



cdef class SurfaceBudget:
    cdef:
        double ocean_heat_flux
        double water_depth_initial
        double water_depth_final
        double water_depth_time
        double water_depth

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS)
    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
