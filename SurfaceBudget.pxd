cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport ParallelMPI
cimport TimeStepping
cimport Radiation
from Thermodynamics cimport  LatentHeat, ClausiusClapeyron
from NetCDFIO cimport NetCDFIO_Stats



cdef class SurfaceBudget:
    cdef:
        double ocean_heat_flux
        double water_depth_initial
        double water_depth_final
        double water_depth_time
        double water_depth
        double sst

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Radiation.Radiation Ra, ParallelMPI.ParallelMPI Pa,double time, double shf, double lhf)
    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
