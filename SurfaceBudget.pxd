cimport Grid
cimport ReferenceState

cimport ParallelMPI
cimport TimeStepping
cimport Radiation
cimport Surface
from NetCDFIO cimport NetCDFIO_Stats



cdef class SurfaceBudgetNone:

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, Radiation.RadiationBase Ra, Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Surface.SurfaceBase Sur, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef init_from_restart(self, Restart)
    cpdef restart(self, Restart)

cdef class SurfaceBudget:
    cdef:
        bint constant_sst
        bint constant_ohu
        double fixed_sst_time
        double water_depth
        # double ocean_heat_flux
        # double ohu_adjustment_timescale


    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, Radiation.RadiationBase Ra, Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Surface.SurfaceBase Sur, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef init_from_restart(self, Restart)
    cpdef restart(self, Restart)