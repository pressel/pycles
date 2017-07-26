#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True
cimport mpi4py.libmpi as mpi
cimport Grid
cimport ReferenceState
cimport ParallelMPI
cimport TimeStepping
cimport Radiation
cimport Surface
from NetCDFIO cimport NetCDFIO_Stats
import cython

cimport numpy as np
import numpy as np
include "parameters.pxi"

def SurfaceBudgetFactory(namelist):
    if namelist['meta']['casename'] == 'ZGILS':
        return SurfaceBudget(namelist)
    else:
        return SurfaceBudgetNone()

cdef class SurfaceBudgetNone:
    def __init__(self):
        return

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef update(self,Grid.Grid Gr, Radiation.RadiationBase Ra, Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef stats_io(self, Surface.SurfaceBase Sur, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return


cdef class SurfaceBudget:
    def __init__(self, namelist):
        try:
            self.constant_sst = namelist['surface_budget']['constant_sst']
        except:
            self.constant_sst = False
        try:
            self.ocean_heat_flux = namelist['surface_budget']['ocean_heat_flux']
        except:
            self.ocean_heat_flux = 0.0
        try:
            self.water_depth_initial = namelist['surface_budget']['water_depth_initial']
        except:
            self.water_depth_initial = 1.0
        try:
            self.water_depth_final = namelist['surface_budget']['water_depth_final']
        except:
            self.water_depth_final = 1.0
        try:
            self.water_depth_time = namelist['surface_budget']['water_depth_time']
        except:
            self.water_depth_time = 0.0
        # Allow spin up time with fixed sst
        try:
            self.fixed_sst_time = namelist['surface_budget']['fixed_sst_time']
        except:
            self.fixed_sst_time = 0.0
        self.water_depth = self.water_depth_initial
        return

    cpdef initialize(self, Grid.Grid Gr,  NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        NS.add_ts('surface_temperature', Gr, Pa)
        return

    cpdef update(self, Grid.Grid Gr, Radiation.RadiationBase Ra, Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):

        cdef:
            int root = 0
            int count = 1
            double rho_liquid = 1000.0
            double mean_shf = Pa.HorizontalMeanSurface(Gr, &Sur.shf[0])
            double mean_lhf = Pa.HorizontalMeanSurface(Gr, &Sur.lhf[0])
            double net_flux, tendency

        if self.constant_sst:
            return
        if TS.rk_step != 0:
            return
        if TS.t < self.fixed_sst_time:
            return

        if Pa.sub_z_rank == 0:

            if TS.t > self.water_depth_time:
                self.water_depth = self.water_depth_final
            else:
                self.water_depth = self.water_depth_initial

            net_flux =  -self.ocean_heat_flux - Ra.srf_lw_up - Ra.srf_sw_up - mean_shf - mean_lhf + Ra.srf_lw_down + Ra.srf_sw_down
            tendency = net_flux/cl/rho_liquid/self.water_depth
            Sur.T_surface += tendency *TS.dt

        mpi.MPI_Bcast(&Sur.T_surface,count,mpi.MPI_DOUBLE,root, Pa.cart_comm_sub_z)
        return

    cpdef stats_io(self, Surface.SurfaceBase Sur, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        NS.write_ts('surface_temperature', Sur.T_surface, Pa)
        return
