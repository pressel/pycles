#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport ParallelMPI
cimport TimeStepping
cimport Radiation
from Thermodynamics cimport LatentHeat,ClausiusClapeyron
from NetCDFIO cimport NetCDFIO_Stats
import cython
from thermodynamic_functions import exner, cpm
from thermodynamic_functions cimport cpm_c, pv_c, pd_c, exner_c
from entropies cimport sv_c, sd_c
from libc.math cimport sqrt, log, fabs,atan, exp, fmax
cimport numpy as np
import numpy as np
include "parameters.pxi"

import cython

cdef class SurfaceBudget:
    def __init__(self, namelist, init_sst):

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

        self.water_depth = self.water_depth_initial
        self.sst = init_sst
        return

    def __getattr__(self, item):
        if item == 'sst':
            return self.sst
        else:
            print('what???')
            return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return

    cpdef update(self, Radiation.Radiation Ra, ParallelMPI.ParallelMPI Pa,  double time, double shf, double lhf):
        if time > self.water_depth_time:
            self.water_depth = self.water_depth_final
        else:
            self.water_depth = self.water_depth_initial

        Pa.root_print('Water depth '+str(self.water_depth) )

        cdef double rho_liquid = 1000.0
        # cdef double net_flux =  -self.ocean_heat_flux - Ra.srf_lw_up - shf - lhf + Ra.srf_lw_down + Ra.srf_sw_down
        # cdef double tendency = net_flux/cl/rho_liquid/self.water_depth
        # self.sst  += tendency



        return
    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return
