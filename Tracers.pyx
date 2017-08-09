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

from NetCDFIO cimport NetCDFIO_Stats
import cython

from libc.math cimport fmax
cimport numpy as np
import numpy as np
include "parameters.pxi"

import cython

cdef extern from "thermodynamic_functions.h":
    inline double pv_c(double p0, double qt, double qv) nogil

def TracersFactory(namelist):
    try:
        use_tracers = namelist['tracers']['use_tracers']
    except:
        use_tracers = False
    if use_tracers:
        return UpdraftTracers(namelist)
    else:
        return TracersNone()


cdef class TracersNone:
    def __init__(self):
        return
    cpdef initialize(self, Grid.Grid Gr,  PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa):
        return
    cpdef update_cleanup(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa):
        return
    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return


cdef class UpdraftTracers:
    def __init__(self, namelist):

        if namelist['microphysics']['scheme'] == 'None_SA' or namelist['microphysics']['scheme'] == 'SB_Liquid':
            self.lcl_tracers = True
        else:
            self.lcl_tracers = False

        self.index_lcl = 0

        return

    cpdef initialize(self, Grid.Grid Gr,  PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        # Assemble a dictionary with the tracer information
        # Can be expanded for different init heights or timescales
        self.tracer_dict = {}
        self.tracer_dict['surface'] = {}
        self.tracer_dict['surface']['c_srf_15'] = {}
        self.tracer_dict['surface']['c_srf_15']['timescale'] = 15.0 * 60.0
        if self.lcl_tracers:
            self.tracer_dict['lcl'] = {}
            self.tracer_dict['lcl']['c_lcl_15'] = {}
            self.tracer_dict['lcl']['c_lcl_15']['timescale'] = 15.0 * 60.0

        for var in self.tracer_dict['surface'].keys():
            PV.add_variable(var, '-', var, 'tracer diagnostics' , "sym", "scalar", Pa)

        if self.lcl_tracers:
            for var in self.tracer_dict['lcl'].keys():
                PV.add_variable(var, '-', var, 'tracer diagnostics', "sym", "scalar", Pa)
            NS.add_ts('grid_lcl', Gr, Pa )

        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i,j,k,ishift,jshift,ijk
            Py_ssize_t var_shift
            double tau

        # Set the source term
        for level in self.tracer_dict.keys():
            for var in self.tracer_dict[level].keys():
                var_shift = PV.get_varshift(Gr, var)
                tau = self.tracer_dict[level][var]['timescale']
                with nogil:
                    for i in xrange(Gr.dims.npg):
                        PV.tendencies[var_shift + i] += -fmax(PV.values[var_shift + i],0.0)/tau


        #Below we assume domain uses only x-y decomposition, for the general case we should use pencils
        #but we don't address that complication for now

        # Set the value of the surface based tracers
        for var in self.tracer_dict['surface'].keys():
            var_shift = PV.get_varshift(Gr, var)
            with nogil:
                for i in xrange(Gr.dims.nlg[0]):
                    for j in xrange(Gr.dims.nlg[1]):
                        ijk = i * istride + j * jstride + Gr.dims.gw
                        PV.values[var_shift + ijk] = 1.0


        # Find surface thermodynamic properties to compute LCL (lifting condensation level)
        # Ref: M. G. Lawrence, "The relationship between relative humidity and the dew point
        # temperature in moist air: A simple conversion and applications", Bull. Am. Meteorol. Soc., 86, 225-233, 2005
        cdef:
            double [:] T_dew = np.zeros(Gr.dims.nl[0]*Gr.dims.nl[1],dtype=np.double,order='c')
            double [:] z_lcl = np.zeros(Gr.dims.nl[0]*Gr.dims.nl[1],dtype=np.double,order='c')
            Py_ssize_t t_shift, qv_shift, qt_shift
            Py_ssize_t  count = 0
            double B1 = 243.04  # C
            double A1 = 17.625 # no units
            double C1 = 610.94 # Pa
            double CtoK = 273.15 # additive celsius to kelvin conversion factor
            double vapor_pressure
            double nxny_i =1.0/( Gr.dims.n[0] * Gr.dims.n[1])
            double lcl_, lcl



        if self.lcl_tracers:
            t_shift = DV.get_varshift(Gr,'temperature')
            qt_shift = PV.get_varshift(Gr, 'qt')
            qv_shift = DV.get_varshift(Gr,'qv')
            with nogil:
                for i in xrange(Gr.dims.gw, Gr.dims.nlg[0]-Gr.dims.gw):
                    for j in xrange(Gr.dims.gw, Gr.dims.nlg[1]-Gr.dims.gw):
                        ijk = i * istride + j * jstride + Gr.dims.gw
                        vapor_pressure = pv_c(Ref.p0_half[Gr.dims.gw], PV.values[qt_shift + ijk], DV.values[qv_shift + ijk])
                        T_dew[count] = B1 * log(vapor_pressure/C1)/(A1-log(vapor_pressure/C1))  + CtoK
                        z_lcl[count] = 125.0 * (DV.values[t_shift+ijk] - T_dew[count])
                        count += 1
            lcl_ = np.sum(z_lcl) * nxny_i
            lcl = Pa.domain_scalar_sum(lcl_)
            # Pa.root_print('LCL is ' + str(lcl))

            for k in xrange(Gr.dims.nlg[2]-Gr.dims.gw, Gr.dims.gw-1, -1):
                if Gr.zl_half[k] <= lcl:
                    self.index_lcl = k
                    break

            # Pa.root_print('Grid LCL ' + str(Gr.zl_half[self.index_lcl]))
            for var in self.tracer_dict['lcl'].keys():
                var_shift = PV.get_varshift(Gr, var)
                with nogil:
                    for i in xrange( Gr.dims.nlg[0]):
                        for j in xrange( Gr.dims.nlg[1]):
                            ijk = i * istride + j * jstride + self.index_lcl
                            PV.values[var_shift + ijk] = 1.0


        return


    cpdef update_cleanup(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i,j,k,ishift,jshift,ijk
            Py_ssize_t var_shift


        #Below we assume domain uses only x-y decomposition, for the general case we should use pencils
        #but we don't address that complication for now

        # Set the value of the surface based tracers
        for var in self.tracer_dict['surface'].keys():
            var_shift = PV.get_varshift(Gr, var)
            with nogil:
                for i in xrange(Gr.dims.nlg[0]):
                    for j in xrange(Gr.dims.nlg[1]):
                        ijk = i * istride + j * jstride + Gr.dims.gw
                        PV.tendencies[var_shift + ijk] = 0.0
                        for k in xrange( Gr.dims.nlg[2]):
                            ijk = i * istride + j * jstride + k
                            PV.values[var_shift + ijk] = fmax(PV.values[var_shift + ijk],0.0)

        for var in self.tracer_dict['lcl'].keys():
            var_shift = PV.get_varshift(Gr, var)
            with nogil:
                for i in xrange( Gr.dims.nlg[0]):
                    for j in xrange( Gr.dims.nlg[1]):
                        ijk = i * istride + j * jstride + self.index_lcl
                        PV.tendencies[var_shift + ijk] = 0.0
                        for k in xrange( Gr.dims.nlg[2]):
                            ijk = i * istride + j * jstride + k
                            PV.values[var_shift + ijk] = fmax(PV.values[var_shift + ijk],0.0)

        return


    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        if self.lcl_tracers:
            NS.write_ts('grid_lcl',Gr.zl_half[self.index_lcl], Pa)
        return