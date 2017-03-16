#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

import netCDF4 as nc
cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
from thermodynamic_functions cimport cpm_c, pv_c, pd_c, exner_c
from entropies cimport sv_c, sd_c, s_tendency_c
import numpy as np
import cython
from libc.math cimport fabs, sin, cos
from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI
cimport Lookup
from Thermodynamics cimport LatentHeat, ClausiusClapeyron

# import pylab as plt
include 'parameters.pxi'

cdef class Forcing:
    def __init__(self, namelist, LatentHeat LH, ParallelMPI.ParallelMPI Pa):
        casename = namelist['meta']['casename']
        if casename == 'SullivanPatton':
            self.scheme = ForcingSullivanPatton()
        elif casename == 'Bomex':
            self.scheme = ForcingBomex()
        elif casename == 'Gabls':
            self.scheme = ForcingGabls()
        elif casename == 'DYCOMS_RF01':
            self.scheme = ForcingDyCOMS_RF01(casename)
        elif casename == 'DYCOMS_RF02':
            #Forcing for DYCOMS_RF02 is same as DYCOMS_RF01
            self.scheme = ForcingDyCOMS_RF01(casename)
        elif casename == 'SMOKE':
            self.scheme = ForcingNone()
        elif casename == 'Rico':
            self.scheme = ForcingRico()
        elif casename == 'StableBubble':
            self.scheme = ForcingNone()
        elif casename == 'CGILS':
            self.scheme = ForcingCGILS(namelist, Pa)
        elif casename == 'ZGILS':
            self.scheme = ForcingZGILS(namelist, LH, Pa)
        else:
            Pa.root_print('No focing for casename: ' +  casename)
            Pa.root_print('Killing simulation now!!!')
            Pa.kill()
        return

    cpdef initialize(self, Grid.Grid Gr,ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.scheme.initialize(Gr, Ref, NS, Pa)
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        self.scheme.update(Gr, Ref, PV, DV, Pa)
        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        self.scheme.stats_io(Gr, Ref, PV, DV, NS, Pa)

        return


cdef class ForcingNone:
    def __init__(self):
        pass
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return

cdef class ForcingBomex:
    def __init__(self):
        return

    cpdef initialize(self, Grid.Grid Gr,ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.ug = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.vg = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.dtdt = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.dqtdt = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.subsidence = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.coriolis_param = 0.376e-4 #s^{-1}

        cdef:
            Py_ssize_t k


        with nogil:
            for k in xrange(Gr.dims.nlg[2]):
                self.ug[k] = -10.0 + (1.8e-3)*Gr.zpl_half[k]

                #Set large scale cooling
                # Convert given form of tendencies (theta) to temperature tendency
                if Gr.zpl_half[k] <= 1500.0:
                    self.dtdt[k] = (-2.0/(3600 * 24.0))  * exner_c(Ref.p0_half[k])     #K/s
                if Gr.zpl_half[k] > 1500.0:
                    self.dtdt[k] = (-2.0/(3600 * 24.0) + (Gr.zpl_half[k] - 1500.0)
                                    * (0.0 - -2.0/(3600 * 24.0)) / (3000.0 - 1500.0)) * exner_c(Ref.p0_half[k])

                #Set large scale drying
                if Gr.zpl_half[k] <= 300.0:
                    self.dqtdt[k] = -1.2e-8   #kg/(kg * s)
                if Gr.zpl_half[k] > 300.0 and Gr.zpl_half[k] <= 500.0:
                    self.dqtdt[k] = -1.2e-8 + (Gr.zpl_half[k] - 300.0)*(0.0 - -1.2e-8)/(500.0 - 300.0) #kg/(kg * s)

                #Set large scale subsidence
                if Gr.zpl_half[k] <= 1500.0:
                    self.subsidence[k] = 0.0 + Gr.zpl_half[k]*(-0.65/100.0 - 0.0)/(1500.0 - 0.0)
                if Gr.zpl_half[k] > 1500.0 and Gr.zpl_half[k] <= 2100.0:
                    self.subsidence[k] = -0.65/100 + (Gr.zpl_half[k] - 1500.0)* (0.0 - -0.65/100.0)/(2100.0 - 1500.0)


        #Initialize Statistical Output
        NS.add_profile('s_subsidence_tendency', Gr, Pa)
        NS.add_profile('qt_subsidence_tendency', Gr, Pa)
        NS.add_profile('u_subsidence_tendency', Gr, Pa)
        NS.add_profile('v_subsidence_tendency', Gr, Pa)
        NS.add_profile('u_coriolis_tendency', Gr, Pa)
        NS.add_profile('v_coriolis_tendency',Gr, Pa)

        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i,j,k,ishift,jshift,ijk
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            double qt, qv, p0, t
            double [:] umean = Pa.HorizontalMean(Gr, &PV.values[u_shift])
            double [:] vmean = Pa.HorizontalMean(Gr, &PV.values[v_shift])




        #Apply Coriolis Forcing
        large_scale_p_gradient(&Gr.dims, &umean[0], &vmean[0], &PV.tendencies[u_shift],
                       &PV.tendencies[v_shift], &self.ug[0], &self.vg[0], self.coriolis_param, Ref.u0, Ref.v0)

        #Apply large scale source terms
        with nogil:
            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        p0 = Ref.p0_half[k]
                        qt = PV.values[qt_shift + ijk]
                        qv = qt - DV.values[ql_shift + ijk]
                        t  = DV.values[t_shift + ijk]
                        PV.tendencies[s_shift + ijk] += s_tendency_c(p0,qt,qv,t, self.dqtdt[k], self.dtdt[k])
                        PV.tendencies[qt_shift + ijk] += self.dqtdt[k]

        apply_subsidence(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence[0], &PV.values[s_shift], &PV.tendencies[s_shift])
        apply_subsidence(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence[0], &PV.values[qt_shift], &PV.tendencies[qt_shift])
        apply_subsidence(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence[0], &PV.values[u_shift], &PV.tendencies[u_shift])
        apply_subsidence(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence[0], &PV.values[v_shift], &PV.tendencies[v_shift])
        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            double [:] tmp_tendency  = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
            double [:] tmp_tendency_2 = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
            double [:] mean_tendency = np.empty((Gr.dims.nlg[2],),dtype=np.double,order='c')
            double [:] mean_tendency_2 = np.zeros((Gr.dims.nlg[2]),dtype=np.double,order='c')
            double [:] umean = Pa.HorizontalMean(Gr, &PV.values[u_shift])
            double [:] vmean = Pa.HorizontalMean(Gr, &PV.values[v_shift])

        #Output subsidence tendencies
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[s_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('s_subsidence_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        tmp_tendency[:] = 0.0
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[qt_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('qt_subsidence_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        tmp_tendency[:] = 0.0
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[u_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('u_subsidence_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        tmp_tendency[:] = 0.0
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[v_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('v_subsidence_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        #Output Coriolis tendencies
        tmp_tendency[:] = 0.0
        large_scale_p_gradient(&Gr.dims, &umean[0], &vmean[0], &tmp_tendency[0],
                       &tmp_tendency_2[0], &self.ug[0], &self.vg[0], self.coriolis_param, Ref.u0, Ref.v0)
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        mean_tendency_2 = Pa.HorizontalMean(Gr,&tmp_tendency_2[0])
        NS.write_profile('u_coriolis_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('v_coriolis_tendency',mean_tendency_2[Gr.dims.gw:-Gr.dims.gw],Pa)

        return

cdef class ForcingSullivanPatton:
    def __init__(self):

        return
    cpdef initialize(self, Grid.Grid Gr,ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.ug = np.ones(Gr.dims.nlg[2],dtype=np.double, order='c') #m/s
        self.vg = np.zeros(Gr.dims.nlg[2],dtype=np.double, order='c')  #m/s
        self.coriolis_param = 1.0e-4 #s^{-1}

        NS.add_profile('u_coriolis_tendency', Gr, Pa)
        NS.add_profile('v_coriolis_tendency',Gr, Pa)
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')

        coriolis_force(&Gr.dims,&PV.values[u_shift],&PV.values[v_shift],&PV.tendencies[u_shift],
                       &PV.tendencies[v_shift],&self.ug[0], &self.vg[0],self.coriolis_param, Ref.u0, Ref.v0  )
        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            double [:] tmp_tendency  = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
            double [:] tmp_tendency_2 = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
            double [:] mean_tendency = np.empty((Gr.dims.nlg[2],),dtype=np.double,order='c')
            double [:] mean_tendency_2 = np.zeros((Gr.dims.nlg[2]),dtype=np.double,order='c')

        #Only need to output coriolis_forcing
        coriolis_force(&Gr.dims,&PV.values[u_shift],&PV.values[v_shift],&tmp_tendency[0],
                       &tmp_tendency_2[0],&self.ug[0], &self.vg[0],self.coriolis_param, Ref.u0, Ref.v0)
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        mean_tendency_2 = Pa.HorizontalMean(Gr,&tmp_tendency_2[0])
        NS.write_profile('u_coriolis_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('v_coriolis_tendency',mean_tendency_2[Gr.dims.gw:-Gr.dims.gw],Pa)

        return


cdef class ForcingGabls:
    def __init__(self):
        return

    cpdef initialize(self, Grid.Grid Gr,ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.ug = np.ones(Gr.dims.nlg[2],dtype=np.double,order='c') * 8.0
        self.vg = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.coriolis_param = 1.39e-4 #s^{-1}
        NS.add_profile('u_coriolis_tendency', Gr, Pa)
        NS.add_profile('v_coriolis_tendency',Gr, Pa)

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')

        coriolis_force(&Gr.dims,&PV.values[u_shift],&PV.values[v_shift],&PV.tendencies[u_shift],
                       &PV.tendencies[v_shift],&self.ug[0], &self.vg[0],self.coriolis_param, Ref.u0, Ref.v0  )
        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            double [:] tmp_tendency  = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
            double [:] tmp_tendency_2 = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
            double [:] mean_tendency = np.empty((Gr.dims.npg,),dtype=np.double,order='c')
            double [:] mean_tendency_2 = np.zeros((Gr.dims.npg),dtype=np.double,order='c')

        #Only need to output coriolis_forcing
        coriolis_force(&Gr.dims,&PV.values[u_shift],&PV.values[v_shift],&tmp_tendency[0],
                       &tmp_tendency_2[0],&self.ug[0], &self.vg[0],self.coriolis_param, Ref.u0, Ref.v0)
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        mean_tendency_2 = Pa.HorizontalMean(Gr,&tmp_tendency_2[0])
        NS.write_profile('u_coriolis_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('v_coriolis_tendency',mean_tendency_2[Gr.dims.gw:-Gr.dims.gw],Pa)

        return



cdef class ForcingDyCOMS_RF01:
    def __init__(self,casename):
        self.divergence = 3.75e-6
        self.coriolis_param = 2.0 * omega * sin(31.5 * pi / 180.0 )
        if casename == 'DYCOMS_RF02':
            self.rf02_flag = True
        else:
            self.rf02_flag = False

        return

    cpdef initialize(self, Grid.Grid Gr,ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t k

        self.subsidence = np.empty((Gr.dims.nlg[2]),dtype=np.double, order='c')
        self.ug = np.empty((Gr.dims.nlg[2]),dtype=np.double, order='c')
        self.vg = np.empty((Gr.dims.nlg[2]),dtype=np.double, order='c')

        if self.rf02_flag:
            with nogil:
                for k in range(Gr.dims.nlg[2]):
                    self.subsidence[k] = -Gr.zpl_half[k] * self.divergence
                    self.ug[k] = 3.0 + 4.3*Gr.zpl_half[k]/1000.0
                    self.vg[k] = -9.0 + 5.6 * Gr.zpl_half[k]/1000.0
        else:
            with nogil:
                for k in range(Gr.dims.nlg[2]):
                    self.subsidence[k] = -Gr.zpl_half[k] * self.divergence
                    self.ug[k] = 7.0
                    self.vg[k] = -5.5


        #Initialize Statistical Output
        NS.add_profile('s_subsidence_tendency', Gr, Pa)
        NS.add_profile('qt_subsidence_tendency', Gr, Pa)
        NS.add_profile('u_subsidence_tendency', Gr, Pa)
        NS.add_profile('v_subsidence_tendency', Gr, Pa)
        NS.add_profile('u_coriolis_tendency', Gr, Pa)
        NS.add_profile('v_coriolis_tendency',Gr, Pa)

        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr,'qt')

        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[s_shift],&PV.tendencies[s_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[qt_shift],&PV.tendencies[qt_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[u_shift],&PV.tendencies[u_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[v_shift],&PV.tendencies[v_shift])

        coriolis_force(&Gr.dims,&PV.values[u_shift],&PV.values[v_shift],&PV.tendencies[u_shift],
                       &PV.tendencies[v_shift],&self.ug[0], &self.vg[0],self.coriolis_param, Ref.u0, Ref.v0  )

        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            double [:] tmp_tendency  = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
            double [:] tmp_tendency_2 = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
            double [:] mean_tendency = np.empty((Gr.dims.nlg[2],),dtype=np.double,order='c')
            double [:] mean_tendency_2 = np.zeros((Gr.dims.nlg[2]),dtype=np.double,order='c')
            double [:] umean = Pa.HorizontalMean(Gr, &PV.values[u_shift])
            double [:] vmean = Pa.HorizontalMean(Gr, &PV.values[v_shift])

        #Output subsidence tendencies
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[s_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('s_subsidence_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        tmp_tendency[:] = 0.0
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[qt_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('qt_subsidence_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        tmp_tendency[:] = 0.0
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[u_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('u_subsidence_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        tmp_tendency[:] = 0.0
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[v_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('v_subsidence_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        #Output Coriolis tendencies
        tmp_tendency[:] = 0.0
        large_scale_p_gradient(&Gr.dims, &umean[0], &vmean[0], &tmp_tendency[0],
                       &tmp_tendency_2[0], &self.ug[0], &self.vg[0], self.coriolis_param, Ref.u0, Ref.v0)
        coriolis_force(&Gr.dims,&PV.values[u_shift],&PV.values[v_shift],&tmp_tendency[0],
                       &tmp_tendency_2[0],&self.ug[0], &self.vg[0],self.coriolis_param, Ref.u0, Ref.v0)
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        mean_tendency_2 = Pa.HorizontalMean(Gr,&tmp_tendency_2[0])
        NS.write_profile('u_coriolis_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('v_coriolis_tendency',mean_tendency_2[Gr.dims.gw:-Gr.dims.gw],Pa)

        return

cdef class ForcingRico:
    def __init__(self):
        latitude = 18.0 # degrees
        self.coriolis_param = 2.0 * omega * sin(latitude * pi / 180.0 )
        self.momentum_subsidence = 0
        return

    cpdef initialize(self, Grid.Grid Gr,ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        cdef Py_ssize_t k

        self.subsidence = np.empty((Gr.dims.nlg[2]),dtype=np.double, order='c')
        self.ug = np.empty(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.vg = np.empty(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.dtdt = np.ones(Gr.dims.nlg[2],dtype=np.double,order='c') * -2.5/86400.0 #Here this is theta forcing
        self.dqtdt = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')

        # Convert given theta forcing to temperature forcing
        with nogil:
            for k in range(Gr.dims.nlg[2]):
                self.dtdt[k] = self.dtdt[k] * exner_c(Ref.p0_half[k])
                if Gr.zpl_half[k] <= 2260.0:
                    self.subsidence[k] = -(0.005/2260.0) * Gr.zpl_half[k]
                else:
                    self.subsidence[k] = -0.005
                if Gr.zpl_half[k]<=2980.0:
                    self.dqtdt[k] = (-1.0 + 1.3456/2980.0 * Gr.zpl_half[k])/86400.0/1000.0
                else:
                    self.dqtdt[k] = 0.3456/86400.0/1000.0
                self.ug[k] = -9.9 + 2.0e-3 * Gr.zpl_half[k]
                self.vg[k] = -3.8


        NS.add_profile('u_coriolis_tendency', Gr, Pa)
        NS.add_profile('v_coriolis_tendency',Gr, Pa)
        NS.add_profile('s_subsidence_tendency', Gr, Pa)
        NS.add_profile('qt_subsidence_tendency', Gr, Pa)
        if self.momentum_subsidence == 1:
            NS.add_profile('u_subsidence_tendency', Gr, Pa)
            NS.add_profile('v_subsidence_tendency', Gr, Pa)

        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i, j, k, ishift, jshift, ijk

            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr,'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            double qt, qv, p0, t

        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[s_shift],&PV.tendencies[s_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[qt_shift],&PV.tendencies[qt_shift])
        if self.momentum_subsidence == 1:
            apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[u_shift],&PV.tendencies[u_shift])
            apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[v_shift],&PV.tendencies[v_shift])


                #Apply large scale source terms
        with nogil:
            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        p0 = Ref.p0_half[k]
                        qt = PV.values[qt_shift + ijk]
                        qv = qt - DV.values[ql_shift + ijk]
                        t  = DV.values[t_shift + ijk]
                        PV.tendencies[s_shift + ijk] += s_tendency_c(p0, qt, qv, t, self.dqtdt[k], self.dtdt[k])
                        PV.tendencies[qt_shift + ijk] += self.dqtdt[k]

        coriolis_force(&Gr.dims,&PV.values[u_shift],&PV.values[v_shift],&PV.tendencies[u_shift],
                       &PV.tendencies[v_shift],&self.ug[0], &self.vg[0],self.coriolis_param, Ref.u0, Ref.v0  )

        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        return







cdef class ForcingCGILS:
    def __init__(self, namelist, ParallelMPI.ParallelMPI Pa):
        try:
            self.loc = namelist['meta']['CGILS']['location']
        except:
            Pa.root_print('Must provide a CGILS location (6/11/12) in namelist')
            Pa.kill()
        try:
            self.is_p2 = namelist['meta']['CGILS']['P2']
        except:
            Pa.root_print('Must specify if CGILS run is perturbed')
            Pa.kill()


        try:
            self.is_ctl_omega = namelist['meta']['CGILS']['CTL_omega']
        except:
            self.is_ctl_omega = False
            if self.is_p2:
                Pa.root_print('ForcingCGILS: Assuming perturbed omega is used')


        if self.loc == 12:
            self.z_relax = 1200.0
            self.z_relax_plus = 1500.0
        elif self.loc == 11:
            self.z_relax = 2500.0
            self.z_relax_plus = 3000.0
        elif self.loc == 6:
            self.z_relax = 4000.0
            self.z_relax_plus = 4800.0

        self.tau_inverse = 1.0/(60.0*60.0) # inverse of  max nudging timescale, 1 hr, for all cases
        self.tau_vel_inverse = 1.0/(10.0*60.0) # nudging timescale of horizontal winds


        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        self.dtdt = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.dqtdt = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.subsidence = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')


        if self.is_p2:
            file = './CGILSdata/p2k_s'+str(self.loc)+'.nc'
        else:
            file = './CGILSdata/ctl_s'+str(self.loc)+'.nc'

        data = nc.Dataset(file, 'r')

        pressure = data.variables['lev'][:]
        divq_data = data.variables['divq'][0,:,0,0]
        divT_data = data.variables['divT'][0,:,0,0]
        omega_data = data.variables['omega'][0,:,0,0]
        temperature_data = data.variables['T'][0,:,0,0]
        q_data = data.variables['q'][0,:,0,0]
        u_data = data.variables['u'][0,:,0,0]
        v_data = data.variables['v'][0,:,0,0]
        Ps = data.variables['Ps'][0,0,0]
        n_data = np.shape(pressure)[0] - 1
        data.close()

        if self.is_p2 and  self.is_ctl_omega:
            file = './CGILSdata/ctl_s'+str(self.loc)+'.nc'
            data = nc.Dataset(file, 'r')
            omega_data = data.variables['omega'][0,:,0,0]
            data.close()

        # Now we need to use interpolation to get the forcing at the LES grid.
        # We need to extrapolate the data arrays to the surface
        omega_right = (omega_data[n_data-1] - omega_data[n_data])/(pressure[n_data-1]-pressure[n_data])*(Ps-pressure[n_data]) + omega_data[n_data]
        omega_data = np.append(omega_data,omega_right)
        divq_right = (divq_data[n_data-1] - divq_data[n_data])/(pressure[n_data-1]-pressure[n_data])*(Ps-pressure[n_data]) + divq_data[n_data]
        divq_data = np.append(divq_data,divq_right)
        divT_right = (divT_data[n_data-1] - divT_data[n_data])/(pressure[n_data-1]-pressure[n_data])*(Ps-pressure[n_data]) + divT_data[n_data]
        divT_data = np.append(divT_data,divT_right)
        temperature_right = (temperature_data[n_data-1] - temperature_data[n_data])/(pressure[n_data-1]-pressure[n_data])*(Ps-pressure[n_data]) + temperature_data[n_data]
        temperature_data = np.append(temperature_data, temperature_right)
        q_right = (q_data[n_data-1] - q_data[n_data])/(pressure[n_data-1]-pressure[n_data])*(Ps-pressure[n_data]) + q_data[n_data]
        q_data = np.append(q_data, q_right)
        u_right = (u_data[n_data-1] - u_data[n_data])/(pressure[n_data-1]-pressure[n_data])*(Ps-pressure[n_data]) + u_data[n_data]
        u_data = np.append(u_data,u_right)
        v_right = (v_data[n_data-1] - v_data[n_data])/(pressure[n_data-1]-pressure[n_data])*(Ps-pressure[n_data]) + v_data[n_data]
        v_data = np.append(v_data, v_right)

        pressure = np.append(pressure, Ps)

        # interpolate the subsidence profile from the data (below we convert from dp--> dz)
        self.subsidence = np.array(np.interp(Ref.p0_half, pressure, omega_data),dtype=np.double, order='c')
        # interpolate large scale advection forcings
        self.dqtdt = np.array(np.interp(Ref.p0_half, pressure, divq_data),dtype=np.double, order='c')
        self.dtdt = np.array(np.interp(Ref.p0_half, pressure, divT_data),dtype=np.double, order='c')

        # interpolate the reference profiles
        self.nudge_qt = np.array(np.interp(Ref.p0_half, pressure, q_data),dtype=np.double, order='c')
        self.nudge_temperature = np.array(np.interp(Ref.p0_half, pressure, temperature_data),dtype=np.double, order='c')
        self.nudge_u = np.array(np.interp(Ref.p0_half, pressure, u_data),dtype=np.double, order='c')
        self.nudge_v = np.array(np.interp(Ref.p0_half, pressure, v_data),dtype=np.double, order='c')

        # Initialize arrays for the  nudging related source terms
        self.source_qt_floor = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.source_qt_nudge = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.source_t_nudge = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.source_u_nudge = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.source_v_nudge = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')

        self.source_s_nudge = np.zeros(Gr.dims.npg,dtype=np.double,order='c')
        self.s_ls_adv = np.zeros(Gr.dims.npg,dtype=np.double,order='c')

        # convert subsidence velocity to physical space
        cdef Py_ssize_t k
        with nogil:
            for k in xrange(Gr.dims.nlg[2]):
                self.subsidence[k] = -self.subsidence[k]*Ref.alpha0_half[k]/g


        # Obtain the moisture floor and find the max index corresponding to z <= 1300 m

        self.qt_floor = np.interp(1300.0, Gr.zpl_half, self.nudge_qt)
        for k in range(Gr.dims.gw, Gr.dims.nlg[2]-Gr.dims.gw):
            if Gr.zpl_half[k] > 1300.0:
                break
            self.floor_index = k


        # initialize the inverse timescale coefficient arrays for nudging
        self.gamma_zhalf = np.zeros((Gr.dims.nlg[2]),dtype=np.double,order='c')
        self.gamma_z = np.zeros((Gr.dims.nlg[2]), dtype=np.double, order='c')

        with nogil:
            for k in range(Gr.dims.nlg[2]):
                if Gr.zpl_half[k] < self.z_relax:
                    self.gamma_zhalf[k] = 0.0
                elif Gr.zpl_half[k] > self.z_relax_plus:
                    self.gamma_zhalf[k] = self.tau_inverse
                else:
                    self.gamma_zhalf[k] = 0.5*self.tau_inverse * (1.0 - cos(pi* (Gr.zpl_half[k]-self.z_relax)/(self.z_relax_plus-self.z_relax)))

                if Gr.zpl[k] < self.z_relax:
                    self.gamma_z[k] = 0.0
                elif Gr.zpl[k] > self.z_relax_plus:
                    self.gamma_z[k] = self.tau_inverse
                else:
                    self.gamma_z[k] = 0.5*self.tau_inverse * (1.0 - cos(pi* (Gr.zpl[k]-self.z_relax)/(self.z_relax_plus-self.z_relax)))

        #Initialize Statistical Output
        NS.add_profile('s_subsidence_tendency', Gr, Pa)
        NS.add_profile('qt_subsidence_tendency', Gr, Pa)
        NS.add_profile('qt_floor_nudging', Gr, Pa)
        NS.add_profile('qt_ref_nudging', Gr, Pa)
        NS.add_profile('temperature_ref_nudging', Gr, Pa)
        NS.add_profile('s_ref_nudging', Gr, Pa)
        NS.add_profile('s_ls_adv', Gr, Pa)

        # Add profiles of fixed forcing profiles that do not depend on time
        NS.add_reference_profile('subsidence_velocity', Gr, Pa)
        NS.write_reference_profile('subsidence_velocity', self.subsidence[Gr.dims.gw:-Gr.dims.gw], Pa)

        NS.add_reference_profile('temperature_ls_adv', Gr, Pa)
        NS.write_reference_profile('temperature_ls_adv', self.dtdt[Gr.dims.gw:-Gr.dims.gw], Pa)

        NS.add_reference_profile('qt_ls_adv', Gr, Pa)
        NS.write_reference_profile('qt_ls_adv', self.dqtdt[Gr.dims.gw:-Gr.dims.gw], Pa)

        NS.add_reference_profile('u_ref_profile', Gr, Pa)
        NS.write_reference_profile('u_ref_profile', self.nudge_u[Gr.dims.gw:-Gr.dims.gw], Pa)

        NS.add_reference_profile('v_ref_profile', Gr, Pa)
        NS.write_reference_profile('v_ref_profile', self.nudge_v[Gr.dims.gw:-Gr.dims.gw], Pa)


        NS.add_reference_profile('temperature_ref_profile', Gr, Pa)
        NS.write_reference_profile('temperature_ref_profile', self.nudge_temperature[Gr.dims.gw:-Gr.dims.gw], Pa)

        NS.add_reference_profile('qt_ref_profile', Gr, Pa)
        NS.write_reference_profile('qt_ref_profile', self.nudge_qt[Gr.dims.gw:-Gr.dims.gw], Pa)



        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i,j,k,ishift,jshift,ijk
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            double qt, qv, p0, t, qt_floor_nudge

            double [:] qtmean = Pa.HorizontalMean(Gr, &PV.values[qt_shift])
            double [:] tmean = Pa.HorizontalMean(Gr, &DV.values[t_shift])
            double [:] umean = Pa.HorizontalMean(Gr, &PV.values[u_shift])
            double [:] vmean = Pa.HorizontalMean(Gr, &PV.values[v_shift])

        # Apply subsidence
        apply_subsidence(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence[0], &PV.values[s_shift], &PV.tendencies[s_shift])
        apply_subsidence(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence[0], &PV.values[qt_shift], &PV.tendencies[qt_shift])

       # Calculate nudging
        with nogil:
            for k in xrange(kmin, kmax):
                # Nudge s, qt to reference profiles in free troposphere
                self.source_t_nudge[k] = -(tmean[k] - self.nudge_temperature[k]) * self.gamma_z[k]
                self.source_qt_nudge[k]= -(qtmean[k] - self.nudge_qt[k]) * self.gamma_z[k]
                # Nudge mean wind profiles through entire depth
                self.source_u_nudge[k] = -(umean[k] + Ref.u0 - self.nudge_u[k]) * self.tau_vel_inverse
                self.source_v_nudge[k] = -(vmean[k] + Ref.v0 - self.nudge_v[k]) * self.tau_vel_inverse

        # Moisture floor nudging for S12 case
        if self.loc == 12:
            with nogil:
                for k in xrange(kmin, self.floor_index):
                    if qtmean[k] < self.qt_floor:
                        self.source_qt_floor[k] = -(qtmean[k] - self.qt_floor) * self.tau_inverse
                    else:
                        self.source_qt_floor[k] = 0.0

        cdef double total_qt_source
        cdef double total_t_source

        #Apply large scale source terms
        with nogil:
            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        p0 = Ref.p0_half[k]
                        qt = PV.values[qt_shift + ijk]
                        qv = qt - DV.values[ql_shift + ijk]
                        t  = DV.values[t_shift + ijk]
                        total_qt_source = self.dqtdt[k] + self.source_qt_floor[k] + self.source_qt_nudge[k]
                        total_t_source  = self.dtdt[k]  + self.source_t_nudge[k]
                        PV.tendencies[s_shift  + ijk] += s_tendency_c(p0, qt, qv, t, total_qt_source, total_t_source)

                        PV.tendencies[qt_shift + ijk] += total_qt_source
                        PV.tendencies[u_shift  + ijk] += self.source_u_nudge[k]
                        PV.tendencies[v_shift  + ijk] += self.source_v_nudge[k]

                        self.source_s_nudge[ijk] = s_tendency_c(p0, qt, qv, t, self.source_qt_nudge[k] + self.source_qt_floor[k],
                                                                self.source_t_nudge[k] )

                        self.s_ls_adv[ijk]= s_tendency_c(p0, qt, qv, t, self.dqtdt[k], self.dtdt[k])


        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            double [:] tmp_tendency  = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
            double [:] tmp_tendency_2 = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
            double [:] mean_tendency = np.empty((Gr.dims.nlg[2],),dtype=np.double,order='c')
            double [:] mean_tendency_2 = np.zeros((Gr.dims.nlg[2]),dtype=np.double,order='c')
            double [:] umean = Pa.HorizontalMean(Gr, &PV.values[u_shift])
            double [:] vmean = Pa.HorizontalMean(Gr, &PV.values[v_shift])

        #Output subsidence tendencies
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[s_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('s_subsidence_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        tmp_tendency[:] = 0.0
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[qt_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('qt_subsidence_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        mean_tendency = Pa.HorizontalMean(Gr,&self.source_s_nudge[0])
        NS.write_profile('s_ref_nudging',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        mean_tendency = Pa.HorizontalMean(Gr,&self.s_ls_adv[0])
        NS.write_profile('s_ls_adv',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)



        NS.write_profile('qt_floor_nudging',self.source_qt_floor[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('qt_ref_nudging',self.source_qt_nudge[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('temperature_ref_nudging',self.source_t_nudge[Gr.dims.gw:-Gr.dims.gw],Pa)

        return





cdef class ForcingZGILS:
    def __init__(self, namelist, LatentHeat LH, ParallelMPI.ParallelMPI Pa):
        try:
            self.loc = namelist['meta']['ZGILS']['location']
        except:
            Pa.root_print('FORCING: Must provide a ZGILS location (6/11/12) in namelist')
            Pa.kill()
        if self.loc == 12:
            self.divergence = 6.0e-6
            self.coriolis_param = 2.0 * omega * sin(34.5/180.0*pi)
        elif self.loc == 11:
            self.divergence = 3.5e-6
            self.coriolis_param = 2.0 * omega * sin(31.5/180.0*pi)
        elif self.loc == 6:
            self.divergence = 2.0e-6
            self.coriolis_param = 2.0 * omega * sin(16.5/180.0*pi)
        else:
            Pa.root_print('FORCING: Unrecognized ZGILS location ' + str(self.loc))
            Pa.kill()


        self.t_adv_max = -1.2/86400.0 # K/s BL tendency of temperature due to horizontal advection
        self.qt_adv_max = -0.6e-3/86400.0 # kg/kg/s BL tendency of qt due to horizontal advection
        self.tau_relax_inverse = 1.0/86400.0 # relaxation time scale = 24 h
        self.alpha_h = 1.2 # threshold ratio for determining BL height
        # Initialize the reference profiles classe
        self.forcing_ref = AdjustedMoistAdiabat(namelist, LH, Pa)

        self.CC = ClausiusClapeyron()
        self.CC.initialize(namelist, LH, Pa)

        return

    cpdef initialize(self, Grid.Grid Gr,ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.ug = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.vg = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.dtdt = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.dqtdt = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.subsidence = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.source_rh_nudge = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.source_qt_nudge = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.source_t_nudge = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.source_s_nudge = np.zeros(Gr.dims.npg,dtype=np.double,order='c')

        self.s_ls_adv = np.zeros(Gr.dims.npg,dtype=np.double,order='c')

        # compute the reference profiles for forcing/nudging
        cdef double Pg_parcel = 1000.0e2
        cdef double Tg_parcel = 295.0
        cdef double RH_ref = 0.3

        self.forcing_ref.initialize(Pa, Ref.p0_half[:], Gr.dims.nlg[2],
                                    Pg_parcel, Tg_parcel, RH_ref)

        cdef:
            Py_ssize_t k
            double sub_factor = self.divergence/(Ref.Pg*Ref.Pg)/g

        # initialize the profiles of geostrophic velocity, subsidence, and large scale advection
        with nogil:
            for k in xrange(Gr.dims.nlg[2]):
                self.ug[k] =  min(-10.0 + (-7.0-(-10.0))/(750.0e2-1000.0e2)*(Ref.p0_half[k]-1000.0e2),-4.0)

                self.subsidence[k]= sub_factor * (Ref.p0_half[k] - Ref.Pg) * Ref.p0_half[k] * Ref.p0_half[k] * Ref.alpha0_half[k]

                #Set large scale cooling
                if Ref.p0_half[k] > 900.0e2:
                    self.dtdt[k] = self.t_adv_max
                    self.dqtdt[k] = self.qt_adv_max
                elif  Ref.p0_half[k]< 800.0e2:
                    self.dtdt[k] = 0.0
                    self.dqtdt[k] = 0.0
                else:
                    self.dtdt[k] = self.t_adv_max * (Ref.p0_half[k]-800.0e2)/(900.0e2-800.0e2)
                    self.dqtdt[k] = self.qt_adv_max * (Ref.p0_half[k]-800.0e2)/(900.0e2-800.0e2)



        #Initialize Statistical Output
        NS.add_profile('s_subsidence_tendency', Gr, Pa)
        NS.add_profile('qt_subsidence_tendency', Gr, Pa)
        NS.add_profile('u_coriolis_tendency', Gr, Pa)
        NS.add_profile('v_coriolis_tendency',Gr, Pa)
        NS.add_profile('qt_rh_nudging', Gr, Pa)
        NS.add_profile('qt_ref_nudging', Gr, Pa)
        NS.add_profile('temperature_ref_nudging', Gr, Pa)
        NS.add_profile('s_ref_nudging', Gr, Pa)
        NS.add_profile('s_ls_adv', Gr, Pa)
        NS.add_ts('nudging_height', Gr, Pa)

        # Add profiles of fixed forcing profiles that do not depend on time
        NS.add_reference_profile('subsidence_velocity', Gr, Pa)
        NS.write_reference_profile('subsidence_velocity', self.subsidence[Gr.dims.gw:-Gr.dims.gw], Pa)

        NS.add_reference_profile('temperature_ls_adv', Gr, Pa)
        NS.write_reference_profile('temperature_ls_adv', self.dtdt[Gr.dims.gw:-Gr.dims.gw], Pa)

        NS.add_reference_profile('qt_ls_adv', Gr, Pa)
        NS.write_reference_profile('qt_ls_adv', self.dqtdt[Gr.dims.gw:-Gr.dims.gw], Pa)

        NS.add_reference_profile('u_geostrophic', Gr, Pa)
        NS.write_reference_profile('u_geostrophic', self.ug[Gr.dims.gw:-Gr.dims.gw], Pa)

        NS.add_reference_profile('v_geostrophic', Gr, Pa)
        NS.write_reference_profile('v_geostrophic', self.vg[Gr.dims.gw:-Gr.dims.gw], Pa)


        NS.add_reference_profile('ref_temperature', Gr, Pa)
        NS.write_reference_profile('ref_temperature', self.forcing_ref.temperature[Gr.dims.gw:-Gr.dims.gw], Pa)

        NS.add_reference_profile('ref_qt', Gr, Pa)
        NS.write_reference_profile('ref_qt', self.forcing_ref.qt[Gr.dims.gw:-Gr.dims.gw], Pa)


        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - gw
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i,j,k,ishift,jshift,ijk
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            double qt, qv, p0, t

            double [:] qtmean = Pa.HorizontalMean(Gr, &PV.values[qt_shift])
            double [:] tmean = Pa.HorizontalMean(Gr, &DV.values[t_shift])

        #Apply Coriolis Forcing

        coriolis_force(&Gr.dims,&PV.values[u_shift],&PV.values[v_shift],&PV.tendencies[u_shift],
                       &PV.tendencies[v_shift],&self.ug[0], &self.vg[0],self.coriolis_param, Ref.u0, Ref.v0  )

        # Apply Subsidence
        apply_subsidence(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence[0], &PV.values[s_shift], &PV.tendencies[s_shift])
        apply_subsidence(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence[0], &PV.values[qt_shift], &PV.tendencies[qt_shift])

        # Prepare for nudging by finding the boundary layer height
        self.h_BL = Gr.z_half[Gr.dims.n[2]]
        with nogil:
            for k in xrange(kmax, gw-1, -1):
                if qtmean[k] <= self.alpha_h * self.forcing_ref.qt[k]:
                    self.h_BL = Gr.zpl_half[k]

        # Now set the relaxation coefficient (depends on time-varying BL height diagnosed above)
        # Find the source term profiles for temperature, moisture nudging (2 components: free tropo and BL)
        cdef double [:] xi_relax = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        cdef double z_h, pv_star, qv_star
        with nogil:
            for k in xrange(Gr.dims.nlg[2]):
                self.source_rh_nudge[k] = 0.0
                z_h = Gr.zpl_half[k]/self.h_BL
                if z_h < 1.2:
                    xi_relax[k] = 0.0
                elif z_h > 1.5:
                    xi_relax[k] = self.tau_relax_inverse
                else:
                    xi_relax[k] = 0.5*self.tau_relax_inverse*(1.0 -   cos(pi*(z_h-1.2)/(1.5-1.2)))
                # here we also set the nudging to 20% rh in the BL
                if Gr.zpl_half[k] < 2000.0:
                    pv_star = self.CC.LT.fast_lookup(tmean[k])
                    qv_star = eps_v * pv_star/(Ref.p0_half[k] + (eps_v-1.0)*pv_star)
                    if qtmean[k] < 0.2 * qv_star:
                        self.source_rh_nudge[k] = (qv_star*0.2-qtmean[k])/3600.0
                # Here we find the nudging rates
                self.source_qt_nudge[k] = xi_relax[k] * (self.forcing_ref.qt[k]-qtmean[k])
                self.source_t_nudge[k]  = xi_relax[k] * (self.forcing_ref.temperature[k]-tmean[k])


        cdef double total_t_source, total_qt_source

        #Apply large scale source terms (BL advection, Free Tropo relaxation, BL humidity nudging)
        with nogil:
            for i in xrange(gw,imax):
                ishift = i * istride
                for j in xrange(gw,jmax):
                    jshift = j * jstride
                    for k in xrange(gw,kmax):
                        ijk = ishift + jshift + k
                        p0 = Ref.p0_half[k]
                        qt = PV.values[qt_shift + ijk]
                        qv = qt - DV.values[ql_shift + ijk]
                        t  = DV.values[t_shift + ijk]
                        total_t_source = self.dtdt[k] + self.source_t_nudge[k]
                        total_qt_source = self.dqtdt[k] + self.source_qt_nudge[k] + self.source_rh_nudge[k]

                        PV.tendencies[s_shift + ijk] += s_tendency_c(p0, qt, qv, t, total_qt_source, total_t_source)

                        PV.tendencies[qt_shift + ijk] += total_qt_source

                        self.source_s_nudge[ijk] = s_tendency_c(p0, qt, qv, t, self.source_qt_nudge[k] + self.source_rh_nudge[k], self.source_t_nudge[k] )

                        self.s_ls_adv[ijk]= s_tendency_c(p0,qt, qv, t, self.dqtdt[k], self.dtdt[k])


        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            double [:] tmp_tendency  = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
            double [:] tmp_tendency_2 = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
            double [:] mean_tendency = np.empty((Gr.dims.nlg[2],),dtype=np.double,order='c')
            double [:] mean_tendency_2 = np.zeros((Gr.dims.nlg[2]),dtype=np.double,order='c')
            double [:] umean = Pa.HorizontalMean(Gr, &PV.values[u_shift])
            # double [:] vmean = Pa.HorizontalMean(Gr, &PV.values[v_shift])

        #Output subsidence tendencies
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[s_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('s_subsidence_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        tmp_tendency[:] = 0.0
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[qt_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('qt_subsidence_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)


        #Output Coriolis tendencies
        tmp_tendency[:] = 0.0


        coriolis_force(&Gr.dims,&PV.values[u_shift],&PV.values[v_shift],&tmp_tendency[0],
                       &tmp_tendency_2[0],&self.ug[0], &self.vg[0],self.coriolis_param, Ref.u0, Ref.v0  )

        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        mean_tendency_2 = Pa.HorizontalMean(Gr,&tmp_tendency_2[0])
        NS.write_profile('u_coriolis_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('v_coriolis_tendency',mean_tendency_2[Gr.dims.gw:-Gr.dims.gw],Pa)


        NS.write_profile('qt_rh_nudging',self.source_rh_nudge[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('qt_ref_nudging',self.source_qt_nudge[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('temperature_ref_nudging',self.source_t_nudge[Gr.dims.gw:-Gr.dims.gw],Pa)

        mean_tendency = Pa.HorizontalMean(Gr,&self.source_s_nudge[0])
        NS.write_profile('s_ref_nudging',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        mean_tendency = Pa.HorizontalMean(Gr,&self.s_ls_adv[0])
        NS.write_profile('s_ls_adv',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        NS.write_ts('nudging_height',self.h_BL, Pa)
        return



cdef extern from "thermodynamics_sa.h":
    void eos_c(Lookup.LookupStruct *LT, double(*lam_fp)(double), double(*L_fp)(double, double), double p0, double s, double qt, double *T, double *qv, double *ql, double *qi) nogil
cdef extern from "thermodynamic_functions.h":
    inline double pd_c(double p0, double qt, double qv) nogil
    inline double pv_c(double p0, double qt, double qv) nogil
cdef extern from "entropies.h":
    inline double sd_c(double pd, double T) nogil
    inline double sv_c(double pv, double T) nogil
    inline double sc_c(double L, double T) nogil


# This class computes the reference profiles needed for ZGILS cases
# Reference temperature profile correspondends to a moist adiabat
# Reference moisture profile corresponds to a fixed relative humidity given the reference temperature profile
cdef class AdjustedMoistAdiabat:
    def __init__(self,namelist,  LatentHeat LH, ParallelMPI.ParallelMPI Pa ):


        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp
        self.CC = ClausiusClapeyron()
        self.CC.initialize(namelist, LH, Pa)

        return
    cpdef get_pv_star(self, t):
        return self.CC.LT.fast_lookup(t)

    cpdef entropy(self, double p0, double T, double qt, double ql, double qi):
        cdef:
            double qv = qt - ql - qi
            double qd = 1.0 - qt
            double pd = pd_c(p0, qt, qv)
            double pv = pv_c(p0, qt, qv)
            double Lambda = self.Lambda_fp(T)
            double L = self.L_fp(T, Lambda)

        return sd_c(pd, T) * (1.0 - qt) + sv_c(pv, T) * qt + sc_c(L, T) * (ql + qi)

    cpdef eos(self, double p0, double s, double qt):
        cdef:
            double T, qv, qc, ql, qi, lam
        eos_c(&self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, p0, s, qt, &T, &qv, &ql, &qi)
        return T, ql, qi



    cpdef initialize(self, ParallelMPI.ParallelMPI Pa,
                   double [:] pressure_array, Py_ssize_t n_levels, double Pg, double Tg, double RH):
        '''
        Initialize the forcing reference profiles. These profiles use the temperature corresponding to a moist adiabat,
        but modify the water vapor content to have a given relative humidity. Thus entropy and qt are not conserved.
        '''
        self.s = np.zeros(n_levels, dtype=np.double, order='c')
        self.qt = np.zeros(n_levels, dtype=np.double, order='c')
        self.temperature = np.zeros(n_levels, dtype=np.double, order='c')
        self.rv = np.zeros(n_levels, dtype=np.double, order='c')
        cdef double pvg = self.get_pv_star(Tg)
        cdef double qtg = eps_v * pvg / (Pg + (eps_v-1.0)*pvg)
        cdef double sg = self.entropy(Pg, Tg, qtg, 0.0, 0.0)


        cdef double temperature, ql, qi, pv

        # Compute reference state thermodynamic profiles
        for k in xrange(n_levels):
            temperature, ql, qi = self.eos(pressure_array[k], sg, qtg)
            pv = self.get_pv_star(temperature) * RH
            self.qt[k] = eps_v * pv / (pressure_array[k] + (eps_v-1.0)*pv)
            self.s[k] = self.entropy(pressure_array[k],temperature, self.qt[k] , 0.0, 0.0)
            self.temperature[k] = temperature
            self.rv[k] = self.qt[k]/(1.0-self.qt[k])
        return




cdef coriolis_force(Grid.DimStruct *dims, double *u, double *v, double *ut, double *vt, double *ug, double *vg, double coriolis_param, double u0, double v0 ):

    cdef:
        Py_ssize_t imin = dims.gw
        Py_ssize_t jmin = dims.gw
        Py_ssize_t kmin = dims.gw
        Py_ssize_t imax = dims.nlg[0] -dims.gw
        Py_ssize_t jmax = dims.nlg[1] -dims.gw
        Py_ssize_t kmax = dims.nlg[2] -dims.gw
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2]
        Py_ssize_t jstride = dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k
        double u_at_v, v_at_u

    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    u_at_v = 0.25*(u[ijk] + u[ijk-istride] + u[ijk-istride+jstride] + u[ijk +jstride]) + u0
                    v_at_u = 0.25*(v[ijk] + v[ijk+istride] + v[ijk+istride-jstride] + v[ijk-jstride]) + v0
                    ut[ijk] = ut[ijk] - coriolis_param * (vg[k] - v_at_u)
                    vt[ijk] = vt[ijk] + coriolis_param * (ug[k] - u_at_v)
    return


cdef large_scale_p_gradient(Grid.DimStruct *dims, double *umean, double *vmean, double *ut, double *vt, double *ug, double *vg,
                          double coriolis_param, double u0, double v0 ):

    cdef:
        Py_ssize_t imin = dims.gw
        Py_ssize_t jmin = dims.gw
        Py_ssize_t kmin = dims.gw
        Py_ssize_t imax = dims.nlg[0] -dims.gw
        Py_ssize_t jmax = dims.nlg[1] -dims.gw
        Py_ssize_t kmax = dims.nlg[2] -dims.gw
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2]
        Py_ssize_t jstride = dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k
        double u_at_v, v_at_u


    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    u_at_v = umean[k] + u0
                    v_at_u = vmean[k] + v0
                    ut[ijk] = ut[ijk] - coriolis_param * (vg[k] - v_at_u)
                    vt[ijk] = vt[ijk] + coriolis_param * (ug[k] - u_at_v)


    return

cdef apply_subsidence(Grid.DimStruct *dims, double *rho0, double *rho0_half, double *subsidence, double* values,  double *tendencies):

    cdef:
        Py_ssize_t imin = dims.gw
        Py_ssize_t jmin = dims.gw
        Py_ssize_t kmin = dims.gw
        Py_ssize_t imax = dims.nlg[0] -dims.gw
        Py_ssize_t jmax = dims.nlg[1] -dims.gw
        Py_ssize_t kmax = dims.nlg[2] -dims.gw
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2]
        Py_ssize_t jstride = dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k
        double dxi = dims.dxi[2]
    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    tendencies[ijk] -= (values[ijk+1] - values[ijk]) * dxi * subsidence[k] * dims.imetl[k]

    return

