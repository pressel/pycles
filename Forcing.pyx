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
cimport Thermodynamics
from thermodynamic_functions cimport cpm_c, pv_c, pd_c, exner_c
from entropies cimport sv_c, sd_c
import numpy as np
import cython
from libc.math cimport fabs, sin, cos, fmin, fmax
from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI
include 'parameters.pxi'
from Initialization import sat_adjst, qv_unsat
# import matplotlib.pyplot as plt

cdef class Forcing:
    def __init__(self, namelist, ParallelMPI.ParallelMPI Pa):
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
        elif casename == 'Isdac':
            self.scheme = ForcingIsdac()
        elif casename == 'IsdacCC':
            self.scheme = ForcingIsdacCC()
        elif casename == 'StableBubble':
            self.scheme = ForcingNone()
        elif casename == 'Mpace':
            self.scheme = ForcingMpace()
        elif casename == 'Sheba':
            self.scheme = ForcingSheba()
        else:
            Pa.root_print('No focing for casename: ' +  casename)
            Pa.root_print('Killing simulation now!!!')
            Pa.kill()
        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, namelist):
        self.scheme.initialize(Gr, RS, Th, NS, Pa, namelist)
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
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
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

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
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
                self.ug[k] = -10.0 + (1.8e-3)*Gr.zl_half[k]

                #Set large scale cooling
                if Gr.zl_half[k] <= 1500.0:
                    self.dtdt[k] = -2.0/(3600 * 24.0)      #K/s
                if Gr.zl_half[k] > 1500.0:
                    self.dtdt[k] = -2.0/(3600 * 24.0) + (Gr.zl_half[k] - 1500.0) * (0.0 - -2.0/(3600 * 24.0)) / (3000.0 - 1500.0)

                #Set large scale drying
                if Gr.zl_half[k] <= 300.0:
                    self.dqtdt[k] = -1.2e-8   #kg/(kg * s)
                if Gr.zl_half[k] > 300.0 and Gr.zl_half[k] <= 500.0:
                    self.dqtdt[k] = -1.2e-8 + (Gr.zl_half[k] - 300.0)*(0.0 - -1.2e-8)/(500.0 - 300.0) #kg/(kg * s)

                #Set large scale subsidence
                if Gr.zl_half[k] <= 1500.0:
                    self.subsidence[k] = 0.0 + Gr.zl_half[k]*(-0.65/100.0 - 0.0)/(1500.0 - 0.0)
                if Gr.zl_half[k] > 1500.0 and Gr.zl_half[k] <= 2100.0:
                    self.subsidence[k] = -0.65/100 + (Gr.zl_half[k] - 1500.0)* (0.0 - -0.65/100.0)/(2100.0 - 1500.0)


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
            double pd
            double pv
            double qt
            double qv
            double p0
            double rho0
            double t
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
                        rho0 = Ref.rho0_half[k]
                        qt = PV.values[qt_shift + ijk]
                        qv = qt - DV.values[ql_shift + ijk]
                        pd = pd_c(p0,qt,qv)
                        pv = pv_c(p0,qt,qv)
                        t  = DV.values[t_shift + ijk]
                        PV.tendencies[s_shift + ijk] += (cpm_c(qt)
                                                         * self.dtdt[k] * exner_c(p0) * rho0)/t
                        PV.tendencies[s_shift + ijk] += (sv_c(pv,t) - sd_c(pd,t))*self.dqtdt[k]
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
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
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

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
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

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t k

        self.subsidence = np.empty((Gr.dims.nlg[2]),dtype=np.double, order='c')
        self.ug = np.empty((Gr.dims.nlg[2]),dtype=np.double, order='c')
        self.vg = np.empty((Gr.dims.nlg[2]),dtype=np.double, order='c')

        if self.rf02_flag:
            with nogil:
                for k in range(Gr.dims.nlg[2]):
                    self.subsidence[k] = -Gr.zl_half[k] * self.divergence
                    self.ug[k] = 3.0 + 4.3*Gr.zl_half[k]/1000.0
                    self.vg[k] = -9.0 + 5.6 * Gr.zl_half[k]/1000.0
        else:
            with nogil:
                for k in range(Gr.dims.nlg[2]):
                    self.subsidence[k] = -Gr.zl_half[k] * self.divergence
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
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
            Py_ssize_t thetali_shift = DV.get_varshift(Gr,'thetali')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')

            double [:] thetal_subs = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
            double [:] qt_subs = np.zeros(Gr.dims.npg, dtype=np.double, order='c')

            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i,j,k,ishift,jshift,ijk

            double pd
            double pv
            double qt
            double qv
            double p0
            double rho0
            double t

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

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        cdef Py_ssize_t k

        self.subsidence = np.empty((Gr.dims.nlg[2]),dtype=np.double, order='c')
        self.ug = np.empty(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.vg = np.empty(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.dtdt = np.ones(Gr.dims.nlg[2],dtype=np.double,order='c') * -2.5/86400.0
        self.dqtdt = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')

        with nogil:
            for k in range(Gr.dims.nlg[2]):
                if Gr.zl_half[k] <= 2260.0:
                    self.subsidence[k] = -(0.005/2260.0) * Gr.zl_half[k]
                else:
                    self.subsidence[k] = -0.005
                if Gr.zl_half[k]<=2980.0:
                    self.dqtdt[k] = (-1.0 + 1.3456/2980.0 * Gr.zl_half[k])/86400.0/1000.0
                else:
                    self.dqtdt[k] = 0.3456/86400.0/1000.0
                self.ug[k] = -9.9 + 2.0e-3 * Gr.zl_half[k]
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
            double pd
            double pv
            double qt
            double qv
            double p0
            double rho0
            double t

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
                        rho0 = Ref.rho0_half[k]
                        qt = PV.values[qt_shift + ijk]
                        qv = qt - DV.values[ql_shift + ijk]
                        pd = pd_c(p0,qt,qv)
                        pv = pv_c(p0,qt,qv)
                        t  = DV.values[t_shift + ijk]
                        PV.tendencies[s_shift + ijk] += (cpm_c(qt)
                                                         * self.dtdt[k] * exner_c(p0) * rho0)/t
                        PV.tendencies[s_shift + ijk] += (sv_c(pv,t) - sd_c(pd,t))*self.dqtdt[k]
                        PV.tendencies[qt_shift + ijk] += self.dqtdt[k]



        coriolis_force(&Gr.dims,&PV.values[u_shift],&PV.values[v_shift],&PV.tendencies[u_shift],
                       &PV.tendencies[v_shift],&self.ug[0], &self.vg[0],self.coriolis_param, Ref.u0, Ref.v0  )

        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        return



cdef class ForcingIsdac:
    def __init__(self):
        self.divergence = 5e-6

        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, Namelist):

        try:
            self.merra2 = Namelist['forcing']['merra2']
            Pa.root_print('Forcing from MERRA-2 is applied to ISDAC!')
        except:
            self.merra2 = False


        self.initial_entropy = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.initial_qt = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.nudge_coeff_velocities = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.nudge_coeff_scalars = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.initial_u = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.initial_v = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.w_half =  np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        # self.ls_adv_Q = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.ls_adv_qt = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.ls_adv_t = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')

        cdef:
            Py_ssize_t k
            double [:] thetal = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
            #double [:] qt = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
            double T, ql
            double Q_rad_tropo = -1.0/86400.

        #self.w = np.empty((Gr.dims.nlg[2]),dtype=np.double, order='c')
        with nogil:
            for k in range(Gr.dims.nlg[2]):
                if Gr.zl_half[k] < 825.0:
                    self.w_half[k] = -Gr.zl_half[k] * self.divergence
                else:
                    self.w_half[k] = -0.4125e-2

        #Get profiles for nudging
        for k in xrange(Gr.dims.nlg[2]):

            #Set thetal and qt profile
            if Gr.zl_half[k] < 400.0:
                thetal[k] = 265.0 + 0.004 * (Gr.zl_half[k] - 400.0)
                self.initial_qt[k] = 1.5 - 0.00075 * (Gr.zl_half[k] - 400.0)
            if 400.0 <= Gr.zl_half[k] < 825.0:
                thetal[k] = 265.0
                self.initial_qt[k] = 1.5
            if 825.0 <= Gr.zl_half[k] < 2045.0:
                thetal[k] = 266.0 + (Gr.zl_half[k] - 825.0) ** 0.3
                self.initial_qt[k] = 1.2
            if Gr.zl_half[k] >= 2045.0:
                thetal[k] = 271.0 + (Gr.zl_half[k] - 2000.0) ** 0.33
                self.initial_qt[k] = 0.5 - 0.000075 * (Gr.zl_half[k] - 2045.0)

            #Change units to kg/kg
            self.initial_qt[k]/= 1000.0

            #Set velocity profile
            self.initial_v[k] = -2.0 + 0.003 * Gr.zl_half[k]
            self.initial_u[k] = -7.0

            #Now get entropy profile
            T, ql = sat_adjst(RS.p0_half[k], thetal[k], self.initial_qt[k], Th)
            self.initial_entropy[k] = Th.entropy(RS.p0_half[k], T, self.initial_qt[k], ql, 0.0)


            #Nudging coefficients
            if Gr.zl_half[k] <= 1200.0:
                self.nudge_coeff_scalars[k] = 0.0
            if 1200.0 < Gr.zl_half[k] <= 1500.0:
                self.nudge_coeff_scalars[k] = (1/3600.0)*0.5*(1.0 - cos(pi * (Gr.zl_half[k] - 1200.0)/300.0))
            if Gr.zl_half[k] > 1500.0:
                self.nudge_coeff_scalars[k] = 1/3600.0

            if Gr.zl_half[k] <= 825.0:
                self.nudge_coeff_velocities[k] = (1/7200.0)*0.5*(1 - cos(pi*Gr.zl_half[k]/825.0))
            if Gr.zl_half[k] > 825.0:
                self.nudge_coeff_velocities[k] = 1/7200.0

            # #large-scale advection
            # if Gr.zl_half[k] >= 825.0 and Gr.zl_half[k] < 2045.0:
            #     self.ls_adv_Q[k] = -self.w_half[k] * 0.3 * (Gr.zl_half[k] - 825.0) ** (-0.7) + Q_rad_tropo
            # if Gr.zl_half[k] >= 2045.0:
            #     self.ls_adv_Q[k] = -self.w_half[k] * 0.33 * (Gr.zl_half[k] - 2000.0) ** (-0.67) + Q_rad_tropo
            #     self.ls_adv_qt[k] = self.w_half[k] * 7.5e-8




        if self.merra2:

            #Specify MERRA-2 forcing profiles
            m_lev = np.array([1000., 975., 950., 925., 900., 875., 850., 825., 800., 775.])*100.0 #pressure in Pa
            m_dtdt = np.array([-4.90852108e-06, 3.65721316e-05, 1.45196245e-05, 5.68313408e-06, 4.66431666e-05,
                  3.75870804e-05, 4.69653423e-05, 9.46793007e-05, 1.11585236e-04, 4.47569037e-05])
            m_dqvdt = np.array([2.90906654e-09, 2.95498737e-09, -1.16688170e-09, -5.69702507e-10, 3.71956865e-09,
                   -1.34783029e-09, -3.74744680e-09, -8.82443452e-09, -6.83885393e-09, 3.92175181e-10])

            #Now interpolate onto z coordinate
            ls_adv_t = interp_pchip(RS.p0_half[::-1], m_lev[::-1], m_dtdt[::-1])
            ls_adv_qt = interp_pchip(RS.p0_half[::-1], m_lev[::-1], m_dqvdt[::-1])

            self.ls_adv_qt = ls_adv_qt[::-1]
            self.ls_adv_t = ls_adv_t[::-1]



       #Initialize Statistical Output
        NS.add_profile('s_subsidence_tendency', Gr, Pa)
        NS.add_profile('qt_subsidence_tendency', Gr, Pa)
        NS.add_profile('u_subsidence_tendency', Gr, Pa)
        NS.add_profile('v_subsidence_tendency', Gr, Pa)
        NS.add_profile('u_nudging_tendency', Gr, Pa)
        NS.add_profile('v_nudging_tendency',Gr, Pa)
        NS.add_profile('s_nudging_tendency',Gr, Pa)
        NS.add_profile('qt_nudging_tendency',Gr, Pa)
        NS.add_profile('s_ls_adv_tendency', Gr, Pa)
        NS.add_profile('qt_ls_adv_tendency', Gr, Pa)

        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr,'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - gw
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i,j,k,ishift,jshift,ijk
            double pd, pv, qt, qv, p0, rho0, t

        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[s_shift],&PV.tendencies[s_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[qt_shift],&PV.tendencies[qt_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[u_shift],&PV.tendencies[u_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[v_shift],&PV.tendencies[v_shift])

        apply_nudging(&Gr.dims,&self.nudge_coeff_scalars[0],&self.initial_entropy[0],&PV.values[s_shift],&PV.tendencies[s_shift])
        apply_nudging(&Gr.dims,&self.nudge_coeff_scalars[0],&self.initial_qt[0],&PV.values[qt_shift],&PV.tendencies[qt_shift])
        apply_nudging(&Gr.dims,&self.nudge_coeff_velocities[0],&self.initial_u[0],&PV.values[u_shift],&PV.tendencies[u_shift])
        apply_nudging(&Gr.dims,&self.nudge_coeff_velocities[0],&self.initial_v[0],&PV.values[v_shift],&PV.tendencies[v_shift])

        # apply_ls_advection_entropy(&Gr.dims, &PV.tendencies[s_shift], &DV.values[t_shift], &self.ls_adv_Q[0])
        # apply_ls_advection_qt(&Gr.dims, &PV.tendencies[qt_shift], &self.ls_adv_qt[0])

        with nogil:
            for i in xrange(gw,imax):
                ishift = i * istride
                for j in xrange(gw,jmax):
                    jshift = j * jstride
                    for k in xrange(gw,kmax):
                        ijk = ishift + jshift + k
                        p0 = Ref.p0_half[k]
                        rho0 = Ref.rho0_half[k]
                        qt = PV.values[qt_shift + ijk]
                        qv = qt - DV.values[ql_shift + ijk]
                        pd = pd_c(p0,qt,qv)
                        pv = pv_c(p0,qt,qv)
                        t  = DV.values[t_shift + ijk]

                        PV.tendencies[s_shift + ijk] += (cpm_c(qt) * (self.ls_adv_t[k]))/t
                        PV.tendencies[s_shift + ijk] += (sv_c(pv,t) - sd_c(pd,t)) * self.ls_adv_qt[k]
                        PV.tendencies[qt_shift + ijk] += (self.ls_adv_qt[k])


        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            double [:] tmp_tendency  = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
            double [:] mean_tendency = np.empty((Gr.dims.npg,),dtype=np.double,order='c')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - gw
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i,j,k,ishift,jshift,ijk
            double pd, pv, qt, qv, p0, rho0, t


        #Output subsidence tendencies
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[s_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('s_subsidence_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        tmp_tendency[:] = 0.0
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[qt_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('qt_subsidence_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        tmp_tendency[:] = 0.0
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[u_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('u_subsidence_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        tmp_tendency[:] = 0.0
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[v_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('v_subsidence_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        tmp_tendency[:] = 0.0
        apply_nudging(&Gr.dims,&self.nudge_coeff_scalars[0],&self.initial_entropy[0],&PV.values[s_shift],
                      &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('s_nudging_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        tmp_tendency[:] = 0.0
        apply_nudging(&Gr.dims,&self.nudge_coeff_scalars[0],&self.initial_qt[0],&PV.values[qt_shift],
                      &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('qt_nudging_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        tmp_tendency[:] = 0.0
        apply_nudging(&Gr.dims,&self.nudge_coeff_velocities[0],&self.initial_u[0],&PV.values[u_shift],
                      &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('u_nudging_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        tmp_tendency[:] = 0.0
        apply_nudging(&Gr.dims,&self.nudge_coeff_velocities[0],&self.initial_v[0],&PV.values[v_shift],
                      &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('v_nudging_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        # tmp_tendency[:] = 0.0
        # apply_ls_advection_entropy(&Gr.dims, &tmp_tendency[0], &DV.values[t_shift], &self.ls_adv_Q[0])
        # mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        # NS.write_profile('s_ls_adv_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)
        #
        # tmp_tendency[:] = 0.0
        # apply_ls_advection_qt(&Gr.dims, &tmp_tendency[0], &self.ls_adv_qt[0])
        # mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        # NS.write_profile('qt_ls_adv_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        with nogil:
            for i in xrange(Gr.dims.npg):
                tmp_tendency[i] = 0.0

            for i in xrange(gw,imax):
                ishift = i * istride
                for j in xrange(gw,jmax):
                    jshift = j * jstride
                    for k in xrange(gw,kmax):
                        ijk = ishift + jshift + k
                        p0 = Ref.p0_half[k]
                        rho0 = Ref.rho0_half[k]
                        qt = PV.values[qt_shift + ijk]
                        qv = qt - DV.values[ql_shift + ijk]
                        pd = pd_c(p0,qt,qv)
                        pv = pv_c(p0,qt,qv)
                        t  = DV.values[t_shift + ijk]

                        tmp_tendency[ijk] += (cpm_c(qt) * (self.ls_adv_t[k]))/t
                        tmp_tendency[ijk] += (sv_c(pv,t) - sd_c(pd,t)) * self.ls_adv_qt[k]

        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('s_ls_adv_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('qt_ls_adv_tendency',self.ls_adv_qt[Gr.dims.gw:-Gr.dims.gw],Pa)

        return


cdef class ForcingIsdacCC:
    def __init__(self):

        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, namelist):

        self.initial_entropy = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.initial_qt = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.nudge_coeff_scalars = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.nudge_coeff_velocities = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.initial_u = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.initial_v = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.w_half =  np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')

        try:
            self.divergence = namelist['initial']['divergence']
        except:
            self.divergence = 5.0e-6

        cdef:
            Py_ssize_t k
            double z_top = namelist['initial']['z_top'] #cloud top height
            double dz_inv = namelist['initial']['dzi'] #inversion depth

        #self.w = np.empty((Gr.dims.nlg[2]),dtype=np.double, order='c')
        for k in range(Gr.dims.nlg[2]):
            if Gr.zl_half[k] <= z_top:
                self.w_half[k] = -self.divergence * z_top * np.cos(pi*0.5*(z_top - Gr.zl_half[k])/z_top)
            else:
                self.w_half[k] = -self.divergence * z_top * np.cos(pi*0.5*(Gr.zl_half[k] - z_top)/(10000.0 - z_top))

        #Get velocity profiles for nudging
        for k in xrange(Gr.dims.nlg[2]):

            #Set velocity profile
            self.initial_v[k] = -2.0 + 0.003 * Gr.zl_half[k]
            self.initial_u[k] = -7.0

            if Gr.zl_half[k] <= z_top:
                self.nudge_coeff_velocities[k] = (1/7200.0)*0.5*(1 - cos(pi*Gr.zl_half[k]/z_top))
            if Gr.zl_half[k] > z_top:
                self.nudge_coeff_velocities[k] = 1/7200.0

            #Now get entropy profile
            T, ql = sat_adjst(RS.p0_half[k], RS.ic_thetal[k], RS.ic_qt[k], Th)
            self.initial_entropy[k] = Th.entropy(RS.p0_half[k], T, RS.ic_qt[k], ql, 0.0)
            self.initial_qt[k] = RS.ic_qt[k]

            #Nudging coefficients
            if Gr.zl_half[k] <= 1200.0:
                self.nudge_coeff_scalars[k] = 0.0
            else:
                self.nudge_coeff_scalars[k] = 1/3600.0


       #Initialize Statistical Output
        NS.add_profile('s_subsidence_tendency', Gr, Pa)
        NS.add_profile('qt_subsidence_tendency', Gr, Pa)
        NS.add_profile('u_subsidence_tendency', Gr, Pa)
        NS.add_profile('v_subsidence_tendency', Gr, Pa)
        NS.add_profile('u_nudging_tendency', Gr, Pa)
        NS.add_profile('v_nudging_tendency',Gr, Pa)
        NS.add_profile('s_nudging_tendency', Gr, Pa)
        NS.add_profile('qt_nudging_tendency', Gr, Pa)

        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr,'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')

        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[s_shift],&PV.tendencies[s_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[qt_shift],&PV.tendencies[qt_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[u_shift],&PV.tendencies[u_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[v_shift],&PV.tendencies[v_shift])

        apply_nudging(&Gr.dims,&self.nudge_coeff_velocities[0],&self.initial_u[0],&PV.values[u_shift],&PV.tendencies[u_shift])
        apply_nudging(&Gr.dims,&self.nudge_coeff_velocities[0],&self.initial_v[0],&PV.values[v_shift],&PV.tendencies[v_shift])
        apply_nudging(&Gr.dims,&self.nudge_coeff_scalars[0],&self.initial_entropy[0],&PV.values[s_shift],&PV.tendencies[s_shift])
        apply_nudging(&Gr.dims,&self.nudge_coeff_scalars[0],&self.initial_qt[0],&PV.values[qt_shift],&PV.tendencies[qt_shift])


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
            double [:] mean_tendency = np.empty((Gr.dims.npg,),dtype=np.double,order='c')


        #Output subsidence tendencies
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[s_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('s_subsidence_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        tmp_tendency[:] = 0.0
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[qt_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('qt_subsidence_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        tmp_tendency[:] = 0.0
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[u_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('u_subsidence_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        tmp_tendency[:] = 0.0
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[v_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('v_subsidence_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        tmp_tendency[:] = 0.0
        apply_nudging(&Gr.dims,&self.nudge_coeff_velocities[0],&self.initial_u[0],&PV.values[u_shift],
                      &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('u_nudging_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        tmp_tendency[:] = 0.0
        apply_nudging(&Gr.dims,&self.nudge_coeff_velocities[0],&self.initial_v[0],&PV.values[v_shift],
                      &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('v_nudging_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        tmp_tendency[:] = 0.0
        apply_nudging(&Gr.dims,&self.nudge_coeff_scalars[0],&self.initial_entropy[0],&PV.values[s_shift],
                      &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('s_nudging_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        tmp_tendency[:] = 0.0
        apply_nudging(&Gr.dims,&self.nudge_coeff_scalars[0],&self.initial_qt[0],&PV.values[qt_shift],
                      &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('qt_nudging_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        return


cdef class ForcingMpace:
    def __init__(self):
        self.divergence = 5.8e-6
        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, namelist):

        cdef:
            Py_ssize_t k
            double omega_min = self.divergence*(RS.Pg - 85000.0)

        self.subsidence = np.empty((Gr.dims.nlg[2]), dtype=np.double, order='c')
        self.dtdt = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.dqtdt = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.u0 = np.ones(Gr.dims.nlg[2],dtype=np.double,order='c') * -13.0
        self.v0 = np.ones(Gr.dims.nlg[2],dtype=np.double,order='c') * -3.0
        self.nudge_coeff_velocities = np.ones(Gr.dims.nlg[2],dtype=np.double,order='c') / 3600.0

        with nogil:
            for k in range(Gr.dims.nlg[2]):
                self.dtdt[k] = fmin(-4.0, -15.0*(1.0 - (RS.Pg - RS.p0_half[k])/21818.0))/86400.0
                self.dqtdt[k] = fmin(0.164, -3.0*(1.0 - (RS.Pg - RS.p0_half[k])/15171.0))/86400000.0
                self.subsidence[k] = -fmin(self.divergence*(RS.Pg - RS.p0_half[k]), omega_min) / RS.rho0_half[k] / g

        NS.add_profile('s_subsidence_tendency', Gr, Pa)
        NS.add_profile('qt_subsidence_tendency', Gr, Pa)

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
            double pd
            double pv
            double qt
            double qv
            double p0
            double rho0
            double t

        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[s_shift],&PV.tendencies[s_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[qt_shift],&PV.tendencies[qt_shift])

        #Apply large scale source terms
        with nogil:
            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        p0 = Ref.p0_half[k]
                        rho0 = Ref.rho0_half[k]
                        qt = PV.values[qt_shift + ijk]
                        qv = qt - DV.values[ql_shift + ijk]
                        pd = pd_c(p0,qt,qv)
                        pv = pv_c(p0,qt,qv)
                        t  = DV.values[t_shift + ijk]
                        PV.tendencies[s_shift + ijk] += (cpm_c(qt)
                                                         * self.dtdt[k] * exner_c(p0) * rho0)/t
                        PV.tendencies[s_shift + ijk] += (sv_c(pv,t) - sd_c(pd,t))*self.dqtdt[k]
                        PV.tendencies[qt_shift + ijk] += self.dqtdt[k]

        apply_nudging(&Gr.dims, &self.nudge_coeff_velocities[0], &self.u0[0], &PV.values[u_shift], &PV.tendencies[u_shift])
        apply_nudging(&Gr.dims, &self.nudge_coeff_velocities[0], &self.v0[0], &PV.values[v_shift], &PV.tendencies[v_shift])

        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            double [:] tmp_tendency  = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
            double [:] mean_tendency = np.empty((Gr.dims.npg,),dtype=np.double,order='c')

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

        #Output advective forcing tendencies
        tmp_tendency[:] = 0.0



        return

cdef class ForcingSheba:
    def __init__(self):

        return

    @cython.wraparound(True)
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, namelist):

        try:
            input_file = str(namelist['forcing']['input_file'])
            Pa.root_print('Using SHEBA input file for forcing profiles.')
        except:
            Pa.root_print('No SHEBA input file! Kill simulation now!')
            Pa.kill()

        cdef:
            Py_ssize_t k, idx
            double omega
            double p_inv = 95700.0
            double [:] EC_dTdt
            double [:] EC_dqdt
            double [:] EC_pressure
            double [:] EC_u
            double [:] EC_v
            double EC_ps
            double [:] dtdt
            double [:] dqtdt
            double [:] u0
            double [:] v0

        self.subsidence = np.zeros((Gr.dims.nlg[2]), dtype=np.double, order='c')
        self.nudge_coeff_velocities = 1.0 / 3600.0
        self.dtdt = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.dqtdt = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.u0 = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.v0 = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')

        data = nc.Dataset(input_file, 'r')
        time_all = data.variables['yymmddhh'][:]
        idx = np.where(time_all==1998050712)[0][0]
        EC_dTdt = np.mean(data.variables['dTdt-hor-adv'][idx:idx+12,:],axis=0)
        EC_dqdt = np.mean(data.variables['dqdt-hor-adv'][idx:idx+12,:],axis=0)
        EC_u = np.mean(data.variables['u'][idx:idx+12,:],axis=0)
        EC_v = np.mean(data.variables['v'][idx:idx+12,:],axis=0)
        EC_ps = np.mean(data.variables['psurf'][idx:idx+12])
        EC_pressure = np.array(data.variables['sigma'][:])*EC_ps
        data.close()

        Pa.root_print('Finish reading in SHEBA forcing fields.')

        # dtdt = interp_pchip(np.array(RS.p0_half)[::-1], np.array(EC_pressure)[:], np.array(EC_dTdt)[:])
        # dqtdt = interp_pchip(np.array(RS.p0_half)[::-1], np.array(EC_pressure)[:], np.array(EC_dqdt)[:])
        # u0 = interp_pchip(np.array(RS.p0)[::-1], np.array(EC_pressure)[:], np.array(EC_u)[:])
        # v0 = interp_pchip(np.array(RS.p0)[::-1], np.array(EC_pressure)[:], np.array(EC_v)[:])
        #
        # self.dtdt = np.array(dtdt[::-1],dtype=np.double, order='c')
        # self.dqtdt = np.array(dqtdt[::-1],dtype=np.double, order='c')
        # self.u0 = np.array(u0[::-1],dtype=np.double, order='c')
        # self.v0 = np.array(v0[::-1],dtype=np.double, order='c')

        for k in xrange(Gr.dims.nlg[2]):
            self.dtdt[k] = interp_pchip(RS.p0_half[k], EC_pressure, EC_dTdt)
            self.dqtdt[k] = interp_pchip(RS.p0_half[k], EC_pressure, EC_dqdt)
            self.u0[k] = interp_pchip(RS.p0[k], EC_pressure, EC_u)
            self.v0[k] = interp_pchip(RS.p0[k], EC_pressure, EC_v)

        with nogil:
            for k in xrange(Gr.dims.nlg[2]):
                if RS.p0_half[k] < p_inv:
                    self.dtdt[k] = fmin(1.815e-9*(p_inv - RS.p0_half[k]), 2.85e-5) - 0.05 * RS.alpha0_half[k] / cpd
                    self.dqtdt[k] = 7.0e-10

                omega = fmin((RS.Pg - RS.p0[k])*0.05/6000.0, 0.05)
                self.subsidence[k] = -omega / RS.rho0[k] / g

        NS.add_profile('s_subsidence_tendency', Gr, Pa)
        NS.add_profile('qt_subsidence_tendency', Gr, Pa)

        return

    @cython.wraparound(False)
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
            double pd
            double pv
            double qt
            double qv
            double p0
            double rho0
            double t

        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[s_shift],&PV.tendencies[s_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[qt_shift],&PV.tendencies[qt_shift])

        #Apply large scale source terms
        with nogil:
            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        p0 = Ref.p0_half[k]
                        rho0 = Ref.rho0_half[k]
                        qt = PV.values[qt_shift + ijk]
                        qv = qt - DV.values[ql_shift + ijk]
                        pd = pd_c(p0,qt,qv)
                        pv = pv_c(p0,qt,qv)
                        t  = DV.values[t_shift + ijk]
                        PV.tendencies[s_shift + ijk] += (cpm_c(qt)
                                                         * self.dtdt[k] * exner_c(p0) * rho0)/t
                        PV.tendencies[s_shift + ijk] += (sv_c(pv,t) - sd_c(pd,t))*self.dqtdt[k]
                        PV.tendencies[qt_shift + ijk] += self.dqtdt[k]

        #Horizontal wind nudging
        cdef:
            double [:] umean = Pa.HorizontalMean(Gr, &PV.values[u_shift])
            double [:] vmean = Pa.HorizontalMean(Gr, &PV.values[v_shift])
            double [:] nudge_source_u = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
            double [:] nudge_source_v = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')

        # Calculate nudging
        with nogil:
            for k in xrange(Gr.dims.nlg[2]):
                # Nudge mean wind profiles through entire depth
                nudge_source_u[k] = -(umean[k] + Ref.u0 - self.u0[k]) * self.nudge_coeff_velocities
                nudge_source_v[k] = -(vmean[k] + Ref.v0 - self.v0[k]) * self.nudge_coeff_velocities

        with nogil:
            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        PV.tendencies[u_shift + ijk] += nudge_source_u[k]
                        PV.tendencies[v_shift + ijk] += nudge_source_v[k]


        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            double [:] tmp_tendency  = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
            double [:] mean_tendency = np.empty((Gr.dims.npg,),dtype=np.double,order='c')

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

        #Output advective forcing tendencies
        tmp_tendency[:] = 0.0



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
        Py_ssize_t kmax = dims.nlg[2] -dims.gw -1
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2]
        Py_ssize_t jstride = dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k
        double dxi = dims.dxi[2]
        double tend
    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    tend = (values[ijk+1] - values[ijk]) * dxi * subsidence[k]
                    tendencies[ijk] -= tend
                for k in xrange(kmax, dims.nlg[2]):
                    ijk = ishift + jshift + k
                    tendencies[ijk] -= tend

    return

cdef apply_nudging(Grid.DimStruct *dims, double *coefficient, double *mean, double *values, double *tendencies):

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
        double phim, fluxm
        double phip, fluxp

    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    tendencies[ijk] = tendencies[ijk] - (values[ijk] - mean[k]) * coefficient[k]

    return

cdef apply_ls_advection_entropy(Grid.DimStruct *dims, double *tendencies, double *temperature, double *ls_adv_Q):

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

    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    tendencies[ijk] += ls_adv_Q[k]/temperature[ijk]

    return

cdef apply_ls_advection_qt(Grid.DimStruct *dims, double *tendencies, double *ls_adv_qt):

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

    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    tendencies[ijk] += ls_adv_qt[k]

    return

from scipy.interpolate import pchip
def interp_pchip(z_out, z_in, v_in, pchip_type=True):
    if pchip_type:
        p = pchip(z_in, v_in, extrapolate=True)
        return p(z_out)
    else:
        return np.interp(z_out, z_in, v_in)
