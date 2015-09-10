#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport Thermodynamics
from thermodynamic_functions cimport cpm_c, pv_c, pd_c, exner_c
from entropies cimport sv_c, sd_c
import numpy as np
import cython
from libc.math cimport fabs, cos, fmin, fmax
from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI
include 'parameters.pxi'
from Initialization import thetal_mpace, sat_adjst

cdef class Forcing:
    def __init__(self, namelist):
        casename = namelist['meta']['casename']
        if casename == 'SullivanPatton':
            self.scheme = ForcingSullivanPatton()
        elif casename == 'Bomex':
            self.scheme = ForcingBomex()
        elif casename == 'Gabls':
            self.scheme = ForcingGabls()
        elif casename == 'DyCOMS_RF01':
            self.scheme = ForcingDyCOMS_RF01()
        elif casename == 'Mpace':
            self.scheme = ForcingMpace()
        elif casename == 'Isdac':
            self.scheme = ForcingIsdac()
        else:
            self.scheme= ForcingNone()
        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.scheme.initialize(Gr, RS, Th, NS, Pa)
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV):
        self.scheme.update(Gr, Ref, PV, DV)
        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        self.scheme.stats_io(Gr, Ref, PV, DV, NS, Pa)

        return


cdef class ForcingNone:
    def __init__(self):
        pass
    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV):
        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return

cdef class ForcingBomex:
    def __init__(self):
        return

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
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
                if Gr.zl[k] <= 1500.0:
                    self.subsidence[k] = 0.0 + Gr.zl[k]*(-0.65/100.0 - 0.0)/(1500.0 - 0.0)
                if Gr.zl[k] > 1500.0 and Gr.zl[k] <= 2100.0:
                    self.subsidence[k] = -0.65/100 + (Gr.zl[k] - 1500.0)* (0.0 - -0.65/100.0)/(2100.0 - 1500.0)


        #Initialize Statistical Output
        NS.add_profile('s_subsidence_tendency', Gr, Pa)
        NS.add_profile('qt_subsidence_tendency', Gr, Pa)
        NS.add_profile('u_subsidence_tendency', Gr, Pa)
        NS.add_profile('v_subsidence_tendency', Gr, Pa)
        NS.add_profile('u_coriolis_tendency', Gr, Pa)
        NS.add_profile('v_coriolis_tendency',Gr, Pa)

        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV):

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

        #Apply Coriolis Forcing
        coriolis_force(&Gr.dims,&PV.values[u_shift],&PV.values[v_shift],&PV.tendencies[u_shift],
                       &PV.tendencies[v_shift],&self.ug[0], &self.vg[0],self.coriolis_param, Ref.u0, Ref.v0)

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

        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[s_shift],&PV.tendencies[s_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[qt_shift],&PV.tendencies[qt_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[u_shift],&PV.tendencies[u_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[v_shift],&PV.tendencies[v_shift])
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
            double [:] mean_tendency = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
            double [:] mean_tendency_2 = np.zeros((Gr.dims.nlg[2]),dtype=np.double,order='c')


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
        coriolis_force(&Gr.dims,&PV.values[u_shift],&PV.values[v_shift],&tmp_tendency[0],
                       &tmp_tendency_2[0],&self.ug[0], &self.vg[0],self.coriolis_param, Ref.u0, Ref.v0)
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        mean_tendency_2 = Pa.HorizontalMean(Gr,&tmp_tendency_2[0])
        NS.write_profile('u_coriolis_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('v_coriolis_tendency',mean_tendency_2[Gr.dims.gw:-Gr.dims.gw],Pa)

        return

cdef class ForcingSullivanPatton:
    def __init__(self):

        return
    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.ug = np.ones(Gr.dims.nlg[2],dtype=np.double, order='c') #m/s
        self.vg = np.zeros(Gr.dims.nlg[2],dtype=np.double, order='c')  #m/s
        self.coriolis_param = 1.0e-4 #s^{-1}

        NS.add_profile('u_coriolis_tendency', Gr, Pa)
        NS.add_profile('v_coriolis_tendency',Gr, Pa)
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV):
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


cdef class ForcingGabls:
    def __init__(self):
        return

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.ug = np.ones(Gr.dims.nlg[2],dtype=np.double,order='c') * 8.0
        self.vg = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.coriolis_param = 1.39e-4 #s^{-1}
        NS.add_profile('u_coriolis_tendency', Gr, Pa)
        NS.add_profile('v_coriolis_tendency',Gr, Pa)

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV):
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
    def __init__(self):
        self.divergence = 3.75e-6
        return

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t k

        self.w = np.empty((Gr.dims.nlg[2]),dtype=np.double, order='c')
        with nogil:
            for k in range(Gr.dims.nlg[2]):
                self.w_half[k] = -Gr.zl[k] * self.divergence

        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV):
        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr,'qt')

        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[s_shift],&PV.tendencies[s_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[qt_shift],&PV.tendencies[qt_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[u_shift],&PV.tendencies[u_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[v_shift],&PV.tendencies[v_shift])



        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        return


cdef class ForcingMpace:
    def __init__(self):
        return

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.ug = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.vg = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.dtdt = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.dqtdt = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.subsidence = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.divergence = 5.8e-6
        self.initial_u = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.initial_v = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.initial_u[:] = -13.0
        self.initial_v[:] = -3.0
        self.nudging_timescale = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.nudging_timescale[:] = 1/(3600.0*2.0)

        cdef:
            Py_ssize_t k
            double omega

        with nogil:
            for k in xrange(Gr.dims.nlg[2]):

                #Set large scale cooling
                self.dtdt[k] = fmin(-4.0, -15.0*(1.0 - (Ref.Pg - Ref.p0_half[k])/21818.0))/(3600 * 24.0)

                #Set large scale drying
                self.dqtdt[k] = fmin(-0.164, -3*(1.0 - (Ref.Pg - Ref.p0_half[k])/15171.0))/(1000*3600*24.0)

                #Set large scale subsidence
                omega = fmin(self.divergence*(Ref.Pg - Ref.p0_half[k]), self.divergence*(Ref.Pg - 85000.0))
                self.subsidence[k] = -omega / (Ref.rho0_half[0] * g)

        #Initialize Statistical Output
        NS.add_profile('s_subsidence_tendency', Gr, Pa)
        NS.add_profile('qt_subsidence_tendency', Gr, Pa)
        NS.add_profile('u_subsidence_tendency', Gr, Pa)
        NS.add_profile('v_subsidence_tendency', Gr, Pa)
        NS.add_profile('u_coriolis_tendency', Gr, Pa)
        NS.add_profile('v_coriolis_tendency',Gr, Pa)

        return

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV):

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

        # #Apply Coriolis Forcing
        # coriolis_force(&Gr.dims,&PV.values[u_shift],&PV.values[v_shift],&PV.tendencies[u_shift],
        #                &PV.tendencies[v_shift],&self.ug[0], &self.vg[0],self.coriolis_param, Ref.u0, Ref.v0)

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
                                                         * self.dtdt[k] * rho0)/t
                        PV.tendencies[s_shift + ijk] += (sv_c(pv,t) - sd_c(pd,t))*self.dqtdt[k]
                        PV.tendencies[qt_shift + ijk] += self.dqtdt[k]

        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[s_shift],&PV.tendencies[s_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[qt_shift],&PV.tendencies[qt_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[u_shift],&PV.tendencies[u_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[v_shift],&PV.tendencies[v_shift])

        apply_nudging(&Gr.dims,&self.nudging_timescale[0],&self.initial_u[0],&PV.values[u_shift],&PV.tendencies[u_shift])
        apply_nudging(&Gr.dims,&self.nudging_timescale[0],&self.initial_v[0],&PV.values[v_shift],&PV.tendencies[v_shift])

        return

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
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
            double [:] mean_tendency = np.empty((Gr.dims.npg,),dtype=np.double,order='c')
            double [:] mean_tendency_2 = np.zeros((Gr.dims.npg),dtype=np.double,order='c')


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

        # #Output Coriolis tendencies
        # tmp_tendency[:] = 0.0
        # coriolis_force(&Gr.dims,&PV.values[u_shift],&PV.values[v_shift],&tmp_tendency[0],
        #                &tmp_tendency_2[0],&self.ug[0], &self.vg[0],self.coriolis_param, Ref.u0, Ref.v0)
        # mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        # mean_tendency_2 = Pa.HorizontalMean(Gr,&tmp_tendency_2[0])
        # NS.write_profile('u_coriolis_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)
        # NS.write_profile('v_coriolis_tendency',mean_tendency_2[Gr.dims.gw:-Gr.dims.gw],Pa)

        return


cdef class ForcingIsdac:
    def __init__(self):
        self.divergence = 5e-6
        return

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.initial_entropy = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.initial_qt = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.nudge_coeff_velocities = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.nudge_coeff_scalars = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.initial_u = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.initial_v = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.w_half =  np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')

        cdef:
            Py_ssize_t k
            double [:] thetal = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
            #double [:] qt = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
            double T, ql

        #self.w = np.empty((Gr.dims.nlg[2]),dtype=np.double, order='c')
        with nogil:
            for k in range(Gr.dims.nlg[2]):
                if Gr.zl[k] < 825.0:
                    self.w_half[k] = -Gr.zl[k] * self.divergence
                else:
                    self.w_half[k] = -0.4125e-2

        #Get profiles for nudging
        for k in xrange(Gr.dims.nlg[2]):

            #Set thetal and qt profile
            if Gr.zl_half[k] < 400.0:
                thetal[k] = 265.0 + 0.004 * (Gr.zl_half[k] - 400.0)
                self.initial_qt[k] = 1.5 - 0.00075 * (Gr.zl_half[k] - 400.0)
            if Gr.zl_half[k] >= 400.0 and Gr.zl_half[k] < 825.0:
                thetal[k] = 265.0
                self.initial_qt[k] = 1.5
            if Gr.zl_half[k] >= 825.0 and Gr.zl_half[k] < 2045.0:
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
            if Gr.zl_half[k] > 1200.0 and Gr.zl_half[k] <= 1500.0:
                self.nudge_coeff_scalars[k] = (1/3600.0)*0.5*(1.0 - cos(pi * (Gr.zl_half[k] - 1200.0)/300.0))
            if Gr.zl_half[k] > 1500.0:
                self.nudge_coeff_scalars[k] = 1/3600.0

            if Gr.zl_half[k] <= 825.0:
                self.nudge_coeff_velocities[k] = (1/7200.0)*0.5*(1 - cos(pi*Gr.zl_half[k]/825.0))
            if Gr.zl_half[k] > 825.0:
                self.nudge_coeff_velocities[k] = 1/7200.0

       #Initialize Statistical Output
        NS.add_profile('s_subsidence_tendency', Gr, Pa)
        NS.add_profile('qt_subsidence_tendency', Gr, Pa)
        NS.add_profile('u_subsidence_tendency', Gr, Pa)
        NS.add_profile('v_subsidence_tendency', Gr, Pa)
        NS.add_profile('u_nudging_tendency', Gr, Pa)
        NS.add_profile('v_nudging_tendency',Gr, Pa)
        NS.add_profile('s_nudging_tendency',Gr, Pa)
        NS.add_profile('qt_nudging_tendency',Gr, Pa)

        return

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV):
        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr,'qt')

        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[s_shift],&PV.tendencies[s_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[qt_shift],&PV.tendencies[qt_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[u_shift],&PV.tendencies[u_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.w_half[0],&PV.values[v_shift],&PV.tendencies[v_shift])

        apply_nudging(&Gr.dims,&self.nudge_coeff_scalars[0],&self.initial_entropy[0],&PV.values[s_shift],&PV.tendencies[s_shift])
        apply_nudging(&Gr.dims,&self.nudge_coeff_scalars[0],&self.initial_qt[0],&PV.values[qt_shift],&PV.tendencies[qt_shift])
        apply_nudging(&Gr.dims,&self.nudge_coeff_velocities[0],&self.initial_u[0],&PV.values[u_shift],&PV.tendencies[u_shift])
        apply_nudging(&Gr.dims,&self.nudge_coeff_velocities[0],&self.initial_v[0],&PV.values[v_shift],&PV.tendencies[v_shift])


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
        double phim, fluxm
        double phip, fluxp
        double dxi = dims.dxi[2]
    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    tendencies[ijk] = tendencies[ijk] - subsidence[k]*(values[ijk+1]-values[ijk])*dxi
                    # phip = values[ijk]
                    # phim = values[ijk+1]
                    # fluxp = (0.5*(subsidence[k]+fabs(subsidence[k]))*phip + 0.5*(subsidence[k]-fabs(subsidence[k]))*phim)*rho0[k]
                    # phip = values[ijk-1]
                    # phim = values[ijk]
                    # fluxm = (0.5*(subsidence[k]+fabs(subsidence[k]))*phip + 0.5*(subsidence[k]-fabs(subsidence[k]))*phim)*rho0[k]
                    # tendencies[ijk] = tendencies[ijk] - (fluxp - fluxm)*dxi/rho0_half[k]
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