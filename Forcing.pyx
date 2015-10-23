#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
from thermodynamic_functions cimport cpm_c, pv_c, pd_c, exner_c
from entropies cimport sv_c, sd_c
import numpy as np
import cython
from libc.math cimport fabs, sin
from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI
include 'parameters.pxi'

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
            self.scheme = ForcingDyCOMS_RF01()
        elif casename == 'SMOKE':
            self.scheme = ForcingNone()
        elif casename == 'Rico':
            self.scheme = ForcingRico()
        else:
            Pa.root_print('No focing for casename: ' +  casename)
            Pa.root_print('Killing simulation now!!!')
            Pa.kill()
        return

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.scheme.initialize(Gr, NS, Pa)
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
    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
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
    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
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

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
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
    def __init__(self):
        self.divergence = 3.75e-6
        self.coriolis_param = 2.0 * omega * sin(31.5 * pi / 180.0 )
        return

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t k

        self.subsidence = np.empty((Gr.dims.nlg[2]),dtype=np.double, order='c')
        self.ug = np.empty((Gr.dims.nlg[2]),dtype=np.double, order='c')
        self.vg = np.empty((Gr.dims.nlg[2]),dtype=np.double, order='c')
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
        #apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&DV.values[thetali_shift],&thetal_subs[0])
        #apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[qt_shift],&qt_subs[0])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[u_shift],&PV.tendencies[u_shift])
        apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0],&PV.values[v_shift],&PV.tendencies[v_shift])

        #Apply large scale source terms
        #with nogil:
        #    for i in xrange(imin,imax):
        #        ishift = i * istride
        #        for j in xrange(jmin,jmax):
        #            jshift = j * jstride
        #            for k in xrange(kmin,kmax):
        #                ijk = ishift + jshift + k
        #                p0 = Ref.p0_half[k]
        #                rho0 = Ref.rho0_half[k]
        #                qt = PV.values[qt_shift + ijk]
        #                qv = qt - DV.values[ql_shift + ijk]
        #                pd = pd_c(p0,qt,qv)
        #                pv = pv_c(p0,qt,qv)
        #                t  = DV.values[t_shift + ijk]
        #                PV.tendencies[s_shift + ijk] += (cpm_c(qt)
        #                                                 * thetal_subs[ijk] * exner_c(p0) * rho0)/t
        #                PV.tendencies[s_shift + ijk] += (sv_c(pv,t) - sd_c(pd,t))*qt_subs[ijk]
        #                PV.tendencies[qt_shift + ijk] += qt_subs[ijk]



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

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
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
                    tendencies[ijk] -= (values[ijk+1] - values[ijk]) * dxi * subsidence[k]

    return

