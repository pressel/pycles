cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
from thermodynamic_functions cimport cpm_c, pv_c, pd_c
from entropies cimport sv_c, sd_c
import numpy as np
import cython
from libc.math cimport fabs


cdef class Forcing:
    def __init__(self, namelist):
        casename = namelist['meta']['casename']
        if casename == 'SullivanPatton':
            self.scheme = ForcingSullivanPatton()
        if casename == 'Bomex':
            self.scheme = ForcingBomex()
        else:
            self.scheme= ForcingNone()

    cpdef initialize(self, Grid.Grid Gr):
        self.scheme.initialize(Gr)

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV):
        self.scheme.update(Gr, Ref, PV, DV)



cdef class ForcingNone:
    def __init__(self):
        pass
    cpdef initialize(self, Grid.Grid Gr):
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV):
        return

    cpdef stats_io(self):
        return

cdef class ForcingBomex:
    def __init__(self):
        return

    @cython.boundscheck(False)  #Turn off numpy array index bounds checking
    @cython.wraparound(False)   #Turn off numpy array wrap around indexing
    @cython.cdivision(True)
    cpdef initialize(self,Grid.Grid Gr):
        self.ug = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.vg = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.dtdt = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.dqtdt = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.subsidence = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.coriolis_param = 0.376e-4 #s^{-1}

        cdef:
            int k

        with nogil:
            for k in xrange(Gr.dims.nlg[2]):
                self.vg[k] = -10.0 + (1.8e-3)*Gr.zl_half[k]

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
        return



    @cython.boundscheck(False)  #Turn off numpy array index bounds checking
    @cython.wraparound(False)   #Turn off numpy array wrap around indexing
    @cython.cdivision(True)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV):

        cdef:
            long imin = Gr.dims.gw
            long jmin = Gr.dims.gw
            long kmin = Gr.dims.gw
            long imax = Gr.dims.nlg[0] - Gr.dims.gw
            long jmax = Gr.dims.nlg[1] - Gr.dims.gw
            long kmax = Gr.dims.nlg[2] - Gr.dims.gw
            long istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            long jstride = Gr.dims.nlg[2]
            long i,j,k,ishift,jshift,ijk
            long u_shift = PV.get_varshift(Gr, 'u')
            long v_shift = PV.get_varshift(Gr, 'v')
            long s_shift = PV.get_varshift(Gr, 's')
            long qt_shift = PV.get_varshift(Gr, 'qt')
            long t_shift = DV.get_varshift(Gr, 'temperature')
            long ql_shift = DV.get_varshift(Gr,'ql')
            double pd
            double pv
            double qt
            double qv
            double p0
            double rho0
            double t

        #Apply Coriolis Forcing
        coriolis_force(&Gr.dims,&PV.values[u_shift],&PV.values[v_shift],&PV.tendencies[u_shift],
                       &PV.tendencies[v_shift],&self.ug[0], &self.vg[0],self.coriolis_param  )

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
        return

    cpdef stats_io(self):
        return

cdef class ForcingSullivanPatton:
    def __init__(self):
        return
    cpdef initialize(self,Grid.Grid Gr):
        self.ug = np.ones(Gr.dims.nlg[2],dtype=np.double, order='c') #m/s
        self.vg = np.zeros(Gr.dims.nlg[2],dtype=np.double, order='c')  #m/s
        self.coriolis_param = 1.0e-4 #s^{-1}
        return
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV):
        cdef:
            long u_shift = PV.get_varshift(Gr, 'u')
            long v_shift = PV.get_varshift(Gr, 'v')

        coriolis_force(&Gr.dims,&PV.values[u_shift],&PV.values[v_shift],&PV.tendencies[u_shift],
                       &PV.tendencies[v_shift],&self.ug[0], &self.vg[0],self.coriolis_param  )
        return

    cpdef stats_io(self):

        return

cdef coriolis_force(Grid.DimStruct *dims, double *u, double *v, double *ut, double *vt, double *ug, double *vg, double coriolis_param ):
    cdef:
        int imin = dims.gw
        int jmin = dims.gw
        int kmin = dims.gw
        int imax = dims.nlg[0] -dims.gw
        int jmax = dims.nlg[1] -dims.gw
        int kmax = dims.nlg[2] -dims.gw
        int istride = dims.nlg[1] * dims.nlg[2]
        int jstride = dims.nlg[2]
        int ishift, jshift, ijk, i,j,k
        double u_at_v, v_at_u

    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    u_at_v = 0.25*(u[ijk] + u[ijk-istride] + u[ijk-istride+jstride] + u[ijk +jstride])
                    v_at_u = 0.25*(v[ijk] + v[ijk+istride] + v[ijk+istride-jstride] + v[ijk-jstride])
                    ut[ijk] = ut[ijk] - coriolis_param * (vg[k] - v_at_u)
                    vt[ijk] = vt[ijk] + coriolis_param * (ug[k] - u_at_v)
    return

cdef apply_subsidence(Grid.DimStruct *dims, double *rho0, double *rho0_half, double *subsidence, double* values,  double *tendencies):

    cdef:
        int imin = dims.gw
        int jmin = dims.gw
        int kmin = dims.gw
        int imax = dims.nlg[0] -dims.gw
        int jmax = dims.nlg[1] -dims.gw
        int kmax = dims.nlg[2] -dims.gw
        int istride = dims.nlg[1] * dims.nlg[2]
        int jstride = dims.nlg[2]
        int ishift, jshift, ijk, i,j,k
        double phim, fluxm
        double phip, fluxp

    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    phip = values[ijk]
                    phim = values[ijk+1]
                    fluxp = (0.5*(subsidence[k]+fabs(subsidence[k]))*phip + 0.5*(subsidence[k]-fabs(subsidence[k]))*phim)*rho0[k]
                    phip = values[ijk-1]
                    phim = values[ijk]
                    fluxm = (0.5*(subsidence[k]+fabs(subsidence[k]))*phip + 0.5*(subsidence[k]-fabs(subsidence[k]))*phim)*rho0[k]

                    tendencies[ijk] = tendencies[ijk] + rho0_half[k] * (fluxp - fluxm)/dims.dx[2]
    return
