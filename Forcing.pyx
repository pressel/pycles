cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables

cdef class Forcing:
    def __init__(self, namelist):
        casename = namelist['meta']['casename']
        if casename == 'SullivanPatton':
            self.scheme = ForcingSullivanPatton()
        else:
            self.scheme= ForcingNone()

    cpdef initialize(self):
        self.scheme.initialize()

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV):
        self.scheme.update(Gr, PV)


cdef class ForcingNone:
    def __init__(self):
        pass
    cpdef initialize(self):
        return

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV):
        return


cdef class ForcingSullivanPatton:
    def __init__(self):
        self.ug = 1.0 #m/s
        self.vg = 0.0 #m/s
        self.coriolis_param = 1.0e-4 #s^{-1}
        return
    cpdef initialize(self):
        return
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV):
        cdef:
            long u_shift = PV.get_varshift(Gr, 'u')
            long v_shift = PV.get_varshift(Gr, 'v')


        coriolis_force(&Gr.dims,&PV.values[u_shift],&PV.values[v_shift],&PV.tendencies[u_shift],
                       &PV.tendencies[v_shift],self.coriolis_param, self.ug, self.vg )


        return



cdef coriolis_force(Grid.DimStruct *dims, double *u, double *v, double *ut, double *vt, double coriolis_param, double ug, double vg ):
    cdef:
        int imin = dims.gw
        int jmin = dims.gw
        int kmin = dims.gw

        int imax = dims.nlg[0] -dims.gw
        int jmax = dims.nlg[1] -dims.gw
        int kmax = dims.nlg[2] -dims.gw

        int istride = dims.nlg[1] * dims.nlg[2];
        int jstride = dims.nlg[2];

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
                    ut[ijk] = ut[ijk] - coriolis_param * (vg - v_at_u)
                    vt[ijk] = vt[ijk] + coriolis_param * (ug - u_at_v)

    return


