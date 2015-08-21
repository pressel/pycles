cimport ParallelMPI
cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables

cimport PressureFFTSerial
cimport PressureFFTParallel

import numpy as np
cimport numpy as np


import cython

cdef class PressureSolver:
    def __init__(self):
        pass

    cpdef initialize(self,namelist, Grid.Grid Gr,ReferenceState.ReferenceState RS ,DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI PM):

        DV.add_variables('dynamic_pressure','Pa','sym',PM)
        DV.add_variables('divergence','1/s','sym',PM)

        self.divergence = np.zeros(Gr.dims.npl,dtype=np.double, order='c')
        #self.poisson_solver = PressureFFTSerial.PressureFFTSerial()
        self.poisson_solver = PressureFFTParallel.PressureFFTParallel()
        self.poisson_solver.initialize(Gr,RS,PM)

        return

    @cython.boundscheck(False)  #Turn off numpy array index bounds checking
    @cython.wraparound(False)   #Turn off numpy array wrap around indexing
    @cython.cdivision(True)
    cpdef update(self,Grid.Grid Gr, ReferenceState.ReferenceState RS,
                  DiagnosticVariables.DiagnosticVariables DV, PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI PM):

        cdef:
            long i
            long d
            long vel_shift
            long u_shift = PV.get_varshift(Gr,'u')
            long v_shift = PV.get_varshift(Gr,'v')
            long w_shift = PV.get_varshift(Gr,'w')
            long pres_shift = DV.get_varshift(Gr,'dynamic_pressure')
            long div_shift = DV.get_varshift(Gr,'divergence')

        cdef double [:] u3_mean = PM.HorizontalMean(Gr,&PV.values[w_shift])
        #Remove mean u3
        remove_mean_u3(&Gr.dims,&u3_mean[0],&PV.values[w_shift])
        u3_mean = PM.HorizontalMean(Gr,&PV.values[w_shift])

        #Zero the divergence array [Perhaps we can replace this with a C-Call to Memset]
        with nogil:
            for i in xrange(Gr.dims.npg):
                DV.values[div_shift + i] = 0.0

        #Now compute the momentum divergence
        for d in xrange(Gr.dims.dims):
            vel_shift = PV.velocity_directions[d]*Gr.dims.npg
            second_order_divergence(&Gr.dims, &RS.alpha0[0], &RS.alpha0_half[0],&PV.values[vel_shift],
                 &DV.values[div_shift] ,d)

        #Now call the pressure solver
        self.poisson_solver.solve(Gr, RS, DV, PM)

        #Update pressure boundary condition
        p_nv = DV.name_index['dynamic_pressure']
        DV.communicate_variable(Gr,PM,p_nv)

        #Apply pressure correction
        second_order_pressure_correction(&Gr.dims,&DV.values[pres_shift],
                                         &PV.values[u_shift],&PV.values[v_shift],&PV.values[w_shift])

        #Switch this call at for a single variable boundary condition update
        PV.update_all_bcs(Gr,PM)

        return


@cython.boundscheck(False)  #Turn off numpy array index bounds checking
@cython.wraparound(False)   #Turn off numpy array wrap around indexing
@cython.cdivision(True)
cdef void second_order_pressure_correction(Grid.DimStruct *dims, double *p, double *u, double *v, double *w ):

    cdef:
        int imin = 0
        int jmin = 0
        int kmin = 0
        int imax = dims.nlg[0] - 1
        int jmax = dims.nlg[1] - 1
        int kmax = dims.nlg[2] - 1
        int istride = dims.nlg[1] * dims.nlg[2]
        int jstride = dims.nlg[2]
        int ishift, jshift
        int i,j,k, ijk
        int ip1 = istride
        int jp1 = jstride
        int kp1 = 1

    for i in xrange(imin,imax):
        ishift = istride * i
        for j in xrange(jmin,jmax):
            jshift = jstride * j
            for k in xrange(kmin,kmax):
                ijk = ishift + jshift + k
                u[ijk] -=  (p[ijk + ip1] - p[ijk])*dims.dxi[0]
                v[ijk] -=  (p[ijk + jp1] - p[ijk])*dims.dxi[1]
                w[ijk] -=  (p[ijk + kp1] - p[ijk])*dims.dxi[2]  #(p[ijk + kp1] - p[ijk])*dims.dxi[2]

    return


@cython.boundscheck(False)  #Turn off numpy array index bounds checking
@cython.wraparound(False)   #Turn off numpy array wrap around indexing
@cython.cdivision(True)
cdef void remove_mean_u3(Grid.DimStruct *dims, double *u3_mean, double *velocity):

    cdef:
        int imin = 0
        int jmin = 0
        int kmin = 0
        int imax = dims.nlg[0]
        int jmax = dims.nlg[1]
        int kmax = dims.nlg[2]
        int istride = dims.nlg[1] * dims.nlg[2]
        int jstride = dims.nlg[2]
        int ishift, jshift
        int ijk, i, j, k

    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j * jstride
                for k in xrange(kmin,kmax):
                     ijk = ishift + jshift + k
                     velocity[ijk] = velocity[ijk] - u3_mean[k]

    return


@cython.boundscheck(False)  #Turn off numpy array index bounds checking
@cython.wraparound(False)   #Turn off numpy array wrap around indexing
@cython.cdivision(True)
cdef void second_order_divergence(Grid.DimStruct *dims, double *alpha0, double *alpha0_half, double *velocity,
                                  double *divergence, long d):

    cdef:
        int imin = dims.gw
        int jmin = dims.gw
        int kmin = dims.gw
        int imax = dims.nlg[0] - dims.gw
        int jmax = dims.nlg[1] - dims.gw
        int kmax = dims.nlg[2] - dims.gw
        int istride = dims.nlg[1] * dims.nlg[2]
        int jstride = dims.nlg[2]
        int ishift, jshift
        int i,j,k,ijk

        #Compute the s+trides given the dimensionality
        int [3] p1 = [istride, jstride, 1]
        int sm1 =  -p1[d]
        double dxi = 1.0/dims.dx[d]

    if d != 2:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j * jstride
                for k in xrange(kmin,kmax):
                     ijk = ishift + jshift + k
                     divergence[ijk] += (velocity[ijk]/alpha0_half[k]  - velocity[ijk+sm1]/alpha0_half[k])*dxi
    else:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j * jstride
                for k in xrange(kmin,kmax):
                     ijk = ishift + jshift + k
                     divergence[ijk] += ((velocity[ijk]) /alpha0[k] - (velocity[ijk+sm1])/alpha0[k-1])*dxi

    return