import numpy as np
cimport numpy as np
cimport ParallelMPI
cimport Grid
cimport PrognosticVariables
from thermodynamic_functions cimport exner_c, entropy_from_thetas_c, thetas_t_c, qv_star_c, thetas_c
cimport ReferenceState
import time
import cython
from libc.math cimport sqrt, fmin, cos
include 'parameters.pxi'

def InitializationFactory(namelist):

        casename = namelist['meta']['casename']
        if casename == 'SullivanPatton':
            return InitSullivanPatton
        elif casename == 'StableBubble':
            return InitStableBubble
        elif casename == 'SaturatedBubble':
            return InitSaturatedBubble
        elif casename == 'Bomex':
            return InitBomex
        else:
            pass

@cython.boundscheck(False)  #Turn off numpy array index bounds checking
@cython.wraparound(False)   #Turn off numpy array wrap around indexing
@cython.cdivision(True)
def InitStableBubble(Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th):

    #First generate reference profiles
    RS.Pg = 1.0e5
    RS.Tg = 300.0
    RS.qtg = 0.0
    RS.u0 = 0.0
    RS.v0 = 0.0
    RS.initialize(Gr,Th)

    #Get the variable number for each of the velocity components
    cdef:
        int u_varshift = PV.get_varshift(Gr,'u')
        int v_varshift = PV.get_varshift(Gr,'v')
        int w_varshift = PV.get_varshift(Gr,'w')
        int s_varshift = PV.get_varshift(Gr,'s')
        int i,j,k
        int ishift, jshift
        int ijk
        double t
        double dist

    t_min = 9999.9
    for i in xrange(Gr.dims.nlg[0]):
        ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[u_varshift + ijk] = 0.0
                PV.values[v_varshift + ijk] = 0.0
                PV.values[w_varshift + ijk] = 0.0
                dist  = np.sqrt(((Gr.x_half[i + Gr.dims.indx_lo[0]]/1000.0 - 25.6)/4.0)**2.0 + ((Gr.z_half[k + Gr.dims.indx_lo[2]]/1000.0 - 3.0)/2.0)**2.0)
                dist = fmin(dist,1.0)
                t = (300.0 )*exner_c(RS.p0_half[k]) - 15.0*( cos(np.pi * dist) + 1.0) /2.0
                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],t,0.0,0.0,0.0)

    return

@cython.boundscheck(False)  #Turn off numpy array index bounds checking
@cython.wraparound(False)   #Turn off numpy array wrap around indexing
@cython.cdivision(True)
def InitSaturatedBubble(Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th):

    #First generate reference profiles
    RS.Pg = 1.0e5
    RS.qtg = 0.02
    #RS.Tg = 300.0

    thetas_sfc = 320.0
    qt_sfc = 0.0196 #RS.qtg
    RS.qtg = qt_sfc

    RS.u0 = 0.0
    RS.v0 = 0.0

    def theta_to_T(p0_,thetas_,qt_):


         T1 = Tt
         T2 = Tt + 1.

         pv1 = Th.get_pv_star(T1)
         pv2 = Th.get_pv_star(T2)

         qs1 = qv_star_c(p0_, RS.qtg,pv1)

         ql1 = np.max([0.0,qt_ - qs1])
         L1 = Th.get_lh(T1)
         f1 = thetas_ - thetas_t_c(p0_,T1,qt_,qt_-ql1,ql1,L1)

         delta = np.abs(T1 - T2)
         while delta >= 1e-12:


            L2 = Th.get_lh(T2)
            pv2 = Th.get_pv_star(T2)
            qs2 = qv_star_c(p0_, RS.qtg, pv2)
            ql2 = np.max([0.0,qt_ - qs2])
            f2 = thetas_ - thetas_t_c(p0_,T2,qt_,qt_-ql2,ql2,L2)

            Tnew = T2 - f2 * (T2 - T1)/(f2 - f1)
            T1 = T2
            T2 = Tnew
            f1 = f2

            delta = np.abs(T1 - T2)
         return T2, ql2

    RS.Tg, ql = theta_to_T(RS.Pg,thetas_sfc,qt_sfc)
    RS.initialize(Gr,Th)

    #Get the variable number for each of the velocity components
    cdef:
        int u_varshift = PV.get_varshift(Gr,'u')
        int v_varshift = PV.get_varshift(Gr,'v')
        int w_varshift = PV.get_varshift(Gr,'w')
        int s_varshift = PV.get_varshift(Gr,'s')
        int qt_varshift = PV.get_varshift(Gr,'qt')
        int i,j,k
        int ishift, jshift
        int ijk
        double t
        double dist
        double thetas

    for i in xrange(Gr.dims.nlg[0]):
        ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                dist = np.sqrt(((Gr.x_half[i + Gr.dims.indx_lo[0]]/1000.0 - 10.0)/2.0)**2.0 + ((Gr.z_half[k + Gr.dims.indx_lo[2]]/1000.0 - 2.0)/2.0)**2.0)
                dist = np.minimum(1.0,dist)
                thetas = RS.Tg
                thetas += 2.0 * np.cos(np.pi * dist / 2.0)**2.0
                PV.values[s_varshift + ijk] = entropy_from_thetas_c(thetas,RS.qtg)
                PV.values[u_varshift + ijk] = 0.0
                PV.values[v_varshift + ijk] = 0.0
                PV.values[w_varshift + ijk] = 0.0
                PV.values[qt_varshift + ijk] = RS.qtg

    return

@cython.boundscheck(False)  #Turn off numpy array index bounds checking
@cython.wraparound(False)   #Turn off numpy array wrap around indexing
@cython.cdivision(True)
def InitSullivanPatton(Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th):

    #First generate the reference profiles
    RS.Pg = 1.0e5  #Pressure at ground
    RS.Tg = 300.0  #Temperature at ground
    RS.qtg = 0.0   #Total water mixing ratio at surface
    RS.u0 = 1.0  # velocities removed in Galilean transformation
    RS.v0 = 0.0

    RS.initialize(Gr, Th)

    #Get the variable number for each of the velocity components
    cdef:
        int u_varshift = PV.get_varshift(Gr,'u')
        int v_varshift = PV.get_varshift(Gr,'v')
        int w_varshift = PV.get_varshift(Gr,'w')
        int s_varshift = PV.get_varshift(Gr,'s')
        int i,j,k
        int ishift, jshift
        int ijk
        double [:] theta = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double t

        #Generate initial perturbations (here we are generating more than we need)
        cdef double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
        cdef double theta_pert_

    for k in xrange(Gr.dims.nlg[2]):
        if Gr.zl_half[k] <=  974.0:
            theta[k] = 300.0
        elif Gr.zl_half[k] <= 1074.0:
            theta[k] = 300.0 + (Gr.zl_half[k] - 974.0) * 0.08
        else:
            theta[k] = 308.0 + (Gr.zl_half[k] - 1074.0) * 0.003

    cdef double [:] p0 = RS.p0_half

    #Now loop and set the initial condition
    #First set the velocities
    for i in xrange(Gr.dims.nlg[0]):
        ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[u_varshift + ijk] = 1.0
                PV.values[v_varshift + ijk] = 0.0
                PV.values[w_varshift + ijk] = 0.0

                #Now set the entropy prognostic variable including a potential temperature perturbation
                if Gr.zl_half[k] < 200.0:
                    theta_pert_ = theta_pert[ijk]
                else:
                    theta_pert_ = 0.0
                t = (theta[k] + theta_pert_)*exner_c(RS.p0_half[k])

                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],t,0.0,0.0,0.0)
    return

@cython.boundscheck(False)  #Turn off numpy array index bounds checking
@cython.wraparound(False)   #Turn off numpy array wrap around indexing
@cython.cdivision(True)
def InitBomex(Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th):

    #First generate the reference profiles
    RS.Pg = 1.015e5  #Pressure at ground
    RS.Tg = 300.4  #Temperature at ground
    RS.qtg = 0.002245   #Total water mixing ratio at surface

    RS.initialize(Gr, Th)

    #Get the variable number for each of the velocity components
    cdef:
        int u_varshift = PV.get_varshift(Gr,'u')
        int v_varshift = PV.get_varshift(Gr,'v')
        int w_varshift = PV.get_varshift(Gr,'w')
        int s_varshift = PV.get_varshift(Gr,'s')
        int qt_varshift = PV.get_varshift(Gr,'qt')
        int i,j,k
        int ishift, jshift
        int ijk
        double temp
        double [:] thetal = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double [:] qt = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double [:] u = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        int count

        theta_pert = np.random.random_sample(Gr.dims.npg)*0.1

    for k in xrange(Gr.dims.nlg[2]):

        #Set Thetal profile
        if Gr.zl_half[k] <= 520.:
            thetal[k] = 298.0
        if Gr.zl_half[k] > 520.0 and Gr.zl_half[k] <= 1480.0:
            thetal[k] = 298.7 + (Gr.zl_half[k] - 520)  * (302.4 - 298.7)/(1480.0 - 520.0)
        if Gr.zl_half[k] > 1480.0 and Gr.zl_half[k] <= 2000:
            thetal[k] = 302.4 + (Gr.zl_half[k] - 1480.0) * (308.2 - 302.4)/(2000.0 - 1480.0)
        if Gr.zl_half[k] > 2000.0:
            thetal[k] = 308.2 + (Gr.zl_half[k] - 2000.0) * (311.85 - 308.2)/(3000.0 - 2000.0)

        #Set qt profile
        if Gr.zl_half[k] <= 520:
            qt[k] = 17.0 + (Gr.zl_half[k]) * (16.3-17.0)/520.0
        if Gr.zl_half[k] > 520.0 and Gr.zl_half[k] <= 1480.0:
            qt[k] = 16.3 + (Gr.zl_half[k] - 520.0)*(10.7 - 16.3)/(1480.0 - 520.0)
        if Gr.zl_half[k] > 1480.0 and Gr.zl_half[k] <= 2000.0:
            qt[k] = 10.7 + (Gr.zl_half[k] - 1480.0) * (4.2 - 10.7)/(2000.0 - 1480.0)
        if Gr.zl_half[k] > 2000.0:
            qt[k] = 4.2 + (Gr.zl_half[k] - 2000.0) * (3.0 - 4.2)/(3000.0  - 2000.0)

        #Change units to kg/kg
        qt[k]/= 1000.0

        #Set u profile
        if Gr.zl_half[k] <= 700.0:
            u[k] = -8.75
        if Gr.zl_half[k] > 700.0:
            u[k] = -8.75 + (Gr.zl_half[k] - 700.0) * (-4.61 - -8.75)/(3000.0 - 700.0)

    #Set velocities for Galilean transformation
    RS.v0 = 0.0
    RS.u0 = 0.5 * (np.amax(u)+np.amin(u))



    #Now loop and set the initial condition
    #First set the velocities
    count = 0
    for i in xrange(Gr.dims.nlg[0]):
        ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[u_varshift + ijk] = u[k]
                PV.values[v_varshift + ijk] = 0.0
                PV.values[w_varshift + ijk] = 0.0
                if Gr.z_half[k] <= 800.0:
                    temp = (thetal[k] + theta_pert[count]) * exner_c(RS.p0_half[k])
                else:
                    temp = (thetal[k]) * exner_c(RS.p0_half[k])
                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],temp,qt[k],0.0,0.0)
                PV.values[qt_varshift + ijk] = qt[k]
                count += 1