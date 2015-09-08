#!python
#cython: boundscheck=False
#cython: wraparound=True
#cython: initializedcheck=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
cimport ParallelMPI
from NetCDFIO cimport NetCDFIO_Stats
cimport Grid
cimport PrognosticVariables

from thermodynamic_functions cimport exner_c, entropy_from_thetas_c, thetas_t_c, qv_star_c, thetas_c
cimport ReferenceState
import time
import cython
from libc.math cimport sqrt, fmin, cos, exp, fabs
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
        elif casename == 'Gabls':
            return InitGabls
        elif casename == 'Mpace':
            return InitMpace
        elif casename == 'Isdac':
            return InitIsdac
        elif casename == 'DYCOMS_RF01':
            return InitDYCOMS_RF01
        else:
            pass

def InitStableBubble(Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa ):

    #Generate reference profiles
    RS.Pg = 1.0e5
    RS.Tg = 300.0
    RS.qtg = 0.0
    #Set velocities for Galilean transformation
    RS.u0 = 0.0
    RS.v0 = 0.0

    RS.initialize(Gr, Th, NS, Pa)

    #Get the variable number for each of the velocity components
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift
        Py_ssize_t ijk
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

def InitSaturatedBubble(Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa ):

    #Generate reference profiles
    RS.Pg = 1.0e5
    RS.qtg = 0.02
    #RS.Tg = 300.0

    thetas_sfc = 320.0
    qt_sfc = 0.0196 #RS.qtg
    RS.qtg = qt_sfc

    #Set velocities for Galilean transformation
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
    RS.initialize(Gr, Th, NS, Pa)

    #Get the variable number for each of the velocity components
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift
        Py_ssize_t ijk
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
                PV.values[u_varshift + ijk] = 0.0 - RS.u0
                PV.values[v_varshift + ijk] = 0.0 - RS.v0
                PV.values[w_varshift + ijk] = 0.0
                PV.values[qt_varshift + ijk] = RS.qtg

    return

def InitSullivanPatton(Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa ):

    #Generate the reference profiles
    RS.Pg = 1.0e5  #Pressure at ground
    RS.Tg = 300.0  #Temperature at ground
    RS.qtg = 0.0   #Total water mixing ratio at surface
    RS.u0 = 1.0  # velocities removed in Galilean transformation
    RS.v0 = 0.0

    RS.initialize(Gr, Th, NS, Pa)

    #Get the variable number for each of the velocity components
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift
        Py_ssize_t ijk
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
    for i in xrange(Gr.dims.nlg[0]):
        ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[u_varshift + ijk] = 1.0 - RS.u0
                PV.values[v_varshift + ijk] = 0.0 - RS.v0
                PV.values[w_varshift + ijk] = 0.0

                #Now set the entropy prognostic variable including a potential temperature perturbation
                if Gr.zl_half[k] < 200.0:
                    theta_pert_ = (theta_pert[ijk] - 0.5)* 0.1
                else:
                    theta_pert_ = 0.0
                t = (theta[k] + theta_pert_)*exner_c(RS.p0_half[k])

                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],t,0.0,0.0,0.0)
    return

def InitBomex(Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa ):

    #First generate the reference profiles
    RS.Pg = 1.015e5  #Pressure at ground
    RS.Tg = 300.4  #Temperature at ground
    RS.qtg = 0.02245   #Total water mixing ratio at surface

    RS.initialize(Gr, Th, NS, Pa)

    #Get the variable number for each of the velocity components
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift
        Py_ssize_t ijk
        double temp
        double [:] thetal = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double [:] qt = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double [:] u = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        Py_ssize_t count

        theta_pert = np.random.random_sample(Gr.dims.npg)*0.1

    for k in xrange(Gr.dims.nlg[2]):

        #Set Thetal profile
        if Gr.zl_half[k] <= 520.:
            thetal[k] = 298.7
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
                PV.values[u_varshift + ijk] = u[k] - RS.u0
                PV.values[v_varshift + ijk] = 0.0 - RS.v0
                PV.values[w_varshift + ijk] = 0.0
                if Gr.z_half[k] <= 800.0:
                    temp = (thetal[k] + (theta_pert[count]-0.05)) * exner_c(RS.p0_half[k])
                else:
                    temp = (thetal[k]) * exner_c(RS.p0_half[k])
                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],temp,qt[k],0.0,0.0)
                PV.values[qt_varshift + ijk] = qt[k]
                count += 1

    return

def InitGabls(Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa ):

    #Generate the reference profiles
    RS.Pg = 1.0e5  #Pressure at ground
    RS.Tg = 265.0  #Temperature at ground
    RS.qtg = 0.0   #Total water mixing ratio at surface
    RS.u0 = 8.0  # velocities removed in Galilean transformation
    RS.v0 = 0.0

    RS.initialize(Gr, Th, NS, Pa)

    #Get the variable number for each of the velocity components
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift
        Py_ssize_t ijk
        double [:] theta = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double t

        #Generate initial perturbations (here we are generating more than we need)
        cdef double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
        cdef double theta_pert_

    for k in xrange(Gr.dims.nlg[2]):
        if Gr.zl_half[k] <=  100.0:
            theta[k] = 265.0

        else:
            theta[k] = 265.0 + (Gr.zl_half[k] - 100.0) * 0.01

    cdef double [:] p0 = RS.p0_half

    #Now loop and set the initial condition
    #First set the velocities
    for i in xrange(Gr.dims.nlg[0]):
        ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[u_varshift + ijk] = 8.0 - RS.u0
                PV.values[v_varshift + ijk] = 0.0 - RS.v0
                PV.values[w_varshift + ijk] = 0.0

                #Now set the entropy prognostic variable including a potential temperature perturbation
                if Gr.zl_half[k] < 50.0:
                    theta_pert_ = (theta_pert[ijk] - 0.5)* 0.1
                else:
                    theta_pert_ = 0.0
                t = (theta[k] + theta_pert_)*exner_c(RS.p0_half[k])

                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],t,0.0,0.0,0.0)
    return

def InitDYCOMS_RF01(Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa ):

    """
    Initialize the DYCOMS_RF01 case described in
    Bjorn Stevens, Chin-Hoh Moeng, Andrew S. Ackerman, Christopher S. Bretherton, Andreas Chlond, Stephan de Roode,
    James Edwards, Jean-Christophe Golaz, Hongli Jiang, Marat Khairoutdinov, Michael P. Kirkpatrick, David C. Lewellen,
    Adrian Lock, Frank Müller, David E. Stevens, Eoin Whelan, and Ping Zhu, 2005: Evaluation of Large-Eddy Simulations
    via Observations of Nocturnal Marine Stratocumulus. Mon. Wea. Rev., 133, 1443–1462.
    doi: http://dx.doi.org/10.1175/MWR2930.1
    :param Gr: Grid cdef extension class
    :param PV: PrognosticVariables cdef extension class
    :param RS: ReferenceState cdef extension class
    :param Th: Thermodynamics class
    :return: None
    """

    # Generate Reference Profiles
    RS.Pg = 1017.8 * 100.0
    RS.qtg = 9.0/1000.0
    RS.u0 = 7.0
    RS.v0 = -5.5

    # Use an exner function with values for Rd, and cp given in Stevens 2004 to compute temperature given $\theta_l$
    RS.Tg = 289.0 * (RS.Pg/p_tilde)**(287.0/1015.0)

    RS.initialize(Gr ,Th, NS, Pa)

    #Set up $\tehta_l$ and $\qt$ profiles
    cdef:
        Py_ssize_t i
        Py_ssize_t j
        Py_ssize_t k
        Py_ssize_t ijk, ishift, jshift
        Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
        Py_ssize_t jstride = Gr.dims.nlg[2]
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        double [:] thetal = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] qt = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')

    for k in xrange(Gr.dims.nlg[2]):
        if Gr.zl_half[k] <=840.0:
            thetal[k] = 289.0
            qt[k] = 9.0/1000.0
        if Gr.zl_half[k] > 840.0:
            thetal[k] = 297.5 + (Gr.zl_half[k] - 840.0)**(1.0/3.0)
            qt[k] = 1.5/1000.0

    def compute_thetal(p_,T_,ql_):
        theta_ = T_ / (p_/p_tilde)**(287.0/1015.0)
        return theta_ * exp(-2.47e6 * ql_ / (1015.0 * T_))

    def sat_adjst(p_,thetal_,qt_):
        '''
        Use saturation adjustment scheme to compute temperature and ql given thetal and qt.
        :param p: pressure [Pa]
        :param thetal: liquid water potential temperature  [K]
        :param qt:  total water specific humidity
        :return: T, ql
        '''

        #Compute temperature
        t_1 = thetal_ * (p_/p_tilde)**(287.0/1015.0)
        #Compute saturation vapor pressure
        pv_star_1 = Th.get_pv_star(t_1)
        #Compute saturation mixing ratio
        qs_1 = qv_star_c(p_,qt_,pv_star_1)

        if qt_ <= qs_1:
            #If not saturated return temperature and ql = 0.0
            return t_1, 0.0
        else:
            ql_1 = qt_ - qs_1
            f_1 = thetal_ - compute_thetal(p_,t_1,ql_1)
            t_2 = t_1 + 2.47e6*ql_1/1015.0
            pv_star_2 = Th.get_pv_star(t_2)
            qs_2 = qv_star_c(p_,qt_,pv_star_2)
            ql_2 = qt_ - qs_2

            while fabs(t_2 - t_1) >= 1e-9:
                pv_star_2 = Th.get_pv_star(t_2)
                qs_2 = qv_star_c(p_,qt_,pv_star_2)
                ql_2 = qt_ - qs_2
                f_2 = thetal_ - compute_thetal(p_, t_2, ql_2)
                t_n = t_2 - f_2 * (t_2 - t_1)/(f_2 - f_1)
                t_1 = t_2
                t_2 = t_n
                f_1 = f_2

            return t_2, ql_2

    #Generate initial perturbations (here we are generating more than we need)
    cdef double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
    cdef double theta_pert_

    for i in xrange(Gr.dims.nlg[0]):
        ishift = istride * i
        for j in xrange(Gr.dims.nlg[1]):
            jshift = jstride * j
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[ijk + u_varshift] = 0.0
                PV.values[ijk + v_varshift] = 0.0
                PV.values[ijk + w_varshift] = 0.0
                PV.values[ijk + qt_varshift]  = qt[k]

                #Now set the entropy prognostic variable including a potential temperature perturbation
                if Gr.zl_half[k] < 200.0:
                    theta_pert_ = (theta_pert[ijk] - 0.5)* 0.1
                else:
                    theta_pert_ = 0.0
                T,ql = sat_adjst(RS.p0_half[k],thetal[k] + theta_pert_,qt[k])
                PV.values[ijk + s_varshift] = Th.entropy(RS.p0_half[k], T, qt[k], ql, 0.0)

    return


def InitMpace(Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa ):

    '''
    Initialize the M-PACE case described in Klein et al. (2009): Intercomparison of model simulations
    of mixed-phase clouds observed during the ARM Mixed-Phase Arctic Cloud Experiment. I: Single-layer cloud

    :param Gr: Grid cdef extension class
    :param PV: PrognosticVariables cdef extension class
    :param RS: ReferenceState cdef extension class
    :param Th: Thermodynamics class
    :return: None

    '''

    #First generate the reference profiles
    RS.Pg = 1.01e5  #Pressure at ground
    RS.Tg = 274.04  #Temperature at ground
    RS.qtg = 0.00195   #Total water mixing ratio at surface

    RS.u0 = -13.0  # velocities removed in Galilean transformation
    RS.v0 = -3.0

    RS.initialize(Gr, Th, NS, Pa)

    #Get the variable number for each of the velocity components
    cdef:
        Py_ssize_t i
        Py_ssize_t j
        Py_ssize_t k
        Py_ssize_t ijk, ishift, jshift
        Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
        Py_ssize_t jstride = Gr.dims.nlg[2]
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        double [:] thetal = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] qt = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')


    for k in xrange(Gr.dims.nlg[2]):

        #Set thetal and qt profiles
        if RS.p0_half[k] >= 85000.:
            thetal[k] = 269.2
            qt[k] = 1.95
        if RS.p0_half[k] < 85000.:
            thetal[k] = 275.33 + 0.0791/100.*(81500. - RS.p0_half[k])
            qt[k] = 0.291 + 0.00204/100.*(RS.p0_half[k] - 59000.)

        #Change units to kg/kg
        qt[k]/= 1000.0

    # #Thetal defined in Klein et al. 2009
    # def thetal_mpace(p_,t_,ql_):
    #     t_cb = 263. #cloud base temperature
    #     return t_*(p_tilde/p_)**(Rd/cpd)*np.exp(-(2.26e6*ql_)/(cpd*t_cb))
    #
    # #Now do saturation adjustment to get temperature and ql
    # def sat_adjst(p_,thetal_,qt_):
    #     '''
    #     Use saturation adjustment scheme to compute temperature and ql given thetal and qt.
    #     :param p: pressure [Pa]
    #     :param thetal: liquid water potential temperature  [K]
    #     :param qt:  total water specific humidity
    #     :return: T, ql
    #     '''
    #
    #     #Compute temperature
    #     t_1 = thetal_ * (p_/p_tilde)**(Rd/cpd)
    #     #Compute saturation vapor pressure
    #     pv_star_1 = Th.get_pv_star(t_1)
    #     #Compute saturation mixing ratio
    #     qs_1 = qv_star_c(p_,qt_,pv_star_1)
    #
    #     if qt_ <= qs_1:
    #         #If not saturated return temperature and ql = 0.0
    #         return t_1, 0.0
    #     else:
    #         ql_1 = qt_ - qs_1
    #         f_1 = thetal_ - thetal_mpace(p_,t_1,ql_1)
    #         t_2 = t_1 + 2.26e6*ql_1/cpd
    #         pv_star_2 = Th.get_pv_star(t_2)
    #         qs_2 = qv_star_c(p_,qt_,pv_star_2)
    #         ql_2 = qt_ - qs_2
    #
    #         while fabs(t_2 - t_1) >= 1e-9:
    #             pv_star_2 = Th.get_pv_star(t_2)
    #             qs_2 = qv_star_c(p_,qt_,pv_star_2)
    #             ql_2 = qt_ - qs_2
    #             f_2 = thetal_ - thetal_mpace(p_, t_2, ql_2)
    #             t_n = t_2 - f_2 * (t_2 - t_1)/(f_2 - f_1)
    #             t_1 = t_2
    #             t_2 = t_n
    #             f_1 = f_2
    #
    #         return t_2, ql_2

    #Generate initial perturbations (here we are generating more than we need)
    cdef double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
    cdef double theta_pert_

    #Now loop and set the initial condition
    for i in xrange(Gr.dims.nlg[0]):
        ishift = istride * i
        for j in xrange(Gr.dims.nlg[1]):
            jshift = jstride * j
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[ijk + u_varshift] = 0.0
                PV.values[ijk + v_varshift] = 0.0
                PV.values[ijk + w_varshift] = 0.0
                PV.values[ijk + qt_varshift]  = qt[k]

                #Now set the entropy prognostic variable including a potential temperature perturbation
                if Gr.zl_half[k] < 200.0:
                    theta_pert_ = (theta_pert[ijk] - 0.5)* 0.1
                else:
                    theta_pert_ = 0.0
                T,ql = sat_adjst(RS.p0_half[k],thetal[k] + theta_pert_,qt[k],Th)
                PV.values[ijk + s_varshift] = Th.entropy(RS.p0_half[k], T, qt[k], ql, 0.0)


    return


def InitIsdac(Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa ):

    '''
    Initialize the ISDAC case described in Ovchinnikov et al. (2014):
    Intercomparison of large-eddy simulations of Arctic mixed-phase clouds:
    Importance of ice size distribution assumptions

    :param Gr: Grid cdef extension class
    :param PV: PrognosticVariables cdef extension class
    :param RS: ReferenceState cdef extension class
    :param Th: Thermodynamics class
    :return: None

    '''

    #First generate the reference profiles
    RS.Pg = 1.02e5  #Pressure at ground
    RS.Tg = 267.0  #Temperature at ground
    RS.qtg = 0.0015   #Total water mixing ratio at surface

    RS.initialize(Gr, Th, NS, Pa)


    #Get the variable number for each of the velocity components
    cdef:
        Py_ssize_t i
        Py_ssize_t j
        Py_ssize_t k
        Py_ssize_t ijk, ishift, jshift
        Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
        Py_ssize_t jstride = Gr.dims.nlg[2]
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        double [:] thetal = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] qt = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] v = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')


    for k in xrange(Gr.dims.nlg[2]):

        #Set thetal and qt profile
        if Gr.zl_half[k] < 400.0:
            thetal[k] = 265.0 + 0.004 * (Gr.zl_half[k] - 400.0)
            qt[k] = 1.5 - 0.00075 * (Gr.zl_half[k] - 400.0)
        if Gr.zl_half[k] >= 400.0 and Gr.zl_half[k] < 825.0:
            thetal[k] = 265.0
            qt[k] = 1.5
        if Gr.zl_half[k] >= 825.0 and Gr.zl_half[k] < 2045.0:
            thetal[k] = 266.0 + (Gr.zl_half[k] - 825.0) ** 0.3
            qt[k] = 1.2
        if Gr.zl_half[k] >= 2045.0:
            thetal[k] = 271.0 + (Gr.zl_half[k] - 2000.0) ** 0.33
            qt[k] = 0.5 - 0.000075 * (Gr.zl_half[k] - 2045.0)

        #Change units to kg/kg
        qt[k]/= 1000.0

        #Set u profile
        v[k] = -2.0 + 0.003 * Gr.zl_half[k]

    #Set velocities for Galilean transformation
    RS.u0 = -7.0
    RS.v0 = 0.5 * (np.amax(v)+np.amin(v))

    # #Thetal defined in Klein et al. 2009
    # def thetal_mpace(p_,t_,ql_):
    #     t_cb = 263. #cloud base temperature
    #     return t_*(p_tilde/p_)**(Rd/cpd)*np.exp(-(2.26e6*ql_)/(cpd*t_cb))
    #
    # #Now do saturation adjustment to get temperature and ql
    # def sat_adjst(p_,thetal_,qt_):
    #     '''
    #     Use saturation adjustment scheme to compute temperature and ql given thetal and qt.
    #     :param p: pressure [Pa]
    #     :param thetal: liquid water potential temperature  [K]
    #     :param qt:  total water specific humidity
    #     :return: T, ql
    #     '''
    #
    #     #Compute temperature
    #     t_1 = thetal_ * (p_/p_tilde)**(Rd/cpd)
    #     #Compute saturation vapor pressure
    #     pv_star_1 = Th.get_pv_star(t_1)
    #     #Compute saturation mixing ratio
    #     qs_1 = qv_star_c(p_,qt_,pv_star_1)
    #
    #     if qt_ <= qs_1:
    #         #If not saturated return temperature and ql = 0.0
    #         return t_1, 0.0
    #     else:
    #         ql_1 = qt_ - qs_1
    #         f_1 = thetal_ - thetal_mpace(p_,t_1,ql_1)
    #         t_2 = t_1 + 2.26e6*ql_1/cpd
    #         pv_star_2 = Th.get_pv_star(t_2)
    #         qs_2 = qv_star_c(p_,qt_,pv_star_2)
    #         ql_2 = qt_ - qs_2
    #
    #         while fabs(t_2 - t_1) >= 1e-9:
    #             pv_star_2 = Th.get_pv_star(t_2)
    #             qs_2 = qv_star_c(p_,qt_,pv_star_2)
    #             ql_2 = qt_ - qs_2
    #             f_2 = thetal_ - thetal_mpace(p_, t_2, ql_2)
    #             t_n = t_2 - f_2 * (t_2 - t_1)/(f_2 - f_1)
    #             t_1 = t_2
    #             t_2 = t_n
    #             f_1 = f_2
    #
    #         return t_2, ql_2

    #Generate initial perturbations (here we are generating more than we need)
    cdef double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
    cdef double theta_pert_

    #Now loop and set the initial condition
    for i in xrange(Gr.dims.nlg[0]):
        ishift = istride * i
        for j in xrange(Gr.dims.nlg[1]):
            jshift = jstride * j
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[ijk + u_varshift] = 0.0
                PV.values[ijk + v_varshift] = v[k] - RS.v0
                PV.values[ijk + w_varshift] = 0.0
                PV.values[ijk + qt_varshift]  = qt[k]

                #Now set the entropy prognostic variable including a potential temperature perturbation
                if Gr.zl_half[k] < 825.0:
                    theta_pert_ = (theta_pert[ijk] - 0.5)* 0.1
                else:
                    theta_pert_ = 0.0
                T,ql = sat_adjst(RS.p0_half[k],thetal[k] + theta_pert_,qt[k], Th)
                PV.values[ijk + s_varshift] = Th.entropy(RS.p0_half[k], T, qt[k], ql, 0.0)


    return

def thetal_mpace(p_, t_, ql_):
    return t_*(p_tilde/p_)**(Rd/cpd)*np.exp(-(2.26e6*ql_)/(cpd*263.0))

def sat_adjst(p_, thetal_, qt_, Th):

    """
    Use saturation adjustment scheme to compute temperature and ql given thetal and qt.
    :param p_: pressure [Pa]
    :param thetal_: liquid water potential temperature  [K]
    :param qt_:  total water specific humidity
    :return: t_2, ql_2
    """

    #Compute temperature
    t_1 = thetal_ * (p_/p_tilde)**(Rd/cpd)
    #Compute saturation vapor pressure
    pv_star_1 = Th.get_pv_star(t_1)
    #Compute saturation mixing ratio
    qs_1 = qv_star_c(p_,qt_,pv_star_1)

    if qt_ <= qs_1:
        #If not saturated return temperature and ql = 0.0
        return t_1, 0.0
    else:
        ql_1 = qt_ - qs_1
        f_1 = thetal_ - thetal_mpace(p_,t_1,ql_1)
        t_2 = t_1 + 2.26e6*ql_1/cpd
        pv_star_2 = Th.get_pv_star(t_2)
        qs_2 = qv_star_c(p_,qt_,pv_star_2)
        ql_2 = qt_ - qs_2

        while fabs(t_2 - t_1) >= 1e-9:
            pv_star_2 = Th.get_pv_star(t_2)
            qs_2 = qv_star_c(p_,qt_,pv_star_2)
            ql_2 = qt_ - qs_2
            f_2 = thetal_ - thetal_mpace(p_, t_2, ql_2)
            t_n = t_2 - f_2 * (t_2 - t_1)/(f_2 - f_1)
            t_1 = t_2
            t_2 = t_n
            f_1 = f_2

    return t_2, ql_2

