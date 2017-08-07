#!python
#cython: boundscheck=False
#cython: wraparound=True
#cython: initializedcheck=False
#cython: cdivision=True


import netCDF4 as nc
import numpy as np
cimport numpy as np
from scipy.interpolate import PchipInterpolator,pchip_interpolate
cimport ParallelMPI
from NetCDFIO cimport NetCDFIO_Stats
cimport Grid
cimport PrognosticVariables
cimport DiagnosticVariables
from thermodynamic_functions cimport exner_c, entropy_from_thetas_c, thetas_t_c, qv_star_c, thetas_c
cimport ReferenceState
from Forcing cimport AdjustedMoistAdiabat
from Thermodynamics cimport LatentHeat
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
        elif casename == 'DYCOMS_RF01':
            return InitDYCOMS_RF01
        elif casename == 'DYCOMS_RF02':
            return InitDYCOMS_RF02
        elif casename == 'SMOKE':
            return InitSmoke
        elif casename == 'Rico':
            return InitRico
        elif casename == 'CGILS':
            return  InitCGILS

        elif casename == 'ZGILS':
            return  InitZGILS

        else:
            pass

def InitStableBubble(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH):

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

def InitSaturatedBubble(namelist,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH ):

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

def InitSullivanPatton(namelist,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH ):

    #Generate the reference profiles
    RS.Pg = 1.0e5  #Pressure at ground
    RS.Tg = 300.0  #Temperature at ground
    RS.qtg = 0.0   #Total water mixing ratio at surface
    RS.u0 = 1.0  # velocities removed in Galilean transformation
    RS.v0 = 0.0

    RS.initialize(Gr, Th, NS, Pa)

    #Get the variable number for each of the velocity components
    np.random.seed(Pa.rank)
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift, e_varshift
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
    if 'e' in PV.name_index:
        e_varshift = PV.get_varshift(Gr, 'e')
        for i in xrange(Gr.dims.nlg[0]):
            ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
            for j in xrange(Gr.dims.nlg[1]):
                jshift = j * Gr.dims.nlg[2]
                for k in xrange(Gr.dims.nlg[2]):
                    ijk = ishift + jshift + k
                    PV.values[e_varshift + ijk] = 0.0
    return

def InitBomex(namelist,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH ):

    #First generate the reference profiles
    RS.Pg = 1.015e5  #Pressure at ground
    RS.Tg = 300.4  #Temperature at ground
    RS.qtg = 0.02245   #Total water mixing ratio at surface

    RS.initialize(Gr, Th, NS, Pa)

    try:
        random_seed_factor = namelist['initialization']['random_seed_factor']
    except:
        random_seed_factor = 1

    np.random.seed(Pa.rank * random_seed_factor)

    #Get the variable number for each of the velocity components

    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift
        Py_ssize_t ijk, e_varshift
        double temp
        double qt_
        double [:] thetal = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double [:] qt = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double [:] u = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        Py_ssize_t count

        theta_pert = (np.random.random_sample(Gr.dims.npg )-0.5)*0.1
        qt_pert = (np.random.random_sample(Gr.dims.npg )-0.5)*0.025/1000.0

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
                if Gr.zl_half[k] <= 1600.0:
                    temp = (thetal[k] + (theta_pert[count])) * exner_c(RS.p0_half[k])
                    qt_ = qt[k]+qt_pert[count]
                else:
                    temp = (thetal[k]) * exner_c(RS.p0_half[k])
                    qt_ = qt[k]
                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],temp,qt_,0.0,0.0)
                PV.values[qt_varshift + ijk] = qt_
                count += 1

    if 'e' in PV.name_index:
        e_varshift = PV.get_varshift(Gr, 'e')
        for i in xrange(Gr.dims.nlg[0]):
            ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
            for j in xrange(Gr.dims.nlg[1]):
                jshift = j * Gr.dims.nlg[2]
                for k in xrange(Gr.dims.nlg[2]):
                    ijk = ishift + jshift + k
                    PV.values[e_varshift + ijk] = 1.0-Gr.zl_half[k]/3000.0


    return

def InitGabls(namelist,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH ):

    #Generate the reference profiles
    RS.Pg = 1.0e5  #Pressure at ground
    RS.Tg = 265.0  #Temperature at ground
    RS.qtg = 0.0   #Total water mixing ratio at surface
    RS.u0 = 8.0  # velocities removed in Galilean transformation
    RS.v0 = 0.0

    RS.initialize(Gr, Th, NS, Pa)

    #Get the variable number for each of the velocity components
    np.random.seed(Pa.rank)
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift, e_varshift
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


    if 'e' in PV.name_index:
        e_varshift = PV.get_varshift(Gr, 'e')
        for i in xrange(Gr.dims.nlg[0]):
            ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
            for j in xrange(Gr.dims.nlg[1]):
                jshift = j * Gr.dims.nlg[2]
                for k in xrange(Gr.dims.nlg[2]):
                    ijk = ishift + jshift + k
                    if Gr.zl_half[k] <= 250.0:
                        PV.values[e_varshift + ijk] = 0.4*(1.0-Gr.zl_half[k]/250.0)**3.0
                    else:
                        PV.values[e_varshift + ijk] = 0.0


    return

def InitDYCOMS_RF01(namelist,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa , LatentHeat LH):

    '''
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
    '''

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
        Py_ssize_t e_varshift

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
    np.random.seed(Pa.rank)
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


    if 'e' in PV.name_index:
        e_varshift = PV.get_varshift(Gr, 'e')
        for i in xrange(Gr.dims.nlg[0]):
            ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
            for j in xrange(Gr.dims.nlg[1]):
                jshift = j * Gr.dims.nlg[2]
                for k in xrange(Gr.dims.nlg[2]):
                    ijk = ishift + jshift + k
                    if Gr.zl_half[k] < 200.0:
                        PV.values[e_varshift + ijk] = 0.0

    return



def InitDYCOMS_RF02(namelist,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH ):


    # Generate Reference Profiles
    RS.Pg = 1017.8 * 100.0
    RS.qtg = 9.0/1000.0
    RS.u0 = 5.0
    RS.v0 = -5.5
    cdef double cp_ref = 1004.0
    cdef double L_ref = 2.5e6

    # Use an exner function with values for Rd, and cp given in Stevens 2004 to compute temperature given $\theta_l$
    RS.Tg = 288.3 * (RS.Pg/p_tilde)**(287.0/cp_ref)

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
        double [:] u = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] v = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')

    for k in xrange(Gr.dims.nlg[2]):
        if Gr.zl_half[k] <=795.0:
            thetal[k] = 288.3
            qt[k] = 9.45/1000.0
        if Gr.zl_half[k] > 795.0:
            thetal[k] = 295.0 + (Gr.zl_half[k] - 795.0)**(1.0/3.0)
            qt[k] = (5.0 - 3.0 * (1.0 - np.exp(-(Gr.zl_half[k] - 795.0)/500.0)))/1000.0
        v[k] = -9.0 + 5.6 * Gr.zl_half[k]/1000.0 - RS.v0
        u[k] = 3.0 + 4.3*Gr.zl_half[k]/1000.0 - RS.u0

    def compute_thetal(p_,T_,ql_):
        theta_ = T_ / (p_/p_tilde)**(287.0/cp_ref)
        return theta_ * exp(-L_ref * ql_ / (cp_ref * T_))

    def sat_adjst(p_,thetal_,qt_):
        '''
        Use saturation adjustment scheme to compute temperature and ql given thetal and qt.
        :param p: pressure [Pa]
        :param thetal: liquid water potential temperature  [K]
        :param qt:  total water specific humidity
        :return: T, ql
        '''

        #Compute temperature
        t_1 = thetal_ * (p_/p_tilde)**(287.0/cp_ref)
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
            t_2 = t_1 + L_ref*ql_1/cp_ref
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
    np.random.seed(Pa.rank)
    cdef double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
    cdef double theta_pert_

    for i in xrange(Gr.dims.nlg[0]):
        ishift = istride * i
        for j in xrange(Gr.dims.nlg[1]):
            jshift = jstride * j
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[ijk + u_varshift] = u[k]
                PV.values[ijk + v_varshift] = v[k]
                PV.values[ijk + w_varshift] = 0.0
                PV.values[ijk + qt_varshift]  = qt[k]

                #Now set the entropy prognostic variable including a potential temperature perturbation
                if Gr.zl_half[k] < 795.0:
                    theta_pert_ = (theta_pert[ijk] - 0.5)* 0.1
                else:
                    theta_pert_ = 0.0
                T,ql = sat_adjst(RS.p0_half[k],thetal[k] + theta_pert_,qt[k])
                PV.values[ijk + s_varshift] = Th.entropy(RS.p0_half[k], T, qt[k], ql, 0.0)

    return

def InitSmoke(namelist,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH ):
    '''
    Initialization for the smoke cloud case
    Bretherton, C. S., and coauthors, 1999:
    An intercomparison of radiatively- driven entrainment and turbulence in a smoke cloud,
    as simulated by different numerical models. Quart. J. Roy. Meteor. Soc., 125, 391-423. Full text copy.
    :param Gr:
    :param PV:
    :param RS:
    :param Th:
    :param NS:
    :param Pa:
    :return:
    '''


    RS.Pg = 1000.0 * 100.0
    RS.qtg = 0.0
    RS.u0 = 0.0
    RS.v0 = 0.0
    RS.Tg = 288.0

    RS.initialize(Gr ,Th, NS, Pa)

    #Get the variable number for each of the velocity components
    np.random.seed(Pa.rank)
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr, 'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr, 'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr, 'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr, 's')
        Py_ssize_t smoke_varshift = PV.get_varshift(Gr, 'smoke')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift, e_varshift
        Py_ssize_t ijk
        double [:] theta = np.empty((Gr.dims.nlg[2]), dtype=np.double, order='c')
        double [:] smoke = np.empty((Gr.dims.nlg[2]), dtype=np.double, order='c')
        double t

        #Generate initial perturbations (here we are generating more than we need)
        cdef double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
        cdef double theta_pert_

    for k in xrange(Gr.dims.nlg[2]):
        if Gr.zl_half[k] <=  687.5:
            theta[k] = 288.0
            smoke[k] = 1.0
        elif Gr.zl_half[k] >= 687.5 and Gr.zl_half[k] <= 712.5:
            theta[k] = 288.0 + (Gr.zl_half[k] - 687.5) * 0.28
            smoke[k] = 1.0 - 0.04 * (Gr.zl_half[k] - 687.5)
            print k, Gr.zl_half[k], smoke[k]
        else:
            theta[k] = 295.0 + (Gr.zl_half[k] - 712.5) * 1e-4
            smoke[k] = 0.0

    cdef double [:] p0 = RS.p0_half

    #Now loop and set the initial condition
    for i in xrange(Gr.dims.nlg[0]):
        ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[u_varshift + ijk] = 0.0 - RS.u0
                PV.values[v_varshift + ijk] = 0.0 - RS.v0
                PV.values[w_varshift + ijk] = 0.0

                #Now set the entropy prognostic variable including a potential temperature perturbation
                if Gr.zl_half[k] < 700.0:
                    theta_pert_ = (theta_pert[ijk] - 0.5)* 0.1
                else:
                    theta_pert_ = 0.0
                t = (theta[k] + theta_pert_)*exner_c(RS.p0_half[k])

                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],t,0.0,0.0,0.0)
                PV.values[smoke_varshift + ijk] = smoke[k]

    if 'e' in PV.name_index:
        e_varshift = PV.get_varshift(Gr, 'e')
        for i in xrange(Gr.dims.nlg[0]):
            ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
            for j in xrange(Gr.dims.nlg[1]):
                jshift = j * Gr.dims.nlg[2]
                for k in xrange(Gr.dims.nlg[2]):
                    ijk = ishift + jshift + k
                    if Gr.zl_half[k] < 700.0:
                        PV.values[e_varshift + ijk] = 0.1
                    else:
                        PV.values[e_varshift + ijk] = 0.0

    return


def InitRico(namelist,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH ):

    #First generate the reference profiles
    RS.Pg = 1.0154e5  #Pressure at ground
    RS.Tg = 299.8  #Temperature at ground
    pvg = Th.get_pv_star(RS.Tg)
    RS.qtg = eps_v * pvg/(RS.Pg - pvg)   #Total water mixing ratio at surface = qsat

    RS.initialize(Gr, Th, NS, Pa)

    #Get the variable number for each of the velocity components
    np.random.seed(Pa.rank)
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift
        Py_ssize_t ijk, e_varshift
        double temp
        double qt_
        double [:] theta = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double [:] qt = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double [:] u = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double [:] v = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        Py_ssize_t count

        theta_pert = (np.random.random_sample(Gr.dims.npg )-0.5)*0.1
        qt_pert = (np.random.random_sample(Gr.dims.npg )-0.5) * 2.5e-5

    for k in xrange(Gr.dims.nlg[2]):

        #Set Thetal profile
        if Gr.zl_half[k] <= 740.0:
            theta[k] = 297.9
        else:
            theta[k] = 297.9 + (317.0-297.9)/(4000.0-740.0)*(Gr.zl_half[k] - 740.0)


        #Set qt profile
        if Gr.zl_half[k] <= 740.0:
            qt[k] =  16.0 + (13.8 - 16.0)/740.0 * Gr.zl_half[k]
        elif Gr.zl_half[k] > 740.0 and Gr.zl_half[k] <= 3260.0:
            qt[k] = 13.8 + (2.4 - 13.8)/(3260.0-740.0) * (Gr.zl_half[k] - 740.0)
        else:
            qt[k] = 2.4 + (1.8-2.4)/(4000.0-3260.0)*(Gr.zl_half[k] - 3260.0)


        #Change units to kg/kg
        qt[k]/= 1000.0

        #Set u profile
        u[k] = -9.9 + 2.0e-3 * Gr.zl_half[k]
        #set v profile
        v[k] = -3.8
    #Set velocities for Galilean transformation
    RS.v0 = -3.8
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
                PV.values[v_varshift + ijk] = v[k] - RS.v0
                PV.values[w_varshift + ijk] = 0.0
                if Gr.zl_half[k] <= 740.0:
                    temp = (theta[k] + (theta_pert[count])) * exner_c(RS.p0_half[k])
                    qt_ = qt[k]+qt_pert[count]
                else:
                    temp = (theta[k]) * exner_c(RS.p0_half[k])
                    qt_ = qt[k]
                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],temp,qt_,0.0,0.0)
                PV.values[qt_varshift + ijk] = qt_
                count += 1

    if 'e' in PV.name_index:
        e_varshift = PV.get_varshift(Gr, 'e')
        for i in xrange(Gr.dims.nlg[0]):
            ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
            for j in xrange(Gr.dims.nlg[1]):
                jshift = j * Gr.dims.nlg[2]
                for k in xrange(Gr.dims.nlg[2]):
                    ijk = ishift + jshift + k
                    if Gr.zl_half[k] <= 740.0:
                        PV.values[e_varshift + ijk] = 0.1


    return



def InitCGILS(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa , LatentHeat LH):
    #
    try:
        loc = namelist['meta']['CGILS']['location']
        if loc !=12 and loc != 11 and loc != 6:
            Pa.root_print('Invalid CGILS location (must be 6, 11, or 12)')
            Pa.kill()
    except:
        Pa.root_print('Must provide a CGILS location (6/11/12) in namelist')
        Pa.kill()
    try:
        is_p2 = namelist['meta']['CGILS']['P2']
    except:
        Pa.root_print('Must specify if CGILS run is perturbed')
        Pa.kill()


    if is_p2:
        file = './CGILSdata/p2k_s'+str(loc)+'.nc'
    else:
        file = './CGILSdata/ctl_s'+str(loc)+'.nc'

    data = nc.Dataset(file, 'r')
    # Get the profile information we need from the data file
    pressure_data = data.variables['lev'][::-1]
    temperature_data = data.variables['T'][0,::-1,0,0]
    q_data = data.variables['q'][0,::-1,0,0]
    u_data = data.variables['u'][0,::-1,0,0]
    v_data = data.variables['v'][0,::-1,0,0]

    for index in np.arange(len(q_data)):
        q_data[index] = q_data[index]/ (1.0 + q_data[index])




    # Get the surface information we need from the data file
    RS.Tg= data.variables['Tg'][0,0,0]
    RS.Pg= data.variables['Ps'][0,0,0]
    rh_srf = data.variables['rh_srf'][0,0,0]

    data.close()

    # Find the surface moisture and initialize the basic state
    pv_ = Th.get_pv_star(RS.Tg)*rh_srf
    RS.qtg =  eps_v * pv_ / (RS.Pg + (eps_v-1.0)*pv_)


    RS.initialize(Gr ,Th, NS, Pa)




    cdef:
        Py_ssize_t i, j, k, ijk, ishift, jshift
        Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
        Py_ssize_t jstride = Gr.dims.nlg[2]
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        double [:] thetal = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] qt = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] u = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] v = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double p_inversion = 940.0 * 100.0 # for S11, S12: pressure at the inversion
        double p_interp_les = 880 * 100.0 # for S11, S12:  pressure at which we start interpolating to the forcing profile
        double p_interp_data = 860 * 100.0 # for S11, S12: pressure at which we blend full to forcing profile

    #Set up profiles. First create thetal from the forcing data, to be used for interpolation
    thetal_data = np.zeros(np.shape(temperature_data))
    for k in xrange(len(thetal_data)):
        thetal_data[k] = temperature_data[k]/exner_c(pressure_data[k])


    # First we handle the S12 and S11 cases
    # This portion of the profiles is fitted to Figures #,# (S11) and #,# (S12) from Blossey et al
    if loc == 12:
        # Use a mixed layer profile
        if not is_p2:
            # CTL profiles
            for k in xrange(Gr.dims.nlg[2]):
                if RS.p0_half[k] > p_inversion:
                    thetal[k] = 288.35
                    qt[k] = 9.374/1000.0
                else:
                    thetal[k] = (3.50361862e+02 +  -5.11283538e-02 * RS.p0_half[k]/100.0)
                    qt[k] = 3.46/1000.0
        else:
            # P2K
            for k in xrange(Gr.dims.nlg[2]):
                if RS.p0_half[k] > p_inversion:
                    thetal[k] = 290.35
                    qt[k] = 11.64/1000.0
                else:
                    thetal[k] = (3.55021347e+02 +  -5.37703211e-02 * RS.p0_half[k]/100.0)
                    qt[k] = 4.28/1000.0
    elif loc == 11:
        if not is_p2:
            for k in xrange(Gr.dims.nlg[2]):
                if RS.p0_half[k] > 935.0*100.0:
                    thetal[k] = 289.6
                    qt[k] = 10.25/1000.0
                else:
                    thetal[k] = (3.47949119e+02 +  -5.02475698e-02 * RS.p0_half[k]/100.0)
                    qt[k] = 3.77/1000.0
        else:
            # P2 parameters
            for k in xrange(Gr.dims.nlg[2]):
                if RS.p0_half[k] > 935.0*100.0:
                    thetal[k] = 291.6
                    qt[k] =11.64/1000.0
                else:
                    thetal[k] = (3.56173912e+02 +  -5.70945946e-02 * RS.p0_half[k]/100.0)
                    qt[k] = 4.28/1000.0


    # Set up for interpolation to forcing profiles
    if loc == 11 or loc == 12:
        pressure_interp = np.empty(0)
        thetal_interp = np.empty(0)
        qt_interp = np.empty(0)
        for k in xrange(Gr.dims.gw,Gr.dims.nlg[2]-Gr.dims.gw):
            if RS.p0_half[k] > p_interp_les:
                pressure_interp = np.append(pressure_interp,RS.p0_half[k])
                thetal_interp = np.append(thetal_interp,thetal[k])
                # qt_interp = np.append(qt_interp, qt[k]/(1.0+qt[k]))
                qt_interp = np.append(qt_interp, qt[k])

        pressure_interp = np.append(pressure_interp, pressure_data[pressure_data<p_interp_data] )
        thetal_interp = np.append(thetal_interp, thetal_data[pressure_data<p_interp_data] )
        qt_interp = np.append(qt_interp, q_data[pressure_data<p_interp_data] )

        # Reverse the arrays so pressure is increasing
        pressure_interp = pressure_interp[::-1]
        thetal_interp = thetal_interp[::-1]
        qt_interp = qt_interp[::-1]

    else:
        # for S6 case, interpolate ALL values
        p_interp_les = RS.Pg
        pressure_interp = pressure_data[::-1]
        thetal_interp = thetal_data[::-1]
        qt_interp = q_data[::-1]

    # PCHIP interpolation helps to make the S11 and S12 thermodynamic profiles nice, but the scipy pchip interpolator
    # does not handle extrapolation kindly. Thus we tack on our own linear extrapolation to deal with the S6 case
    # We also use linear extrapolation to handle the velocity profiles, which it seems are fine to interpolate linearly

    thetal_right = thetal_interp[-1] + (thetal_interp[-2] - thetal_interp[-1])/(pressure_interp[-2]-pressure_interp[-1]) \
                                                 * ( RS.Pg-pressure_interp[-1])
    thetal_interp = np.append(thetal_interp, thetal_right)
    qt_right = qt_interp[-1] + (qt_interp[-2] - qt_interp[-1])/(pressure_interp[-2]-pressure_interp[-1]) \
                                                 * ( RS.Pg-pressure_interp[-1])
    qt_interp = np.append(qt_interp, qt_right)
    pressure_interp = np.append(pressure_interp, RS.Pg)



    # Now do the interpolation
    for k in xrange(Gr.dims.nlg[2]):
            if RS.p0_half[k] <= p_interp_les:

                # thetal_right = thetal_interp[-1] + (thetal_interp[-2] - thetal_interp[-1])/(pressure_interp[-2]-pressure_interp[-1]) \
                #                                  * ( RS.p0_half[k]-pressure_interp[-1])
                # qt_right = qt_interp[-1] + (qt_interp[-2] - qt_interp[-1])/(pressure_interp[-2]-pressure_interp[-1]) \
                #                                  * ( RS.p0_half[k]-pressure_interp[-1])

                # thetal[k] = np.interp(RS.p0_half[k], pressure_interp, thetal_interp, right = thetal_right)
                # qt[k] = np.interp(RS.p0_half[k],pressure_interp,qt_interp, right=qt_right)
                thetal[k] = pchip_interpolate(pressure_interp, thetal_interp, RS.p0_half[k])
                qt[k] = pchip_interpolate(pressure_interp, qt_interp, RS.p0_half[k])
            # Interpolate entire velocity profiles
            u_right = u_data[0] + (u_data[1] - u_data[0])/(pressure_data[1]-pressure_data[0]) * ( RS.p0_half[k]-pressure_data[0])
            v_right = v_data[0] + (v_data[1] - v_data[0])/(pressure_data[1]-pressure_data[0]) * ( RS.p0_half[k]-pressure_data[0])

            u[k] = np.interp(RS.p0_half[k],pressure_data[::-1], u_data[::-1], right=u_right)
            v[k] = np.interp(RS.p0_half[k],pressure_data[::-1], v_data[::-1],right=v_right)
    #Set velocities for Galilean transformation
    RS.u0 = 0.5 * (np.amax(u)+np.amin(u))
    RS.v0 = 0.5 * (np.amax(v)+np.amin(v))


    # We will need these functions to perform saturation adjustment
    def compute_thetal(p_,T_,ql_):
        theta_ = T_ / (p_/p_tilde)**kappa
        return theta_ * exp(-2.501e6 * ql_ / (cpd* T_))

    def sat_adjst(p_,thetal_,qt_):
        '''
        Use saturation adjustment scheme to compute temperature and ql given thetal and qt.
        :param p: pressure [Pa]
        :param thetal: liquid water potential temperature  [K]
        :param qt:  total water specific humidity
        :return: T, ql
        '''

        #Compute temperature
        t_1 = thetal_ * (p_/p_tilde)**kappa
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
            t_2 = t_1 + 2.501e6*ql_1/cpd
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
    np.random.seed(Pa.rank)
    cdef double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
    cdef double theta_pert_

    # Here we fill in the 3D arrays
    # We perform saturation adjustment on the S6 data, although this should not actually be necessary (but doesn't hurt)
    for i in xrange(Gr.dims.nlg[0]):
        ishift = istride * i
        for j in xrange(Gr.dims.nlg[1]):
            jshift = jstride * j
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[ijk + u_varshift] = u[k] - RS.u0
                PV.values[ijk + v_varshift] = v[k] - RS.v0
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






def InitZGILS(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa , LatentHeat LH):

    reference_profiles = AdjustedMoistAdiabat(namelist, LH, Pa)

    RS.Tg= 289.472
    RS.Pg= 1018.0e2
    RS.qtg = 0.008449

    RS.initialize(Gr ,Th, NS, Pa)


    cdef double Pg_parcel = 1000.0e2
    cdef double Tg_parcel = 295.0
    cdef double RH_ref = 0.3
    reference_profiles.initialize(Pa, RS.p0_half[:], Gr.dims.nlg[2],Pg_parcel, Tg_parcel, RH_ref)



    cdef:
        Py_ssize_t i, j, k, ijk, ishift, jshift
        Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
        Py_ssize_t jstride = Gr.dims.nlg[2]
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        double [:] thetal = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] qt = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] u = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] v = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')

    for k in xrange(Gr.dims.nlg[2]):
        if RS.p0_half[k]  > 920.0e2:
            thetal[k] = RS.Tg /exner_c(RS.Pg)
            qt[k] = RS.qtg
        u[k] = min(-10.0 + (-7.0-(-10.0))/(750.0e2-1000.0e2)*(RS.p0_half[k]-1000.0e2),-4.0)


      #Set velocities for Galilean transformation
    RS.u0 = 0.5 * (np.amax(u)+np.amin(u))
    RS.v0 = 0.5 * (np.amax(v)+np.amin(v))


    # We will need these functions to perform saturation adjustment
    def compute_thetal(p_,T_,ql_):
        theta_ = T_ / (p_/p_tilde)**kappa
        return theta_ * exp(-2.501e6 * ql_ / (cpd* T_))

    def sat_adjst(p_,thetal_,qt_):


        #Compute temperature
        t_1 = thetal_ * (p_/p_tilde)**kappa
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
            t_2 = t_1 + 2.501e6*ql_1/cpd
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
    np.random.seed(Pa.rank)
    cdef double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
    cdef double theta_pert_

    # Here we fill in the 3D arrays
    # We perform saturation adjustment on the S6 data, although this should not actually be necessary (but doesn't hurt)
    for i in xrange(Gr.dims.nlg[0]):
        ishift = istride * i
        for j in xrange(Gr.dims.nlg[1]):
            jshift = jstride * j
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[ijk + u_varshift] = u[k] - RS.u0
                PV.values[ijk + v_varshift] = v[k] - RS.v0
                PV.values[ijk + w_varshift] = 0.0
                if RS.p0_half[k] > 920.0e2:
                    PV.values[ijk + qt_varshift]  = qt[k]
                    theta_pert_ = (theta_pert[ijk] - 0.5)* 0.1
                    T,ql = sat_adjst(RS.p0_half[k],thetal[k] + theta_pert_,qt[k])
                    PV.values[ijk + s_varshift] = Th.entropy(RS.p0_half[k], T, qt[k], ql, 0.0)
                else:
                    PV.values[ijk + qt_varshift]  = reference_profiles.qt[k]
                    PV.values[ijk + s_varshift] = reference_profiles.s[k]


    return





def AuxillaryVariables(nml, PrognosticVariables.PrognosticVariables PV,
                       DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):

    casename = nml['meta']['casename']
    if casename == 'SMOKE':
        PV.add_variable('smoke', 'kg/kg', 'smoke', 'radiatively active smoke', "sym", "scalar", Pa)
        return
    return
