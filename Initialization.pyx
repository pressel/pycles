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
from thermodynamic_functions cimport exner_c, entropy_from_thetas_c, thetas_t_c, qv_star_c, thetas_c, thetali_c
cimport ReferenceState
from Forcing cimport AdjustedMoistAdiabat
from Thermodynamics cimport LatentHeat
from libc.math cimport sqrt, fmin, cos, exp, fabs
include 'parameters.pxi'
# import matplotlib.pyplot as plt



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
        elif casename == 'Isdac':
            return InitIsdac
        elif casename == 'IsdacCC':
            return InitIsdacCC
        elif casename == 'Mpace':
            return InitMpace
        elif casename == 'Sheba':
            return InitSheba
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

    #Get shape of the initial bubble    
    try:
        namelist['straka']
        namelist['straka']['amplitude']
        namelist['straka']['x_r']
        namelist['straka']['z_r']
        namelist['straka']['z_c']
        straka_kwargs = namelist['straka']
        print('Using specified initial condition.')
    except: # default values
        straka_kwargs = {}
        straka_kwargs['amplitude'] = 15.0 # [C]
        straka_kwargs['x_r'] = 4.0 # [km]
        straka_kwargs['z_r'] = 2.0 # [km]
        straka_kwargs['z_c'] = 3.0 # [km]
        print('Using default initial condition.')

    #Get initial condition perturbation seed
    try:
        random_seed_factor = namelist['straka']['random_seed_factor']
    except:
        random_seed_factor = 1
    np.random.seed(Pa.rank * random_seed_factor)

    #Get initial condition perturbation size
    try:
        IC_max_pert_ = namelist['straka']['IC_max_pert']
    except:
        IC_max_pert_ = 0

    cdef:
        #Get the variable number for each of the velocity components
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift
        Py_ssize_t ijk
        double t
        double dist

        #Get shape of the initial perturbation
        double a =  float(straka_kwargs['amplitude'])
        double x_r = float(straka_kwargs['x_r'])
        double z_r = float(straka_kwargs['z_r'])
        double z_c = float(straka_kwargs['z_c'])
        double x_c = 25.6 # symmetric

        #Prepare initial temperature perturbations: U( [-IC_max_pert, IC_max_pert) )
        cdef double [:] t_pert = (np.random.random_sample(Gr.dims.npg) - 0.5)*2
        cdef double t_pert_
        double IC_max_pert = float(IC_max_pert_)

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
                dist  = np.sqrt( ((Gr.x_half[i + Gr.dims.indx_lo[0]]/1000.0 - x_c)/x_r)**2.0 + ((Gr.z_half[k + Gr.dims.indx_lo[2]]/1000.0 - z_c)/z_r)**2.0 )
                dist = fmin(dist,1.0)

                #Add initial perturbation
                t_pert_ = t_pert[ijk] * IC_max_pert
                t = (300.0 + t_pert_)*exner_c(RS.p0_half[k]) - a*( cos(np.pi * dist) + 1.0) /2.0

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
            print(k, Gr.zl_half[k], smoke[k])
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


def InitIsdac(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH):

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

        #Set v profile
        v[k] = -2.0 + 0.003 * Gr.zl_half[k]

    #Set velocities for Galilean transformation
    RS.u0 = -7.0
    RS.v0 = 0.5 * (np.amax(v)+np.amin(v))

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
                PV.values[ijk + u_varshift] = -7.0 - RS.u0
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

def InitIsdacCC(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH):

    '''
    Idealized ISDAC setup initialization based on the ISDAC case described in Ovchinnikov et al. (2014):
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
    RS.Tg = namelist['initial']['SST'] + namelist['initial']['dSST']
    pv_sat = Th.get_pv_star(RS.Tg)
    pv = pv_sat * namelist['initial']['rh0']
    RS.qtg =  1.0/(eps_vi * (RS.Pg - pv) / pv + 1.0)  #Total water mixing ratio at surface

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
        double [:] v = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] rh = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')

        #Specify initial condition variables
        double thetal_inv = namelist['initial']['dTi'] #inversion jump at cloud top
        double thetal_gamma = namelist['initial']['gamma'] #lapse rate above cloud top
        double rh_tropo = namelist['initial']['rh'] #lower tropospheric relative humidity
        double z_top = namelist['initial']['z_top'] #cloud top height
        double dz_inv = namelist['initial']['dzi'] #inversion depth
        bint fix_dqt = namelist['initial']['fix_dqt'] #Whether dqt is fixed
        double dqt_baseline = -0.000454106424679 #Value for the reference climate
        double [:] temp = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] p0_half = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double t_above_inv, p_above_inv


    RS.ic_qt = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
    RS.ic_thetal = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
    RS.ic_rh = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')

    for k in xrange(Gr.dims.nlg[2]):
        #Set thetal and qt profile
        if Gr.zl_half[k] <= z_top:
            RS.ic_thetal[k] = RS.Tg
        if z_top < Gr.zl_half[k] <= (z_top + dz_inv):
            RS.ic_thetal[k] = RS.Tg + (Gr.zl_half[k] - z_top) * thetal_inv / dz_inv
        if Gr.zl_half[k] > (z_top + dz_inv):
            RS.ic_thetal[k] = RS.ic_thetal[k-1] + Gr.dims.dx[2] * thetal_gamma
            temp[k] = RS.ic_thetal[k] * (RS.p0_half[k] / p_tilde)**(Rd/cpd)
            pv_sat = Th.get_pv_star(temp[k])
            pv = pv_sat * rh_tropo
            RS.ic_qt[k] = qv_unsat(RS.p0_half[k], pv)
            p0_half[k] = RS.p0_half[k]
            rh[k] = rh_tropo


    cdef double qt_above_inv = np.amax(RS.ic_qt)

    #If to fix dqt above the cloud top to be the baseline value, calculate the qt values above cloud top here
    if fix_dqt:
        qt_above_inv = RS.qtg + dqt_baseline
        t_above_inv = np.amax(temp)
        p_above_inv = np.amax(p0_half)
        pv_sat = Th.get_pv_star(t_above_inv)
        pv = (p_above_inv * qt_above_inv) / (eps_v * (1.0 - qt_above_inv) + qt_above_inv)
        rh_tropo = pv / pv_sat
        Pa.root_print(rh_tropo)

        for k in xrange(Gr.dims.nlg[2]):
            if Gr.zl_half[k] > (z_top + dz_inv):
                pv_sat = Th.get_pv_star(temp[k])
                pv = pv_sat * rh_tropo
                RS.ic_qt[k] = qv_unsat(RS.p0_half[k], pv)
                rh[k] = rh_tropo

    for k in xrange(Gr.dims.nlg[2]):
        if Gr.zl_half[k] <= z_top:
            RS.ic_qt[k] = RS.qtg

    for k in xrange(Gr.dims.nlg[2]):
        if z_top < Gr.zl_half[k] <= (z_top + dz_inv):
            RS.ic_qt[k] = RS.qtg - (Gr.zl_half[k] - z_top) * (RS.qtg - qt_above_inv) / dz_inv

    for k in xrange(Gr.dims.nlg[2]):
        if Gr.zl_half[k] <= (z_top + dz_inv):
            T, ql = sat_adjst(RS.p0_half[k], RS.ic_thetal[k], RS.ic_qt[k], Th)
            pv_sat = Th.get_pv_star(T)
            qv = RS.ic_qt[k] - ql
            pv = (RS.p0_half[k] * qv) / (eps_v * (1.0 - RS.ic_qt[k]) + qv)
            rh[k] = pv / pv_sat


    for k in xrange(Gr.dims.nlg[2]):
        #Set u profile
        v[k] = -2.0 + 0.003 * Gr.zl_half[k]

    #Set velocities for Galilean transformation
    RS.u0 = -7.0
    RS.v0 = 0.5 * (np.amax(v)+np.amin(v))

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
                PV.values[ijk + u_varshift] = -7.0 - RS.u0
                PV.values[ijk + v_varshift] = v[k] - RS.v0
                PV.values[ijk + w_varshift] = 0.0
                PV.values[ijk + qt_varshift]  = RS.ic_qt[k]

                #Now set the entropy prognostic variable including a potential temperature perturbation
                if Gr.zl_half[k] < 825.0:
                    theta_pert_ = (theta_pert[ijk] - 0.5)* 0.1
                else:
                    theta_pert_ = 0.0
                T,ql = sat_adjst(RS.p0_half[k],RS.ic_thetal[k] + theta_pert_, RS.ic_qt[k], Th)
                PV.values[ijk + s_varshift] = Th.entropy(RS.p0_half[k], T, RS.ic_qt[k], ql, 0.0)


    return

def InitMpace(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH):

    #First generate the reference profiles
    RS.Pg = 1.01e5 #Surface pressure
    RS.Tg = 274.01 #Sea surface temperature
    pvg = Th.get_pv_star(RS.Tg) #Saturation vapor pressure
    wtg = eps_v * pvg/(RS.Pg - pvg) #Saturation mixing ratio
    RS.qtg = wtg/(1.0+wtg) #Saturation specific humidity

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
        double [:] wt = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')

    for k in xrange(Gr.dims.nlg[2]):

        #Set thetal and qt profile
        if RS.p0_half[k] > 85000.0:
            thetal[k] = 269.2
            wt[k] = 1.95 #Mixing ratio in g/kg
        else:
            thetal[k] = 275.33 + 0.0791 * (815.0 - RS.p0_half[k]/100.0)
            wt[k] = 0.291 + 0.00204 * (RS.p0_half[k]/100.0 - 590.0)

        #Convert mixing ratio to specific humidity
        qt[k] = wt[k]/(1.0 + wt[k]/1000.0)/1000.0

    #Horizontal wind
    RS.u0 = -13.0
    RS.v0 = -3.0

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
                PV.values[ijk + u_varshift] = -13.0 - RS.u0
                PV.values[ijk + v_varshift] = -3.0 - RS.v0
                PV.values[ijk + w_varshift] = 0.0
                PV.values[ijk + qt_varshift]  = qt[k]

                #Now set the entropy prognostic variable including a potential temperature perturbation
                if RS.p0_half[k] > 85000.0:
                    theta_pert_ = (theta_pert[ijk] - 0.5)* 0.1
                else:
                    theta_pert_ = 0.0
                T,ql = sat_adjst(RS.p0_half[k],thetal[k] + theta_pert_,qt[k], Th)
                PV.values[ijk + s_varshift] = Th.entropy(RS.p0_half[k], T, qt[k], ql, 0.0)

    return

def InitSheba(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH):

    #First generate the reference profiles
    RS.Pg = 1.017e5 #Surface pressure
    RS.Tg = 257.4 #Sea surface temperature (ice-covered)
    pvg = Th.get_pv_star(RS.Tg) #Saturation vapor pressure
    wtg = eps_v * pvg/(RS.Pg - pvg) #Saturation mixing ratio
    RS.qtg = wtg/(1.0+wtg)*0.99 #Saturation specific humidity

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
        double [:] wt = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double p_inv = 95700.0
        double [:] ps
        double [:] us
        double [:] vs
        double [:] v = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] u = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')

    for k in xrange(Gr.dims.nlg[2]):

        #Set thetal and qt profile
        if RS.p0_half[k] > p_inv:
            thetal[k] = 255.7 #257.0
            wt[k] = 0.91 #0.915 #Mixing ratio in g/kg
        else:
            thetal[k] = 255.7+6.9 #263.9
            wt[k] = 0.8

    for k in xrange(Gr.dims.nlg[2]-1):
        if RS.p0_half[k] < p_inv:
            thetal[k+1] = thetal[k] + ((1.0/exner_c(RS.p0_half[k+1]))*fmin(3.631e-8*(p_inv - RS.p0_half[k+1]), 5.7e-4))*(RS.p0_half[k]-RS.p0_half[k+1])
            wt[k+1] = wt[k] - 1.4e-5*(RS.p0_half[k]-RS.p0_half[k+1])

        #Convert mixing ratio to specific humidity
        qt[k] = wt[k]/(1.0 + wt[k]/1000.0)/1000.0

    #Horizontal wind profiles from Colleen's JPLLES code

    ps = np.array([1017.0,1012.0,1007.0,1002.0,997.0,992.0,987.0,982.0,977.0,972.0,967.0,962.0,957.01,957.0,
                             956.99,952.0,947.0,942.0,937.0,932.0,927.0,922.0,917.0,912.0,907.0,902.0,897.0,892.0,
                             887.0,882.0,877.0,872.0,867.0,862.0,857.0,852.0,847.0,842.0,837.0,832.0,827.0,822.0,817.0,
                             812.0,807.0,802.0,797.0,792.0,787.0,782.0,777.0,772.0,767.0,762.0,757.0,752.0,747.0,742.0,
                             737.0,732.0,727.0,722.0,717.0,712.0,707.0,702.0,697.0,692.0,687.0,682.0,677.0,672.0,667.0,
                             662.0,657.0,652.0,647.0,642.0,637.0,632.0,627.0,622.0,617.0,612.0,607.0],dtype=np.double,order='c')*100.0

    us = np.array([2.916,2.8999,2.7895,2.679,2.5568,2.417,2.2772,2.1374,1.9976,1.8856,1.7926,1.6996,1.6066,1.6066,
                   1.6066,1.5136,1.4205,1.3248,1.2203,1.1159,1.0114,0.90701,0.80257,0.69813,0.59368,0.48564,0.37645,
                   0.26726,0.15807,0.048881,-0.06031,-0.1695,-0.27869,-0.42739,-0.61744,-0.80748,-0.99753,-1.1876,
                   -1.3776,-1.5677,-1.7577,-1.9477,-2.0576,-2.1326,-2.2076,-2.2826,-2.3576,-2.4326,-2.5076,-2.5826,
                   -2.6576,-2.6911,-2.6961,-2.701,-2.7059,-2.7109,-2.7158,-2.7207,-2.7257,-2.7306,-2.7137,-2.653,
                   -2.5923,-2.5316,-2.4708,-2.4101,-2.3494,-2.2886,-2.2279,-2.1662,-2.08,-1.9938,-1.9076,-1.8214,
                   -1.7352,-1.649,-1.5629,-1.4767,-1.3905,-1.302,-1.213,-1.124,-1.035,-0.94601,-0.85701],dtype=np.double,order='c')

    vs = np.array([2.8497,2.9023,3.2622,3.6221,3.8701,3.9526,4.0351,4.1177,4.2002,4.2743,4.3427,4.4112,4.4796,4.4796,
                   4.4796,4.548,4.6165,4.6831,4.744,4.8048,4.8657,4.9266,4.9874,5.0483,5.1092,5.1671,5.2241,5.2811,
                   5.3381,5.3951,5.4521,5.5091,5.5662,5.6245,5.6844,5.7442,5.8041,5.8639,5.9238,5.9836,6.0434,6.1033,
                   6.1444,6.1774,6.2105,6.2435,6.2765,6.3096,6.3426,6.3756,6.4086,6.4722,6.5567,6.6412,6.7258,6.8103,
                   6.8948,6.9794,7.0639,7.1484,7.2388,7.3407,7.4426,7.5446,7.6465,7.7485,7.8504,7.9524,8.0543,8.1574,
                   8.2893,8.4211,8.553,8.6848,8.8167,8.9485,9.0804,9.2122,9.3441,9.493,9.6463,9.7995,9.9527,10.106,
                   10.259],dtype=np.double,order='c')

    #Interpolate to LES grid
    for k in xrange(Gr.dims.nlg[2]):
        u[k] = interp_pchip(RS.p0[k], ps[::-1], us[::-1])
        v[k] = interp_pchip(RS.p0[k], ps[::-1], vs[::-1])


    RS.u0 = 0.5 * (np.amax(u)+np.amin(u))
    RS.v0 = 0.5 * (np.amax(v)+np.amin(v))

    #Generate initial perturbations (here we are generating more than we need)
    cdef double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
    cdef double theta_pert_
    cdef double T, ql

    #Now loop and set the initial condition
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
                if RS.p0_half[k] > p_inv:
                    theta_pert_ = (theta_pert[ijk] - 0.5)* 0.1
                else:
                    theta_pert_ = 0.0
                T,ql = sat_adjst(RS.p0_half[k],thetal[k] + theta_pert_,qt[k], Th)
                PV.values[ijk + s_varshift] = Th.entropy(RS.p0_half[k], T, qt[k], ql, 0.0)

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


def thetal_mpace(p_, t_, ql_):
    return t_*(p_tilde/p_)**(Rd/cpd)*np.exp(-(2.26e6*ql_)/(cpd*263.0))

def thetal_isdac(p_, t_, ql_, qt_):
    rl_ = ql_ / (1 - qt_)
    #return (p_tilde/p_)**(Rd/cpd)*(t_ - 2.26e6 * rl_ / cpd)
    return t_*(p_tilde/p_)**(Rd/cpd)*np.exp(-(rl_*2.501e6) / (t_*cpd))

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
        # f_1 = thetal_ - thetal_mpace(p_,t_1,ql_1)
        f_1 = thetal_ - thetal_isdac(p_,t_1,ql_1,qt_)
        t_2 = t_1 + 2.501e6*ql_1/cpd
        pv_star_2 = Th.get_pv_star(t_2)
        qs_2 = qv_star_c(p_,qt_,pv_star_2)
        ql_2 = qt_ - qs_2

        while fabs(t_2 - t_1) >= 1e-9:
            pv_star_2 = Th.get_pv_star(t_2)
            qs_2 = qv_star_c(p_,qt_,pv_star_2)
            ql_2 = qt_ - qs_2
            # f_2 = thetal_ - thetal_mpace(p_, t_2, ql_2)
            f_2 = thetal_ - thetal_isdac(p_, t_2, ql_2, qt_)
            t_n = t_2 - f_2 * (t_2 - t_1)/(f_2 - f_1)
            t_1 = t_2
            t_2 = t_n
            f_1 = f_2

    return t_2, ql_2

def qv_star_rh(p0, rh, pv):
    val = eps_v*pv/(p0-pv)/(1 + rh*eps_v*pv/(p0-pv))
    return val

def qv_unsat(p0, pv):
    val = 1.0/(eps_vi * (p0 - pv)/pv + 1.0)
    return val

from scipy.interpolate import pchip
def interp_pchip(z_out, z_in, v_in, pchip_type=True):
    if pchip_type:
        p = pchip(z_in, v_in, extrapolate=True)
        return p(z_out)
    else:
        return np.interp(z_out, z_in, v_in)
