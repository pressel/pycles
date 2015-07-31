import numpy as np
cimport numpy as np
cimport ParallelMPI
cimport Grid
cimport PrognosticVariables
from thermodynamic_functions cimport exner_c
cimport ReferenceState
import time
import cython
from libc.math cimport sqrt, fmin, cos

def InitializationFactory(namelist):

        casename = namelist['meta']['casename']
        if casename == 'SullivanPatton':
            return InitSullivanPatton
        elif casename == 'StableBubble':
            return InitStableBubble
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


    for i in xrange(Gr.dims.nlg[0]):
        ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[u_varshift + ijk] = 0.0
                PV.values[v_varshift + ijk] = 0.0
                PV.values[w_varshift + ijk] = 0.0
                PV.values[qt_varshift + ijk] = 1e-9
                dist  = np.sqrt(((Gr.x_half[i + Gr.dims.indx_lo[0]]/1000.0 - 25.6)/4.0)**2.0 + ((Gr.z_half[k + Gr.dims.indx_lo[2]]/1000.0 - 3.0)/2.0)**2.0)
                dist = fmin(dist,1.0)
                t = (300.0)/exner_c(RS.p0_half[k]) - 15.0*( cos(np.pi * dist) + 1.0) /2.0
                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],t,PV.values[qt_varshift + ijk],0.0,0.0)

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
                t = (theta[k] + theta_pert_)/exner_c(RS.p0_half[k])

                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],t,0.0,0.0,0.0)



    return


