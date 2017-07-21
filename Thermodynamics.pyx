#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport ParallelMPI
from ThermodynamicsDry cimport ThermodynamicsDry
from ThermodynamicsSA cimport ThermodynamicsSA
from Thermodynamics cimport LatentHeat
import numpy as np
cimport numpy as np
cimport Lookup
cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
from NetCDFIO cimport NetCDFIO_Fields, NetCDFIO_Stats


from scipy.integrate import odeint

include 'parameters.pxi'

cdef class LatentHeat:
    def __init__(self,namelist,ParallelMPI.ParallelMPI Par):

        return

    cpdef L(self,double T, double Lambda):
        '''
        Provide a python interface to the latent heat function pointer.
        :param T (Thermodynamic Temperature):
        :return L (Latent Heat):
        '''
        return self.L_fp(T, Lambda)

    cpdef Lambda(self, double T):
        return self.Lambda_fp(T)

cdef class ClausiusClapeyron:
    def __init__(self):
        return

    def initialize(self, namelist, LatentHeat LH, ParallelMPI.ParallelMPI Par):
        self.LT = Lookup.Lookup()

        #Now integrate the ClausiusClapeyron equation
        cdef:
            double Tmin
            double Tmax
            long n_lookup
            double [:] pv

        try:
            Tmin = namelist['ClausiusClapeyron']['temperature_min']
        except:
            Par.root_print('Clasius-Clayperon lookup table temperature_min not '
                           'given in name list taking default of 180 K')
            Tmin = 1.0

        try:
            Tmax = namelist['ClausiusClapeyron']['temperature_max']
        except:
            Par.root_print('Clasius-Clayperon lookup table temperature_max not '
                           'given in name list taking default of 380.0 K')
            Tmax = 380.0

        try:
            n_lookup = namelist['ClausiusClapeyron']['n_lookup']
        except:
            Par.root_print('Clasius-Clayperon lookup table n_lookup not '
                           'given in name list taking default of 512')
            n_lookup = 512

        #Generate array of equally space temperatures
        T = np.linspace(Tmin, Tmax, n_lookup)

        #Find the maximum index of T where T < T_tilde
        tp_close_index = np.max(np.where(T<=Tt))

        #Check to make sure that T_tilde is not in T
        if T[tp_close_index] == Tt:
            Par.root_print('Array of temperatures for ClasiusClapyeron lookup table contains Tt  \n')
            Par.root_print('Pick different values for ClasiusClapyeron Tmin and Tmax in lookup table \n')
            Par.root_print('Killing Simulation now!')
            Par.kill()

        #Now prepare integration
        T_above_Tt= np.append([Tt],T[tp_close_index+1:])
        T_below_Tt= np.append(T[:tp_close_index+1],[Tt])[::-1]

        #Now set up the RHS
        def rhs(z,T_):
            lam = LH.Lambda(T_)
            L = LH.L(T_,lam)
            return L/(Rv * T_ * T_)

        #set the initial condition
        pv0 = np.log(pv_star_t)

        #Integrate
        pv_above_Tt = np.exp(odeint(rhs,pv0,T_above_Tt,hmax=0.1)[1:])
        pv_below_Tt = np.exp(odeint(rhs,pv0,T_below_Tt,hmax=0.1)[1:])[::-1]
        pv = np.append(pv_below_Tt,pv_above_Tt )

        #For really small values of pv, set pv to a slightly less small number. This avoids problems in integrating
        #the reference profiles, when the reference temperature is <100K. For the vast majority of simulations this
        #modification should have no impact.

        pv_a = np.array(pv)
        pv_a[pv_a < 1e-11]= 1e-11

        self.LT.initialize(T,pv_a)

        return

    cpdef finalize(self):
        self.LT.finalize()
        return


def ThermodynamicsFactory(namelist, Micro, LatentHeat LH,ParallelMPI.ParallelMPI Par):

    if(Micro.thermodynamics_type=='dry'):
        return ThermodynamicsDry(namelist,LH,Par)
    if(Micro.thermodynamics_type=='SA'):
        return ThermodynamicsSA(namelist,LH,Par)

