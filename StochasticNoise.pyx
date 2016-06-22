#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport PrognosticVariables
cimport ParallelMPI
cimport ReferenceState
# from NetCDFIO cimport NetCDFIO_Stats
from thermodynamic_functions cimport exner_c

import numpy as np
cimport numpy as np
import pylab as plt

cdef class StochasticNoise:
    def __init__(self, namelist):
        try:
            self.stoch_noise = namelist['stochastic_noise']['flag']
        except:
            return

        try:
            self.ampl = namelist['stochastic_noise']['amplitude']
        except:
            self.ampl = 0.01

        return



    cpdef initialize(self,ParallelMPI.ParallelMPI Pa):

        if self.stoch_noise == True:
            Pa.root_print('Stochastic Noise activated in every timestep, A = ' + str(self.ampl))


        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,  Th, ParallelMPI.ParallelMPI Pa):

        if self.stoch_noise:
            self.add_theta_noise(Gr, RS, PV, Th, Pa)

        return



    cpdef add_theta_noise(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,  Th, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t i,j,k
            Py_ssize_t ishift, jshift
            Py_ssize_t ijk
            Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')


        np.random.seed(Pa.rank)
        cdef:
            double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
            double theta_pert_
            double t
            double [:,:,:] pert = np.zeros((Gr.dims.nlg[0],Gr.dims.nlg[1],Gr.dims.nlg[2]),dtype=np.double,order='C')
            double [:,:] pert_2d = np.zeros((Gr.dims.nlg[0],Gr.dims.nlg[2]),dtype=np.double,order='C')



        #Now loop and set the initial condition
        for i in xrange(Gr.dims.nlg[0]):
            ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
            for j in xrange(Gr.dims.nlg[1]):
                jshift = j * Gr.dims.nlg[2]
                for k in xrange(Gr.dims.nlg[2]):
                    ijk = ishift + jshift + k
                    # potential temperature perturbation
                    if Gr.zl_half[k] < 200.0:
                        theta_pert_ = (theta_pert[ijk] - 0.5)* self.ampl
                    else:
                        theta_pert_ = 0.0
                    t = theta_pert_*exner_c(RS.p0_half[k])

                    # PV.tendencies[s_varshift + ijk] += Th.entropy(RS.p0_half[k],t,0.0,0.0,0.0)

                    # pert[i,j,k] = Th.entropy(RS.p0_half[k],t,0.0,0.0,0.0)
                    pert_2d[i,k] = Th.entropy(RS.p0_half[k],t,0.0,0.0,0.0)


        plt.figure()
        # plt.contourf(pert[:,5,:].T)
        plt.contourf(pert_2d[:,:].T)
        plt.colorbar()
        # plt.show()
        plt.close()

        #shift_advected
        #PV.tendencies[shift_advected]

        return





