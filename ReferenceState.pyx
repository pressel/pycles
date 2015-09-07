#!python
#cython: boundscheck=False
#cython: wraparound=True
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport numpy as np
import numpy as np

from scipy.integrate import odeint
include 'parameters.pxi'

cdef extern from "thermodynamic_functions.h":
    inline double qt_from_pv(double p0, double pv)

cdef class ReferenceState:
    def __init__(self, Grid.Grid Gr):

        self.p0 = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.p0_half = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.alpha0 = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.alpha0_half = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')

        return

    def initialize(self,Grid.Grid Gr, Thermodynamics):


        self.sg = Thermodynamics.entropy(self.Pg,self.Tg,self.qtg,0.0,0.0)


        #Form a right hand side for integrating the hydrostatic equation to determine the reference pressure
        def rhs(p,z):
            T,ql,qi = Thermodynamics.eos(np.exp(p),self.sg,self.qtg)
            return -g/(Rd*T*(1.0 + self.qtg + eps_vi * (self.qtg - ql - qi)))

        #Construct arrays for integration points
        z = np.array(Gr.z[Gr.dims.gw-1:-Gr.dims.gw+1])
        z_half = np.append([0.0],np.array(Gr.z_half[Gr.dims.gw:-Gr.dims.gw]))

        #We are integrating the log pressure so need to take the log of the surface pressure
        p0 = np.log(self.Pg)

        p = np.zeros(Gr.dims.ng[2],dtype=np.double,order='c')
        p_half = np.zeros(Gr.dims.ng[2],dtype=np.double,order='c')

        #Perform the integration
        p[Gr.dims.gw-1:-Gr.dims.gw+1] = odeint(rhs,p0,z,hmax=100.0)[:,0]
        p_half[Gr.dims.gw:-Gr.dims.gw] = odeint(rhs,p0,z_half,hmax=100.0)[1:,0]

        #Set boundary conditions
        p[:Gr.dims.gw-1] = p[2*Gr.dims.gw-2:Gr.dims.gw-1:-1]
        p[-Gr.dims.gw+1:]  = p[-Gr.dims.gw-1:-2*Gr.dims.gw:-1]

        p_half[:Gr.dims.gw] = p_half[2*Gr.dims.gw-1:Gr.dims.gw-1:-1]
        p_half[-Gr.dims.gw:] = p_half[-Gr.dims.gw-1:-2*Gr.dims.gw-1:-1]

        p = np.exp(p)
        p_half = np.exp(p_half)

        cdef double [:] p_ = p
        cdef double [:] p_half_ = p_half
        cdef double [:] temperature = np.zeros(Gr.dims.ng[2],dtype=np.double,order='c')
        cdef double [:] temperature_half = np.zeros(Gr.dims.ng[2],dtype=np.double,order='c')
        cdef double [:] alpha = np.zeros(Gr.dims.ng[2],dtype=np.double,order='c')
        cdef double [:] alpha_half = np.zeros(Gr.dims.ng[2],dtype=np.double,order='c')

        cdef double ql, qi, qt
        cdef Py_ssize_t k
        for k in xrange(Gr.dims.ng[2]):
            temperature[k], ql, qi  = Thermodynamics.eos(p_[k],self.sg,self.qtg)
            qv = self.qtg -  (ql + qi)
            alpha[k] = Thermodynamics.alpha(p_[k],temperature[k],self.qtg,qv)

            temperature_half[k], ql, qi  = Thermodynamics.eos(p_half_[k],self.sg,self.qtg)
            qt = self.qtg - (ql + qi)
            alpha_half[k] = Thermodynamics.alpha(p_half_[k],temperature_half[k],self.qtg,qv)

        #print(np.array(Gr.extract_local_ghosted(alpha_half,2)))
        self.alpha0_half = Gr.extract_local_ghosted(alpha_half,2)
        self.alpha0 = Gr.extract_local_ghosted(alpha,2)
        self.p0 = Gr.extract_local_ghosted(p_,2)
        self.p0_half = Gr.extract_local_ghosted(p_half,2)
        self.rho0 = 1.0/np.array(self.alpha0)
        self.rho0_half = 1.0/np.array(self.alpha0_half)

        return