#!python
#cython: boundscheck=False
#cython: initializedcheck=False
#cython: cdivision=True
import os
import netCDF4 as nc
from thermodynamic_functions cimport cpm_c, pv_c, pd_c, exner_c
from entropies cimport sv_c, sd_c, s_tendency_c
import numpy as np
import cython
from libc.math cimport fabs, sin, cos, fmax, fmin, log, abs
from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI
cimport Lookup
from Thermodynamics cimport LatentHeat, ClausiusClapeyron
from TimeStepping cimport TimeStepping
from Grid cimport  Grid
import pylab as plt
try:
    import cPickle as pickle
except:
    import pickle as pickle # for Python 3 users
include 'parameters.pxi'


def ForcingReferenceFactory(namelist, LatentHeat LH, ParallelMPI.ParallelMPI Pa):
    casename =namelist['meta']['casename']
    if casename == 'ZGILS':
        try:
            forcing_type = namelist['forcing']['reference_profile']
        except:
            Pa.root_print('Must specify reference profile type')
            Pa.kill()
        if forcing_type == 'AdjustedAdiabat':
            return AdjustedMoistAdiabat(namelist, LH, Pa)
        elif forcing_type == 'InteractiveRCE' or forcing_type == 'InteractiveRCE_constant':
            return InteractiveReferenceRCE_new(namelist, LH, Pa)
        elif forcing_type =='ReferenceFile':
            return ReferenceRCE(namelist,LH,Pa)
        else:
            Pa.root_print('Reference Profile type not recognized')
            Pa.kill()
    else:
        return ForcingReferenceNone()


cdef extern:
    void c_rrtmg_lw_init(double *cpdair)
    void c_rrtmg_lw (
             int *ncol    ,int *nlay    ,int *icld    ,int *idrv    ,
             double *play    ,double *plev    ,double *tlay    ,double *tlev    ,double *tsfc    ,
             double *h2ovmr  ,double *o3vmr   ,double *co2vmr  ,double *ch4vmr  ,double *n2ovmr  ,double *o2vmr,
             double *cfc11vmr,double *cfc12vmr,double *cfc22vmr,double *ccl4vmr ,double *emis    ,
             int *inflglw ,int *iceflglw,int *liqflglw,double *cldfr   ,
             double *taucld  ,double *cicewp  ,double *cliqwp  ,double *reice   ,double *reliq   ,
             double *tauaer  ,
             double *uflx    ,double *dflx    ,double *hr      ,double *uflxc   ,double *dflxc,  double *hrc,
             double *duflx_dt,double *duflxc_dt )
    void c_rrtmg_sw_init(double *cpdair)
    void c_rrtmg_sw (int *ncol    ,int *nlay    ,int *icld    ,int *iaer    ,
             double *play    ,double *plev    ,double *tlay    ,double *tlev    ,double *tsfc    ,
             double *h2ovmr  ,double *o3vmr   ,double *co2vmr  ,double *ch4vmr  ,double *n2ovmr  ,double *o2vmr,
             double *asdir   ,double *asdif   ,double *aldir   ,double *aldif   ,
             double *coszen  ,double *adjes   ,int *dyofyr  ,double *scon    ,
             int *inflgsw ,int *iceflgsw,int *liqflgsw,double *cldfr   ,
             double *taucld  ,double *ssacld  ,double *asmcld  ,double *fsfcld  ,
             double *cicewp  ,double *cliqwp  ,double *reice   ,double *reliq   ,
             double *tauaer  ,double *ssaaer  ,double *asmaer  ,double *ecaer   ,
             double *swuflx  ,double *swdflx  ,double *swhr    ,double *swuflxc ,double *swdflxc ,double *swhrc)

cdef extern from "thermodynamics_sa.h":
    void eos_c(Lookup.LookupStruct *LT, double(*lam_fp)(double), double(*L_fp)(double, double),
               double p0, double s, double qt, double *T, double *qv, double *ql, double *qi) nogil
cdef extern from "thermodynamic_functions.h":
    inline double pd_c(double p0, double qt, double qv) nogil
    inline double pv_c(double p0, double qt, double qv) nogil
    inline double qv_star_c(const double p0, const double qt, const double pv)nogil
cdef extern from "entropies.h":
    inline double sd_c(double pd, double T) nogil
    inline double sv_c(double pv, double T) nogil
    inline double sc_c(double L, double T) nogil

# These classes compute or read in the reference profiles needed for ZGILS cases
# The base class
cdef class ForcingReferenceBase:
    def __init__(self,namelist,  LatentHeat LH, ParallelMPI.ParallelMPI Pa ):

        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp
        self.CC = ClausiusClapeyron()
        self.CC.initialize(namelist, LH, Pa)

        return
    cpdef initialize(self, Grid Gr, ParallelMPI.ParallelMPI Pa, NetCDFIO_Stats NS, double  S_minus_L):
        self.s = np.zeros(self.npressure, dtype=np.double, order='c')
        self.qt = np.zeros(self.npressure, dtype=np.double, order='c')
        self.temperature = np.zeros(self.npressure, dtype=np.double, order='c')
        self.rv = np.zeros(self.npressure, dtype=np.double, order='c')
        self.u = np.zeros(self.npressure, dtype=np.double, order='c')
        self.v = np.zeros(self.npressure, dtype=np.double, order='c')
        return
    cpdef update(self, ParallelMPI.ParallelMPI Pa,  double S_minus_L, TimeStepping TS ):
        return
    cpdef stats_io(self, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef restart(self, Restart):
        return
    cpdef entropy(self, double p0, double T, double qt, double ql, double qi):
        cdef:
            double qv = qt - ql - qi
            double qd = 1.0 - qt
            double pd = pd_c(p0, qt, qv)
            double pv = pv_c(p0, qt, qv)
            double Lambda = self.Lambda_fp(T)
            double L = self.L_fp(T, Lambda)

        return sd_c(pd, T) * (1.0 - qt) + sv_c(pv, T) * qt + sc_c(L, T) * (ql + qi)

    cpdef eos(self, double p0, double s, double qt):
        cdef:
            double T, qv, qc, ql, qi, lam
        eos_c(&self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, p0, s, qt, &T, &qv, &ql, &qi)
        return T, ql, qi

cdef class ForcingReferenceNone(ForcingReferenceBase):
    def __init__(self):
        self.is_init=True

        return
    cpdef initialize(self, Grid Gr, ParallelMPI.ParallelMPI Pa, NetCDFIO_Stats NS, double  S_minus_L):
        return
    cpdef update(self, ParallelMPI.ParallelMPI Pa,  double S_minus_L, TimeStepping TS ):
        return
    cpdef stats_io(self, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef restart(self, Restart):
        return
# Control simulations use AdjustedMoistAdiabat
# Reference temperature profile correspondends to a moist adiabat
# Reference moisture profile corresponds to a fixed relative humidity given the reference temperature profile
cdef class AdjustedMoistAdiabat(ForcingReferenceBase):
    def __init__(self,namelist,  LatentHeat LH, ParallelMPI.ParallelMPI Pa ):
        self.is_init=False

        try:
            self.Tg  = namelist['forcing']['AdjustedMoistAdiabat']['Tg']
        except:
            self.Tg = 295.0
        try:
            self.Pg = namelist['forcing']['AdjustedMoistAdiabat']['Pg']
        except:
            self.Pg = 1000.0e2

        try:
            self.RH_ref = namelist['forcing']['AdjustedMoistAdiabat']['RH_ref']
        except:
            self.RH_ref = 0.3
        try:
            self.npressure = namelist['forcing']['AdjustedMoistAdiabat']['nlayers']
        except:
            self.npressure = 100
        cdef double [:] p_levels = np.linspace(self.Pg, 0.0, num=self.npressure+1, endpoint=True)
        self.pressure = 0.5 * np.add(p_levels[1:], p_levels[:-1])
        ForcingReferenceBase.__init__(self,namelist, LH, Pa)

        return
    cpdef entropy(self, double p0, double T, double qt, double ql, double qi):
        ForcingReferenceBase.entropy(self, p0, T, qt, ql, qi)
        return

    cpdef eos(self, double p0, double s, double qt):
        ForcingReferenceBase.eos(self, p0, s, qt)
        return


    cpdef initialize(self, Grid Gr, ParallelMPI.ParallelMPI Pa, NetCDFIO_Stats NS, double  S_minus_L):
        if self.is_init:
            return
        '''
        Initialize the forcing reference profiles. These profiles use the temperature corresponding to a moist adiabat,
        but modify the water vapor content to have a given relative humidity. Thus entropy and qt are not conserved.
        '''
        # Default values correspond to Tan et al 2016
        ForcingReferenceBase.initialize(self, Gr, Pa, NS, S_minus_L)

        cdef:
            double pvg = self.CC.LT.fast_lookup(self.Tg)
            double qtg = eps_v * pvg / (self.Pg + (eps_v-1.0)*pvg)
            double sg = self.entropy(self.Pg, self.Tg, qtg, 0.0, 0.0)


        cdef double temp, ql, qi, pv


        # Compute reference state thermodynamic profiles
        for k in xrange(self.npressure):
            temp, ql, qi = self.eos(self.pressure[k], sg, qtg)
            pv = self.CC.LT.fast_lookup(temp) *self.RH_ref
            self.qt[k] = eps_v * pv / (self.pressure[k] + (eps_v-1.0)*pv)
            self.s[k] = self.entropy(self.pressure[k],temp, self.qt[k] , 0.0, 0.0)
            self.temperature[k] = temp
            self.rv[k] = self.qt[k]/(1.0-self.qt[k])
            self.u[k] =  min(-10.0 + (-7.0-(-10.0))/(750.0e2-1000.0e2)*(self.pressure[k]-1000.0e2),-4.0)

        self.is_init = True

        return
    cpdef update(self, ParallelMPI.ParallelMPI Pa,  double S_minus_L, TimeStepping TS):
        return

    cpdef stats_io(self, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef restart(self, Restart):
        return
# Climate change simulations use profiles based on radiative--convective equilibrium solutions obtained as described in
# Zhihong Tan's dissertation (Section 2.6). Zhihong has provided his reference profiles to be archived with the code, so
# this class just reads in the data and interpolates to the simulation pressure grid

cdef class ReferenceRCE(ForcingReferenceBase):
    def __init__(self,  namelist,  LatentHeat LH, ParallelMPI.ParallelMPI Pa ):
        self.is_init=False
        try:
            co2_factor=namelist['radiation']['RRTM']['co2_factor']
        except:
            co2_factor = 1.0
        self.filename = './CGILSdata/RCE_'+ str(co2_factor)+'xCO2.nc'
        ForcingReferenceBase.__init__(self,namelist, LH, Pa)

        return
    @cython.wraparound(True)
    cpdef initialize(self, Grid Gr, ParallelMPI.ParallelMPI Pa, NetCDFIO_Stats NS, double  S_minus_L):
        if self.is_init:
            return


        data = nc.Dataset(self.filename, 'r')
        self.pressure = data.variables['p_full']
        self.npressure = len(self.pressure)
        ForcingReferenceBase.initialize(self,Gr, Pa,NS, S_minus_L)

        self.temperature = data.variables['temp_rc']
        self.qt = data.variables['yv_rc']
        self.u = data.variables['u']
        self.v = data.variables['v']

        # Arrays must be flipped (low to high pressure) to use numpy interp function
        # pressure_ref = data.variables['p_full'][::-1]
        # temperature_ref = data.variables['temp_rc'][::-1]
        # qt_ref = data.variables['yv_rc'][::-1]
        # u_ref = data.variables['u'][::-1]
        # v_ref = data.variables['v'][::-1]
        #
        # self.temperature = np.array(np.interp(pressure_array, pressure_ref, temperature_ref),
        #                             dtype=np.double, order='c')
        # self.qt = np.array(np.interp(pressure_array, pressure_ref, qt_ref), dtype=np.double, order='c')
        # self.u = np.array(np.interp(pressure_array, pressure_ref, u_ref), dtype=np.double, order='c')
        # self.v = np.array(np.interp(pressure_array, pressure_ref, v_ref), dtype=np.double, order='c')


        cdef:
            double pd, pv
            Py_ssize_t k
        # computing entropy assuming sub-saturated
        for k in xrange(self.npressure):
            pv = pv_c(self.pressure[k], self.qt[k], self.qt[k])
            pd = pd_c(self.pressure[k], self.qt[k], self.qt[k])

            self.rv[k] =  self.qt[k]/(1.0-self.qt[k])
            self.s[k] = (sd_c(pd, self.temperature[k]) * (1.0 - self.qt[k])
                         + sv_c(pv, self.temperature[k]) * self.qt[k])
            self.u[k] = self.u[k]*0.5 - 5.0

        self.is_init = True

        return
    cpdef update(self, ParallelMPI.ParallelMPI Pa,  double S_minus_L, TimeStepping TS):
        return

    cpdef stats_io(self, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef restart(self, Restart):
        return
# Here we implement the RCE solution algorithm described in Section 2.6 of Zhihong Tan's thesis to allow updates
# of the Reference profiles as the SST changes
#
#
# cdef class InteractiveReferenceRCE_old(ForcingReferenceBase):
#     def __init__(self,namelist,  LatentHeat LH, ParallelMPI.ParallelMPI Pa ):
#         self.is_init = False
#
#         ForcingReferenceBase.__init__(self,namelist, LH, Pa)
#
#         try:
#             self.fix_wv = namelist['radiation']['RRTM']['fix_wv']
#         except:
#             self.fix_wv = False
#
#         if self.fix_wv:
#             self.filename = namelist['radiation']['RRTM']['fix_wv_table_file']
#
#
#         try:
#             self.RH_subtrop = namelist['forcing']['RH_subtropical']
#         except:
#             self.RH_subtrop = 0.3
#
#
#         # Radiation parameters
#         #--Namelist options related to gas concentrations
#         # try:
#         #     self.co2_factor = namelist['radiation']['RRTM']['co2_factor']
#         # except:
#         #     self.co2_factor = 1.0
#         #--Namelist options related to insolation
#         try:
#             self.dyofyr = namelist['radiation']['RRTM']['dyofyr']
#         except:
#             self.dyofyr = 0
#         try:
#             self.adjes = namelist['radiation']['RRTM']['adjes']
#         except:
#             self.adjes = 0.5
#         try:
#             self.scon = namelist['radiation']['RRTM']['solar_constant']
#         except:
#             self.scon = 1360.0
#         try:
#             self.coszen =namelist['radiation']['RRTM']['coszen']
#         except:
#             self.coszen = 2.0/pi
#         try:
#             self.adif = namelist['radiation']['RRTM']['adif']
#         except:
#             self.adif = 0.06
#         try:
#             self.adir = namelist['radiation']['RRTM']['adir']
#         except:
#             if (self.coszen > 0.0):
#                 self.adir = (.026/(self.coszen**1.7 + .065)+(.15*(self.coszen-0.10)*(self.coszen-0.50)*(self.coszen- 1.00)))
#             else:
#                 self.adir = 0.0
#         return
#
#     cpdef entropy(self, double p0, double T, double qt, double ql, double qi):
#         cdef:
#             double qv = qt - ql - qi
#             double qd = 1.0 - qt
#             double pd = pd_c(p0, qt, qv)
#             double pv = pv_c(p0, qt, qv)
#             double Lambda = self.Lambda_fp(T)
#             double L = self.L_fp(T, Lambda)
#
#         return sd_c(pd, T) * (1.0 - qt) + sv_c(pv, T) * qt + sc_c(L, T) * (ql + qi)
#
#     cpdef eos(self, double p0, double s, double qt):
#         cdef:
#             double T, qv, qc, ql, qi, lam
#         eos_c(&self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, p0, s, qt, &T, &qv, &ql, &qi)
#         return T, ql, qi
#
#
#     cpdef initialize_radiation(self, double co2_factor):
#         #Initialize rrtmg_lw and rrtmg_sw
#         cdef double cpdair = np.float64(cpd)
#         c_rrtmg_lw_init(&cpdair)
#         c_rrtmg_sw_init(&cpdair)
#
#         cdef:
#             Py_ssize_t k
#             Py_ssize_t nlevels = np.shape(self.p_levels)[0]
#             Py_ssize_t nlayers = np.shape(self.p_layers)[0]
#
#         # Read in trace gas data
#         lw_input_file = './RRTMG/lw/data/rrtmg_lw.nc'
#         lw_gas = nc.Dataset(lw_input_file,  "r")
#
#         lw_pressure = np.asarray(lw_gas.variables['Pressure'])
#         lw_absorber = np.asarray(lw_gas.variables['AbsorberAmountMLS'])
#         lw_absorber = np.where(lw_absorber>2.0, np.zeros_like(lw_absorber), lw_absorber)
#         lw_ngas = lw_absorber.shape[1]
#         lw_np = lw_absorber.shape[0]
#
#         # 9 Gases: O3, CO2, CH4, N2O, O2, CFC11, CFC12, CFC22, CCL4
#         # From rad_driver.f90, lines 546 to 552
#         trace = np.zeros((9,lw_np),dtype=np.double,order='F')
#         for i in xrange(lw_ngas):
#             gas_name = ''.join(lw_gas.variables['AbsorberNames'][i,:])
#             if 'O3' in gas_name:
#                 trace[0,:] = lw_absorber[:,i].reshape(1,lw_np)
#             elif 'CO2' in gas_name:
#                 trace[1,:] = lw_absorber[:,i].reshape(1,lw_np) * co2_factor * 400.0/355.0
#             elif 'CH4' in gas_name:
#                 trace[2,:] = lw_absorber[:,i].reshape(1,lw_np)
#             elif 'N2O' in gas_name:
#                 trace[3,:] = lw_absorber[:,i].reshape(1,lw_np)
#             elif 'O2' in gas_name:
#                 trace[4,:] = lw_absorber[:,i].reshape(1,lw_np)
#             elif 'CFC11' in gas_name:
#                 trace[5,:] = lw_absorber[:,i].reshape(1,lw_np)
#             elif 'CFC12' in gas_name:
#                 trace[6,:] = lw_absorber[:,i].reshape(1,lw_np)
#             elif 'CFC22' in gas_name:
#                 trace[7,:] = lw_absorber[:,i].reshape(1,lw_np)
#             elif 'CCL4' in gas_name:
#                 trace[8,:] = lw_absorber[:,i].reshape(1,lw_np)
#
#
#         # From rad_driver.f90, lines 585 to 620
#         trpath = np.zeros((nlevels, 9),dtype=np.double,order='F')
#         plev = np.divide(self.p_levels,100.0*np.ones(nlevels))
#         for i in xrange(1, nlevels):
#             trpath[i,:] = trpath[i-1,:]
#             if plev[i-1]> lw_pressure[0]:
#                 trpath[i,:] = trpath[i,:] + (plev[i-1] - np.max((plev[i],lw_pressure[0])))/g*trace[:,0]
#             for m in xrange(1,lw_np):
#                 plow = np.min((plev[i-1],np.max((plev[i], lw_pressure[m-1]))))
#                 pupp = np.min((plev[i-1],np.max((plev[i],    lw_pressure[m]))))
#                 if (plow > pupp):
#                     pmid = 0.5*(plow+pupp)
#                     wgtlow = (pmid-lw_pressure[m])/(lw_pressure[m-1]-lw_pressure[m])
#                     wgtupp = (lw_pressure[m-1]-pmid)/(lw_pressure[m-1]-lw_pressure[m])
#                     trpath[i,:] = trpath[i,:] + (plow-pupp)/g*(wgtlow*trace[:,m-1]  + wgtupp*trace[:,m])
#             if plev[i] < lw_pressure[lw_np-1]:
#                 trpath[i,:] = trpath[i,:] + (np.min((plev[i-1],lw_pressure[lw_np-1]))-plev[i])/g*trace[:,lw_np-1]
#         tmpTrace = np.zeros((nlayers, 9),dtype=np.double,order='F')
#         for i in xrange(9):
#             for k in xrange(nlayers):
#                 tmpTrace[k,i] = g/(plev[k] - plev[k+1])*(trpath[k+1,i]-trpath[k,i])
#
#         self.o3vmr  = np.array(tmpTrace[:,0],dtype=np.double, order='F')
#         self.co2vmr = np.array(tmpTrace[:,1],dtype=np.double, order='F')
#         self.ch4vmr =  np.array(tmpTrace[:,2],dtype=np.double, order='F')
#         self.n2ovmr =  np.array(tmpTrace[:,3],dtype=np.double, order='F')
#         self.o2vmr  =  np.array(tmpTrace[:,4],dtype=np.double, order='F')
#         self.cfc11vmr =  np.array(tmpTrace[:,5],dtype=np.double, order='F')
#         self.cfc12vmr =  np.array(tmpTrace[:,6],dtype=np.double, order='F')
#         self.cfc22vmr = np.array( tmpTrace[:,7],dtype=np.double, order='F')
#         self.ccl4vmr  =  np.array(tmpTrace[:,8],dtype=np.double, order='F')
#
#
#
#     cpdef compute_radiation(self):
#
#         cdef:
#             Py_ssize_t k
#             Py_ssize_t ncols = 1
#             Py_ssize_t nlayers = self.nlayers
#             Py_ssize_t nlevels = self.nlevels
#             double [:,:] play_in = np.array(np.expand_dims(self.p_layers,axis=0), dtype=np.double, order='F')/100.0
#             double [:,:] plev_in = np.array(np.expand_dims(self.p_levels,axis=0), dtype=np.double, order='F')/100.0
#             double [:,:] tlay_in = np.array(np.expand_dims(self.t_layers,axis=0), dtype=np.double, order='F')
#
#             double [:,:] tlev_in = np.zeros((ncols,nlevels), dtype=np.double, order='F')
#             double [:] tsfc_in = np.ones((ncols),dtype=np.double,order='F') * self.sst
#             double [:,:] h2ovmr_in = np.zeros((ncols, nlayers),dtype=np.double,order='F')
#             double [:,:] o3vmr_in  = np.array(np.expand_dims(self.o3vmr,axis=0),dtype=np.double,order='F')
#             double [:,:] co2vmr_in = np.array(np.expand_dims(self.co2vmr,axis=0),dtype=np.double,order='F')
#             double [:,:] ch4vmr_in = np.array(np.expand_dims(self.ch4vmr,axis=0),dtype=np.double,order='F')
#             double [:,:] n2ovmr_in = np.array(np.expand_dims(self.n2ovmr,axis=0),dtype=np.double,order='F')
#             double [:,:] o2vmr_in  = np.array(np.expand_dims(self.o2vmr,axis=0),dtype=np.double,order='F')
#             double [:,:] cfc11vmr_in = np.array(np.expand_dims(self.cfc11vmr,axis=0),dtype=np.double,order='F')
#             double [:,:] cfc12vmr_in = np.array(np.expand_dims(self.cfc12vmr,axis=0),dtype=np.double,order='F')
#             double [:,:] cfc22vmr_in = np.array(np.expand_dims(self.cfc22vmr,axis=0),dtype=np.double,order='F')
#             double [:,:] ccl4vmr_in = np.array(np.expand_dims(self.ccl4vmr,axis=0),dtype=np.double,order='F')
#             double [:,:] emis_in = np.ones((ncols, 16),dtype=np.double,order='F') * 0.95
#             double [:,:] cldfr_in  = np.zeros((ncols,nlayers), dtype=np.double,order='F')
#             double [:,:] cicewp_in = np.zeros((ncols,nlayers),dtype=np.double,order='F')
#             double [:,:] cliqwp_in = np.zeros((ncols,nlayers),dtype=np.double,order='F')
#             double [:,:] reice_in  = np.zeros((ncols,nlayers),dtype=np.double,order='F')
#             double [:,:] reliq_in  = np.ones((ncols,nlayers),dtype=np.double,order='F') * 2.5
#             double [:] coszen_in = np.ones(ncols,dtype=np.double,order='F') *self.coszen
#             double [:] asdir_in = np.ones(ncols,dtype=np.double,order='F') * self.adir
#             double [:] asdif_in = np.ones(ncols,dtype=np.double,order='F') * self.adif
#             double [:] aldir_in = np.ones(ncols,dtype=np.double,order='F') * self.adir
#             double [:] aldif_in = np.ones(ncols,dtype=np.double,order='F') * self.adif
#             double [:,:,:] taucld_lw_in  = np.zeros((16,ncols,nlayers),dtype=np.double,order='F')
#             double [:,:,:] tauaer_lw_in  = np.zeros((ncols,nlayers,16),dtype=np.double,order='F')
#             double [:,:,:] taucld_sw_in  = np.zeros((14,ncols,nlayers),dtype=np.double,order='F')
#             double [:,:,:] ssacld_sw_in  = np.zeros((14,ncols,nlayers),dtype=np.double,order='F')
#             double [:,:,:] asmcld_sw_in  = np.zeros((14,ncols,nlayers),dtype=np.double,order='F')
#             double [:,:,:] fsfcld_sw_in  = np.zeros((14,ncols,nlayers),dtype=np.double,order='F')
#             double [:,:,:] tauaer_sw_in  = np.zeros((ncols,nlayers,14),dtype=np.double,order='F')
#             double [:,:,:] ssaaer_sw_in  = np.zeros((ncols,nlayers,14),dtype=np.double,order='F')
#             double [:,:,:] asmaer_sw_in  = np.zeros((ncols,nlayers,14),dtype=np.double,order='F')
#             double [:,:,:] ecaer_sw_in  = np.zeros((ncols,nlayers,6),dtype=np.double,order='F')
#
#             # Output
#             double[:,:] uflx_lw_out = np.zeros((ncols,nlevels),dtype=np.double,order='F')
#             double[:,:] dflx_lw_out = np.zeros((ncols,nlevels),dtype=np.double,order='F')
#             double[:,:] hr_lw_out = np.zeros((ncols,nlayers),dtype=np.double,order='F')
#             double[:,:] uflxc_lw_out = np.zeros((ncols,nlevels),dtype=np.double,order='F')
#             double[:,:] dflxc_lw_out = np.zeros((ncols,nlevels),dtype=np.double,order='F')
#             double[:,:] hrc_lw_out = np.zeros((ncols,nlayers),dtype=np.double,order='F')
#             double[:,:] duflx_dt_out = np.zeros((ncols,nlevels),dtype=np.double,order='F')
#             double[:,:] duflxc_dt_out = np.zeros((ncols,nlevels),dtype=np.double,order='F')
#             double[:,:] uflx_sw_out = np.zeros((ncols,nlevels),dtype=np.double,order='F')
#             double[:,:] dflx_sw_out = np.zeros((ncols,nlevels),dtype=np.double,order='F')
#             double[:,:] hr_sw_out = np.zeros((ncols,nlayers),dtype=np.double,order='F')
#             double[:,:] uflxc_sw_out = np.zeros((ncols,nlevels),dtype=np.double,order='F')
#             double[:,:] dflxc_sw_out = np.zeros((ncols,nlevels),dtype=np.double,order='F')
#             double[:,:] hrc_sw_out = np.zeros((ncols,nlayers),dtype=np.double,order='F')
#
#             double rv_to_reff = np.exp(np.log(1.2)**2.0)*10.0*1000.0
#
#         with nogil:
#             tlev_in[0,0] = self.sst
#             for k in xrange(1,nlayers):
#                 tlev_in[0,k] = 0.5 * (tlay_in[0,k-1]+tlay_in[0,k])
#             tlev_in[0, nlayers] = 2.0 * tlay_in[0,nlayers-1] - tlev_in[0,nlayers-1]
#             for k in xrange(nlayers):
#                 h2ovmr_in[0,k] = self.qv_layers[k]/(1.0 - self.qv_layers[k]) * Rv/Rd
#
#
#
#
#         cdef:
#             int ncol = ncols
#             int nlay = nlayers
#             int icld = 1
#             int idrv = 0
#             int iaer = 0
#             int inflglw = 2
#             int iceflglw = 3
#             int liqflglw = 1
#             int inflgsw = 2
#             int iceflgsw = 3
#             int liqflgsw = 1
#
#         c_rrtmg_lw (
#             &ncol, &nlay, &icld, &idrv, &play_in[0,0], &plev_in[0,0], &tlay_in[0,0], &tlev_in[0,0], &tsfc_in[0],
#             &h2ovmr_in[0,0], &o3vmr_in[0,0], &co2vmr_in[0,0], &ch4vmr_in[0,0], &n2ovmr_in[0,0], &o2vmr_in[0,0],
#             &cfc11vmr_in[0,0], &cfc12vmr_in[0,0], &cfc22vmr_in[0,0], &ccl4vmr_in[0,0], &emis_in[0,0], &inflglw,
#             &iceflglw,&liqflglw, &cldfr_in[0,0], &taucld_lw_in[0,0,0], &cicewp_in[0,0], &cliqwp_in[0,0], &reice_in[0,0],
#             &reliq_in[0,0], &tauaer_lw_in[0,0,0], &uflx_lw_out[0,0], &dflx_lw_out[0,0], &hr_lw_out[0,0],
#             &uflxc_lw_out[0,0], &dflxc_lw_out[0,0], &hrc_lw_out[0,0], &duflx_dt_out[0,0], &duflxc_dt_out[0,0] )
#
#         c_rrtmg_sw (
#             &ncol, &nlay, &icld, &iaer, &play_in[0,0], &plev_in[0,0], &tlay_in[0,0], &tlev_in[0,0],&tsfc_in[0],
#             &h2ovmr_in[0,0], &o3vmr_in[0,0], &co2vmr_in[0,0], &ch4vmr_in[0,0], &n2ovmr_in[0,0],&o2vmr_in[0,0],
#             &asdir_in[0], &asdif_in[0], &aldir_in[0], &aldif_in[0], &coszen_in[0], &self.adjes, &self.dyofyr,
#             &self.scon, &inflgsw, &iceflgsw, &liqflgsw, &cldfr_in[0,0], &taucld_sw_in[0,0,0], &ssacld_sw_in[0,0,0],
#             &asmcld_sw_in[0,0,0], &fsfcld_sw_in[0,0,0], &cicewp_in[0,0], &cliqwp_in[0,0], &reice_in[0,0], &reliq_in[0,0],
#             &tauaer_sw_in[0,0,0], &ssaaer_sw_in[0,0,0], &asmaer_sw_in[0,0,0], &ecaer_sw_in[0,0,0], &uflx_sw_out[0,0],
#             &dflx_sw_out[0,0], &hr_sw_out[0,0], &uflxc_sw_out[0,0], &dflxc_sw_out[0,0], &hrc_sw_out[0,0])
#
#
#         with nogil:
#             self.toa_flux = -uflx_lw_out[0,nlayers] + dflx_lw_out[0,nlayers] - uflx_sw_out[0,nlayers] + dflx_sw_out[0,nlayers]
#             self.total_column_influx = self.toa_flux +uflx_lw_out[0,0] - dflx_lw_out[0,0] + uflx_sw_out[0,0] - dflx_sw_out[0,0]
#             for k in xrange(nlayers):
#                 self.t_tend_rad[k] = (hr_lw_out[0,k] + hr_sw_out[0,k])/86400.0
#
#         return
#
#
#     cpdef initialize(self,  ParallelMPI.ParallelMPI Pa, double [:] pressure_array, double sst_tropical, double S_minus_L):
#         cdef:
#             double pv, pd
#             Py_ssize_t k
#         ForcingReferenceBase.initialize(self,Pa, pressure_array, sst_tropical, S_minus_L)
#         self.sst = sst_tropical
#         self.dt_rce  =3600.0  #1 hr?
#         self.RH_surf = 0.7 # check this
#         self.RH_tropical = 0.7
#
#
#         # pressure coordinates
#         self.nlayers = 100
#         self.nlevels = self.nlayers + 1
#         self.p_levels = np.linspace(Pg, 0.0, num=self.nlevels, endpoint=True)
#         self.p_layers = 0.5 * np.add(self.p_levels[1:],self.p_levels[:-1])
#
#
#         self.initialize_radiation(co2_factor)
#
#
#         self.t_layers = np.zeros(self.nlayers, dtype=np.double, order='c')
#         self.qv_layers =np.zeros(self.nlayers, dtype=np.double, order='c')
#         self.t_tend_rad = np.zeros(np.shape(self.t_layers),dtype =np.double, order='c')
#
#         if self.fix_wv:
#         # Here open pkl
#         # create t_table_wv and put the values into it
#         # Interpolate the temperature profile into t_layers
#         # then get qv_layers which will need to be held fixed
#             pkl_file = open(self.filename, 'rb')
#             wv_dict = pickle.load(pkl_file)
#             dims = np.shape(wv_dict['t_table'][:,:])
#             self.t_table_wv = LookupProfiles(dims[0], dims[1])
#             self.t_table_wv.table_vals = wv_dict['t_table'][:,:]
#             self.t_table_wv.access_vals = wv_dict['sst'][:]
#             self.t_table_wv.lookup(300.0)
#             with nogil:
#                 for k in xrange(self.nlayers):
#                     # pv = self.CC.LT.fast_lookup(self.t_table_wv.profile_interp[k]) * self.RH_subtrop
#                     pv = self.CC.LT.fast_lookup(self.t_table_wv.profile_interp[k]) * self.RH_tropical
#                     pd = self.p_layers[k] - pv
#                     self.qv_layers[k] = pv/(pd * eps_vi + pv)
#                 for k in xrange(1,self.nlayers):
#                     self.qv_layers[k] = fmin(self.qv_layers[k], self.qv_layers[k-1])
#
#
#
#         #initialize the lookup table
#         cdef Py_ssize_t n_sst = 31
#         self.t_table = LookupProfiles(n_sst,self.nlayers)
#         self.t_table.access_vals =  np.linspace(Tg-15.0, Tg+15.0,n_sst)
#
#         self.p_tropo_store = np.zeros(n_sst, dtype=np.double, order='c')
#         self.toa_store = np.zeros(n_sst, dtype=np.double, order='c')
#         self.tci_store = np.zeros(n_sst, dtype=np.double, order='c')
#
#
#         cdef:
#             Py_ssize_t sst_index
#
#
#         # Set the initial tropopause height guess
#         if Pa.rank == 0:
#             k = 0
#             while self.p_layers[k] > 400.0e2:
#                 self.index_h_min = k
#                 self.index_h = self.index_h_min
#                 k += 1
#
#             for sst_index in xrange(n_sst):
#                 self.sst = self.t_table.access_vals[sst_index]
#                 # print('doing rce for '+str(self.sst))
#                 self.rce_step(self.sst)
#                 self.t_table.table_vals[sst_index,:] = self.t_layers[:]
#                 self.p_tropo_store[sst_index] = self.p_layers[self.index_h]
#                 self.tci_store[sst_index] = self.total_column_influx
#                 self.toa_store[sst_index] = self.toa_flux
#
#         self.t_table.communicate(Pa)
#
#         ###---Commment out below when running on cluster
#         # This is just for checking the results
#
#         ###############################################################
#         # data = nc.Dataset('./CGILSdata/RCE_8xCO2.nc', 'r')
#         # # Arrays must be flipped (low to high pressure) to use numpy interp function
#         # pressure_ref = data.variables['p_full'][:]
#         # temperature_ref = data.variables['temp_rc'][:]
#         #
#         # self.t_table.lookup(Tg-1.5)
#         if Pa.rank==0:
#             dict = {}
#             dict['t_table'] = np.asarray(self.t_table.table_vals)
#             dict['sst'] = np.asarray(self.t_table.access_vals)
#             dict['net_rad_in'] = np.asarray(self.tci_store)
#             dict['toa_influx'] = np.asarray(self.toa_store)
#             dict['p_tropo'] = np.asarray(self.p_tropo_store)
#             pickle.dump(dict, open('IRCE_SST_'+str(int(Tg)) +'_'+str(co2_factor)+'xCO2.pkl', "wb"  ))
#         #
#         #
#         #     plt.figure(1)
#         #     try:
#         #         for k in xrange(n_sst):
#         #             plt.plot(self.t_table.table_vals[k,:], np.divide(self.p_layers[:],100.0),'-k')
#         #         # plt.plot(temperature_ref, np.divide(pressure_ref,100.0), '--k')
#         #         # plt.plot(self.t_table.profile_interp,np.divide(self.p_layers[:],100.0),'-r' )
#         #         plt.xlabel('Temperature, K',fontsize=16)
#         #         plt.ylabel('Pressure, hPa',fontsize=16)
#         #         plt.xlim(180,340)
#         #         plt.gca().invert_yaxis()
#         #     except:
#         #         pass
#         #
#         #
#         #     plt.figure(2)
#         #     try:
#         #         plt.plot(self.t_table.access_vals[:], np.divide(self.p_tropo_store[:],100.0),'-k')
#         #         plt.xlabel('SST, K',fontsize=16)
#         #         plt.ylabel('Pressure at tropopause, hPa',fontsize=16)
#         #         plt.ylim(50,350)
#         #         plt.xlim(285,325)
#         #     except:
#         #         pass
#         #
#         #     plt.figure(3)
#         #     try:
#         #         plt.plot(self.t_table.access_vals[:], self.toa_store[:],'-k')
#         #         plt.xlabel('Tropical SST, K',fontsize=16)
#         #         plt.ylabel(r'TOA influx, W m$^{-2}$',fontsize=16)
#         #         plt.ylim(110,160)
#         #         plt.xlim(285,325)
#         #     except:
#         #         pass
#         #     plt.show()
#             #################################################################
#
#         # Now set the current reference profile (assuming we want it at domain SST+deltaT...)
#
#         self.t_table.lookup(Tg)
#
#         with nogil:
#             for k in xrange(self.nlayers):
#                 self.t_layers[k] = self.t_table.profile_interp[k]
#                 pv = self.CC.LT.fast_lookup(self.t_layers[k]) * self.RH_subtrop
#                 pd = self.p_layers[k] - pv
#                 self.qv_layers[k] = pv/(pd * eps_vi + pv)
#             for k in xrange(1,self.nlayers):
#                 self.qv_layers[k] = fmin(self.qv_layers[k], self.qv_layers[k-1])
#
#         self.temperature = np.array(np.interp(pressure_array, self.p_layers[::-1], self.t_layers[::-1]), dtype=np.double, order='c')
#         self.qt = np.array(np.interp(pressure_array, self.p_layers[::-1], self.qv_layers[::-1]), dtype=np.double, order='c')
#         cdef:
#             Py_ssize_t nz = np.shape(pressure_array)[0]
#
#         for k in xrange(nz):
#             self.s[k] = self.entropy(pressure_array[k], self.temperature[k], self.qt[k], 0.0, 0.0)
#             self.rv[k] = self.qt[k]/(1.0-self.qt[k])
#             self.u[k] = fmin(-10.0 + (-7.0-(-10.0))/(750.0e2-1000.0e2)*(pressure_array[k]-1000.0e2),-4.0)
#
#         self.is_init=True
#
#         return
#
#     cpdef update_qv(self, double p, double t, double rh):
#         cdef double pv, pd, qv
#         pv = self.CC.LT.fast_lookup(t) * rh
#         pd = p - pv
#         qv = pv/(pd * eps_vi + pv)
#         return  qv
#
#     cpdef compute_adiabat(self, double Tg, double Pg, double RH_surf):
#         cdef:
#             double pvg = self.CC.LT.fast_lookup(Tg) * RH_surf
#             double qtg = eps_v * pvg / (Pg + (eps_v-1.0)*pvg)
#             double sg = self.entropy(Pg, Tg, qtg, 0.0, 0.0)
#
#
#         cdef:
#             double temperature, ql, qi, pv
#             Py_ssize_t k
#
#         # Compute reference state thermodynamic profiles
#         for k in xrange(self.nlayers):
#             temperature, ql, qi = self.eos(self.p_layers[k], sg, qtg)
#             qtg = fmax(qtg-ql, 1e-10)
#
#             if np.isnan(temperature):
#                 self.t_layers[k] = self.t_layers[k-1]
#             else:
#                 self.t_layers[k] = temperature
#
#             if not self.fix_wv:
#                 self.qv_layers[k] = self.update_qv(self.p_layers[k], self.t_layers[k], self.RH_tropical)
#
#
#         return
#
#     cpdef rce_step(self, double Tg ):
#         self.compute_adiabat(Tg,self.p_levels[0], self.RH_surf)
#         cdef:
#             Py_ssize_t k, sub
#             double [:] t_adi = np.array(self.t_layers, dtype=np.double, copy=True, order='c')
#             double [:] qv_adi = np.array(self.qv_layers, dtype=np.double, copy=True, order='c')
#             Py_ssize_t index_h_old = 0
#             double delta_t, rhval, pv, pd
#         self.tropo_converged = False
#         self.index_h = self.index_h - 2
#
#         while not self.tropo_converged:
#             # print(self.index_h)
#             for k in xrange(self.nlayers):
#                 self.t_layers[k] = t_adi[k]
#                 if not self.fix_wv:
#                     self.qv_layers[k] = qv_adi[k]
#
#
#             delta_t = 100.0
#             while delta_t > 0.001:
#
#                 # update temperatures due to radiation
#                 self.compute_radiation()
#
#
#                 delta_t = 0.0
#                 with nogil:
#                     for k in xrange(self.index_h,self.nlayers):
#                         self.t_layers[k] = self.t_layers[k] + self.t_tend_rad[k] * self.dt_rce
#                         delta_t = fmax(delta_t, fabs(self.t_tend_rad[k] * self.dt_rce))
#                         if not self.fix_wv:
#                             rhval = self.RH_tropical
#                             pv = self.CC.LT.fast_lookup(self.t_layers[k]) * rhval
#                             pd = self.p_layers[k] - pv
#                             self.qv_layers[k] = pv/(pd * eps_vi + pv)
#                             self.qv_layers[k] = fmin(self.qv_layers[k], self.qv_layers[k-1])
#
#             # print('t_layers ', self.t_layers[self.index_h], 't_adi ', t_adi[self.index_h])
#             if self.t_layers[self.index_h] < t_adi[self.index_h]:
#                 self.index_h +=1
#                 self.tropo_converged = False
#             else:
#                 self.tropo_converged = True
#                 # print('Tropo is converged')
#             #     print('if option 1')
#             #     index_h_old = self.index_h
#             #     k=self.index_h
#             #     while self.t_layers[k] <= t_adi[k]:
#             #         print(k, self.t_layers[k], t_adi[k])
#             #         self.index_h = k
#             #         k+=1
#             # elif self.t_layers[self.index_h] > t_adi[self.index_h]:
#             #     print('if option 2')
#             #     index_h_old = self.index_h
#             #     k=self.index_h
#             #     while self.t_layers[k] >= t_adi[k]:
#             #         print(k, self.t_layers[k], t_adi[k])
#             #         self.index_h = k
#             #         k-=1
#             # else:
#             #     print('if option 4')
#             #     index_h_old = self.index_h
#             # print('old, new index', index_h_old, self.index_h)
#             #
#             # print('total column influx', self.total_column_influx)
#             # plt.figure('T_profiles')
#             # plt.plot(self.t_layers, np.divide(self.p_layers[:],100.0), '-sr')
#             # plt.plot(t_adi, np.divide(self.p_layers[:],100.0), '-sb')
#             # plt.plot(t_adi[self.index_h], self.p_layers[self.index_h]/100.0,'om')
#             # plt.plot(self.t_layers[self.index_h], self.p_layers[self.index_h]/100.0,'oc')
#             # plt.gca().invert_yaxis()
#             # plt.savefig('SST_'+str(Tg)+'_index_'+str(self.index_h)+'.png')
#             # plt.close()
#
#
#
#         return
#     cpdef update(self, double [:] pressure_array, double Tg):
#         # Now set the current reference profile
#         self.t_table.lookup(Tg)
#         cdef:
#             double pv, pd
#             Py_ssize_t k
#         with nogil:
#             for k in xrange(self.nlayers):
#                 self.t_layers[k] = self.t_table.profile_interp[k]
#                 pv = self.CC.LT.fast_lookup(self.t_layers[k]) * self.RH_subtrop
#                 pd = self.p_layers[k] - pv
#                 self.qv_layers[k] = pv/(pd * eps_vi + pv)
#             for k in xrange(1,self.nlayers):
#                 self.qv_layers[k] = fmin(self.qv_layers[k], self.qv_layers[k-1])
#
#         cdef:
#             double [:] temperature_ = np.array(np.interp(pressure_array, self.p_layers[::-1], self.t_layers[::-1]), dtype=np.double, order='c')
#             double [:] qt_ = np.array(np.interp(pressure_array, self.p_layers[::-1], self.qv_layers[::-1]), dtype=np.double, order='c')
#
#         cdef Py_ssize_t nz = np.shape(pressure_array)[0]
#         for k in xrange(nz):
#             self.temperature[k] = temperature_[k]
#             self.qt[k] = qt_[k]
#             self.s[k] = self.entropy(pressure_array[k], self.temperature[k], self.qt[k], 0.0, 0.0)
#             self.rv[k] = self.qt[k]/(1.0-self.qt[k])
#
#         return
#
#

#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################

cdef class InteractiveReferenceRCE_new(ForcingReferenceBase):
    def __init__(self,namelist,  LatentHeat LH, ParallelMPI.ParallelMPI Pa ):
        self.is_init = False

        ForcingReferenceBase.__init__(self,namelist, LH, Pa)
        try:
            self.read_pkl = namelist['forcing']['RCE']['read_pkl']
        except:
            self.read_pkl = False
        if self.read_pkl:
            try:
                self.pkl_file = str(namelist['forcing']['RCE']['pkl_file'])
            except:
                Pa.root_print('Must specify pkl file')
                Pa.kill()

        try:
            self.verbose = namelist['forcing']['RCE']['verbose']
        except:
            self.verbose = False
        try:
            self.out_dir = str(namelist['forcing']['RCE']['out_dir'])
        except:
            uuid = str(namelist['meta']['uuid'])
            self.out_dir = str(os.path.join(namelist['output']['output_root']
                                            + 'Output.' + namelist['meta']['simname']
                                            + '.' + uuid[-5:]))
        if Pa.rank == 0:
            try:
                os.mkdir(self.out_dir)
            except:
                pass
        try:
            self.lapse_rate_type = str(namelist['forcing']['RCE']['lapse_rate_type'])
        except:
            self.lapse_rate_type='saturated'

        try:
            self.RH_subtrop = namelist['forcing']['RCE']['RH_subtropical']
        except:
            self.RH_subtrop = 0.3

        try:
            self.RH_tropical = namelist['forcing']['RCE']['RH_tropical']
        except:
            self.RH_tropical = 0.7

        try:
            self.RH_surface = namelist['forcing']['RCE']['RH_surface']
        except:
            self.RH_surface = 0.7
        if self.lapse_rate_type == 'saturated':
            self.RH_surface = 1.0
        try:
            self.nlayers = namelist['forcing']['RCE']['nlayers']
        except:
            self.nlayers = 100

        self.nlevels = self.nlayers + 1
        self.npressure = self.nlayers

        try:
            self.p_surface = namelist['forcing']['RCE']['p_surface']
        except:
            self.p_surface = 1020.0e2
        try:
            self.dt_rce = namelist['forcing']['RCE']['dt_rce']
        except:
            self.dt_rce = 3600.0 * 6.0
        try:
            self.delta_T_max = namelist['forcing']['RCE']['delta_T_max']
        except:
            self.delta_T_max = 1.0e-3/86400.0
        try:
            self.toa_error_max = namelist['forcing']['RCE']['toa_error_max']
        except:
            self.toa_error_max = 0.05
        try:
            self.max_steps = namelist['forcing']['RCE']['max_steps']
        except:
            self.max_steps = 100000
        try:
            self.toa_update_criterion = namelist['forcing']['RCE']['toa_update_criterion']
        except:
            self.toa_update_criterion= 0.5 # W/m62
        try:
            self.toa_update_timescale = namelist['forcing']['RCE']['toa_update_timescale']
        except:
            self.toa_update_timescale = 10.0 * 86400.0

        try:
            self.adjust_S_minus_L = namelist['forcing']['RCE']['adjust_S_minus_L']
        except:
            self.adjust_S_minus_L = False
        try:
            self.S_minus_L_fixed_val = namelist['forcing']['RCE']['S_minus_L_fixed_val']
        except:
            self.S_minus_L_fixed_val = 50.0

        # Radiation parameters
        #--Namelist options related to gas concentrations
        try:
            self.co2_factor = namelist['forcing']['RCE']['co2_factor']
        except:
            try:
                self.co2_factor = namelist['radiation']['RRTM']['co2_factor']
            except:
                self.co2_factor = 1.0
        #--Namelist options related to insolation
        try:
            self.dyofyr = namelist['forcing']['RCE']['dyofyr']
        except:
            self.dyofyr = 0
        try:
            self.adjes = namelist['forcing']['RCE']['adjes']
        except:
            self.adjes = 1.0
        try:
            self.scon = namelist['forcing']['RCE']['solar_constant']
        except:
            self.scon = 1365.0
        try:
            self.coszen =namelist['forcing']['RCE']['coszen']
        except:
            self.coszen = 0.2916
        try:
            self.adif = namelist['forcing']['RCE']['adif']
        except:
            self.adif = 0.18
        try:
            self.adir = namelist['forcing']['RCE']['adir']
        except:
            self.adir = 0.18
        return



    cpdef initialize_radiation(self):
        #Initialize rrtmg_lw and rrtmg_sw
        cdef double cpdair = np.float64(cpd)
        c_rrtmg_lw_init(&cpdair)
        c_rrtmg_sw_init(&cpdair)

        cdef:
            Py_ssize_t k
            Py_ssize_t nlevels = np.shape(self.p_levels)[0]
            Py_ssize_t nlayers = np.shape(self.p_layers)[0]

        # Read in trace gas data
        lw_input_file = './RRTMG/lw/data/rrtmg_lw.nc'
        lw_gas = nc.Dataset(lw_input_file,  "r")

        lw_pressure = np.asarray(lw_gas.variables['Pressure'])
        lw_absorber = np.asarray(lw_gas.variables['AbsorberAmountMLS'])
        lw_absorber = np.where(lw_absorber>2.0, np.zeros_like(lw_absorber), lw_absorber)
        lw_ngas = lw_absorber.shape[1]
        lw_np = lw_absorber.shape[0]

        # 9 Gases: O3, CO2, CH4, N2O, O2, CFC11, CFC12, CFC22, CCL4
        # From rad_driver.f90, lines 546 to 552
        trace = np.zeros((9,lw_np),dtype=np.double,order='F')
        for i in xrange(lw_ngas):
            gas_name = ''.join(lw_gas.variables['AbsorberNames'][i,:])
            if 'O3' in gas_name:
                trace[0,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CO2' in gas_name:
                trace[1,:] = np.ones((1,lw_np),dtype=np.double,order='F') * self.co2_factor * 400.0e-6
            elif 'CH4' in gas_name:
                trace[2,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'N2O' in gas_name:
                trace[3,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'O2' in gas_name:
                trace[4,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CFC11' in gas_name:
                trace[5,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CFC12' in gas_name:
                trace[6,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CFC22' in gas_name:
                trace[7,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CCL4' in gas_name:
                trace[8,:] = lw_absorber[:,i].reshape(1,lw_np)


        # From rad_driver.f90, lines 585 to 620
        trpath = np.zeros((nlevels, 9),dtype=np.double,order='F')
        plev = np.divide(self.p_levels,100.0*np.ones(nlevels))
        for i in xrange(1, nlevels):
            trpath[i,:] = trpath[i-1,:]
            if plev[i-1]> lw_pressure[0]:
                trpath[i,:] = trpath[i,:] + (plev[i-1] - np.max((plev[i],lw_pressure[0])))/g*trace[:,0]
            for m in xrange(1,lw_np):
                plow = np.min((plev[i-1],np.max((plev[i], lw_pressure[m-1]))))
                pupp = np.min((plev[i-1],np.max((plev[i],    lw_pressure[m]))))
                if (plow > pupp):
                    pmid = 0.5*(plow+pupp)
                    wgtlow = (pmid-lw_pressure[m])/(lw_pressure[m-1]-lw_pressure[m])
                    wgtupp = (lw_pressure[m-1]-pmid)/(lw_pressure[m-1]-lw_pressure[m])
                    trpath[i,:] = trpath[i,:] + (plow-pupp)/g*(wgtlow*trace[:,m-1]  + wgtupp*trace[:,m])
            if plev[i] < lw_pressure[lw_np-1]:
                trpath[i,:] = trpath[i,:] + (np.min((plev[i-1],lw_pressure[lw_np-1]))-plev[i])/g*trace[:,lw_np-1]
        tmpTrace = np.zeros((nlayers, 9),dtype=np.double,order='F')
        for i in xrange(9):
            for k in xrange(nlayers):
                tmpTrace[k,i] = g/(plev[k] - plev[k+1])*(trpath[k+1,i]-trpath[k,i])

        self.o3vmr  = np.array(tmpTrace[:,0],dtype=np.double, order='F')
        self.co2vmr = np.array(tmpTrace[:,1],dtype=np.double, order='F')
        self.ch4vmr =  np.array(tmpTrace[:,2],dtype=np.double, order='F')
        self.n2ovmr =  np.array(tmpTrace[:,3],dtype=np.double, order='F')
        self.o2vmr  =  np.array(tmpTrace[:,4],dtype=np.double, order='F')
        self.cfc11vmr =  np.array(tmpTrace[:,5],dtype=np.double, order='F')
        self.cfc12vmr =  np.array(tmpTrace[:,6],dtype=np.double, order='F')
        self.cfc22vmr = np.array( tmpTrace[:,7],dtype=np.double, order='F')
        self.ccl4vmr  =  np.array(tmpTrace[:,8],dtype=np.double, order='F')

        self.dTdt_rad_lw = np.zeros((nlayers,), dtype=np.double, order='c')
        self.dTdt_rad_sw = np.zeros((nlayers,), dtype=np.double, order='c')
        self.uflux_lw = np.zeros((nlevels,), dtype=np.double, order='c')
        self.dflux_lw = np.zeros((nlevels,), dtype=np.double, order='c')
        self.uflux_sw = np.zeros((nlevels,), dtype=np.double, order='c')
        self.dflux_sw = np.zeros((nlevels,), dtype=np.double, order='c')

        return



    cpdef compute_radiation(self):

        cdef:
            Py_ssize_t k
            Py_ssize_t ncols = 1
            Py_ssize_t nlayers = self.nlayers
            Py_ssize_t nlevels = self.nlevels
            double [:,:] play_in = np.array(np.expand_dims(self.p_layers,axis=0), dtype=np.double, order='F')/100.0
            double [:,:] plev_in = np.array(np.expand_dims(self.p_levels,axis=0), dtype=np.double, order='F')/100.0
            double [:,:] tlay_in = np.array(np.expand_dims(self.t_layers,axis=0), dtype=np.double, order='F')

            double [:,:] tlev_in = np.zeros((ncols,nlevels), dtype=np.double, order='F')
            double [:] tsfc_in = np.ones((ncols),dtype=np.double,order='F') * self.sst
            double [:,:] h2ovmr_in = np.zeros((ncols, nlayers),dtype=np.double,order='F')
            double [:,:] o3vmr_in  = np.array(np.expand_dims(self.o3vmr,axis=0),dtype=np.double,order='F')
            double [:,:] co2vmr_in = np.array(np.expand_dims(self.co2vmr,axis=0),dtype=np.double,order='F')
            double [:,:] ch4vmr_in = np.array(np.expand_dims(self.ch4vmr,axis=0),dtype=np.double,order='F')
            double [:,:] n2ovmr_in = np.array(np.expand_dims(self.n2ovmr,axis=0),dtype=np.double,order='F')
            double [:,:] o2vmr_in  = np.array(np.expand_dims(self.o2vmr,axis=0),dtype=np.double,order='F')
            double [:,:] cfc11vmr_in = np.array(np.expand_dims(self.cfc11vmr,axis=0),dtype=np.double,order='F')
            double [:,:] cfc12vmr_in = np.array(np.expand_dims(self.cfc12vmr,axis=0),dtype=np.double,order='F')
            double [:,:] cfc22vmr_in = np.array(np.expand_dims(self.cfc22vmr,axis=0),dtype=np.double,order='F')
            double [:,:] ccl4vmr_in = np.array(np.expand_dims(self.ccl4vmr,axis=0),dtype=np.double,order='F')
            double [:,:] emis_in = np.ones((ncols, 16),dtype=np.double,order='F') * 0.95
            double [:,:] cldfr_in  = np.zeros((ncols,nlayers), dtype=np.double,order='F')
            double [:,:] cicewp_in = np.zeros((ncols,nlayers),dtype=np.double,order='F')
            double [:,:] cliqwp_in = np.zeros((ncols,nlayers),dtype=np.double,order='F')
            double [:,:] reice_in  = np.zeros((ncols,nlayers),dtype=np.double,order='F')
            double [:,:] reliq_in  = np.ones((ncols,nlayers),dtype=np.double,order='F') * 2.5
            double [:] coszen_in = np.ones(ncols,dtype=np.double,order='F') *self.coszen
            double [:] asdir_in = np.ones(ncols,dtype=np.double,order='F') * self.adir
            double [:] asdif_in = np.ones(ncols,dtype=np.double,order='F') * self.adif
            double [:] aldir_in = np.ones(ncols,dtype=np.double,order='F') * self.adir
            double [:] aldif_in = np.ones(ncols,dtype=np.double,order='F') * self.adif
            double [:,:,:] taucld_lw_in  = np.zeros((16,ncols,nlayers),dtype=np.double,order='F')
            double [:,:,:] tauaer_lw_in  = np.zeros((ncols,nlayers,16),dtype=np.double,order='F')
            double [:,:,:] taucld_sw_in  = np.zeros((14,ncols,nlayers),dtype=np.double,order='F')
            double [:,:,:] ssacld_sw_in  = np.zeros((14,ncols,nlayers),dtype=np.double,order='F')
            double [:,:,:] asmcld_sw_in  = np.zeros((14,ncols,nlayers),dtype=np.double,order='F')
            double [:,:,:] fsfcld_sw_in  = np.zeros((14,ncols,nlayers),dtype=np.double,order='F')
            double [:,:,:] tauaer_sw_in  = np.zeros((ncols,nlayers,14),dtype=np.double,order='F')
            double [:,:,:] ssaaer_sw_in  = np.zeros((ncols,nlayers,14),dtype=np.double,order='F')
            double [:,:,:] asmaer_sw_in  = np.zeros((ncols,nlayers,14),dtype=np.double,order='F')
            double [:,:,:] ecaer_sw_in  = np.zeros((ncols,nlayers,6),dtype=np.double,order='F')

            # Output
            double[:,:] uflx_lw_out = np.zeros((ncols,nlevels),dtype=np.double,order='F')
            double[:,:] dflx_lw_out = np.zeros((ncols,nlevels),dtype=np.double,order='F')
            double[:,:] hr_lw_out = np.zeros((ncols,nlayers),dtype=np.double,order='F')
            double[:,:] uflxc_lw_out = np.zeros((ncols,nlevels),dtype=np.double,order='F')
            double[:,:] dflxc_lw_out = np.zeros((ncols,nlevels),dtype=np.double,order='F')
            double[:,:] hrc_lw_out = np.zeros((ncols,nlayers),dtype=np.double,order='F')
            double[:,:] duflx_dt_out = np.zeros((ncols,nlevels),dtype=np.double,order='F')
            double[:,:] duflxc_dt_out = np.zeros((ncols,nlevels),dtype=np.double,order='F')
            double[:,:] uflx_sw_out = np.zeros((ncols,nlevels),dtype=np.double,order='F')
            double[:,:] dflx_sw_out = np.zeros((ncols,nlevels),dtype=np.double,order='F')
            double[:,:] hr_sw_out = np.zeros((ncols,nlayers),dtype=np.double,order='F')
            double[:,:] uflxc_sw_out = np.zeros((ncols,nlevels),dtype=np.double,order='F')
            double[:,:] dflxc_sw_out = np.zeros((ncols,nlevels),dtype=np.double,order='F')
            double[:,:] hrc_sw_out = np.zeros((ncols,nlayers),dtype=np.double,order='F')

            double rv_to_reff = np.exp(np.log(1.2)**2.0)*10.0*1000.0

        with nogil:
            tlev_in[0,0] = self.sst
            for k in xrange(1,nlayers):
                tlev_in[0,k] = 0.5 * (tlay_in[0,k-1]+tlay_in[0,k])
            tlev_in[0, nlayers] = 2.0 * tlay_in[0,nlayers-1] - tlev_in[0,nlayers-1]
            for k in xrange(nlayers):
                h2ovmr_in[0,k] = self.qv_layers[k]/(1.0 - self.qv_layers[k]) * Rv/Rd




        cdef:
            int ncol = ncols
            int nlay = nlayers
            int icld = 1
            int idrv = 0
            int iaer = 0
            int inflglw = 2
            int iceflglw = 3
            int liqflglw = 1
            int inflgsw = 2
            int iceflgsw = 3
            int liqflgsw = 1

        c_rrtmg_lw (
            &ncol, &nlay, &icld, &idrv, &play_in[0,0], &plev_in[0,0], &tlay_in[0,0], &tlev_in[0,0], &tsfc_in[0],
            &h2ovmr_in[0,0], &o3vmr_in[0,0], &co2vmr_in[0,0], &ch4vmr_in[0,0], &n2ovmr_in[0,0], &o2vmr_in[0,0],
            &cfc11vmr_in[0,0], &cfc12vmr_in[0,0], &cfc22vmr_in[0,0], &ccl4vmr_in[0,0], &emis_in[0,0], &inflglw,
            &iceflglw,&liqflglw, &cldfr_in[0,0], &taucld_lw_in[0,0,0], &cicewp_in[0,0], &cliqwp_in[0,0], &reice_in[0,0],
            &reliq_in[0,0], &tauaer_lw_in[0,0,0], &uflx_lw_out[0,0], &dflx_lw_out[0,0], &hr_lw_out[0,0],
            &uflxc_lw_out[0,0], &dflxc_lw_out[0,0], &hrc_lw_out[0,0], &duflx_dt_out[0,0], &duflxc_dt_out[0,0] )

        c_rrtmg_sw (
            &ncol, &nlay, &icld, &iaer, &play_in[0,0], &plev_in[0,0], &tlay_in[0,0], &tlev_in[0,0],&tsfc_in[0],
            &h2ovmr_in[0,0], &o3vmr_in[0,0], &co2vmr_in[0,0], &ch4vmr_in[0,0], &n2ovmr_in[0,0],&o2vmr_in[0,0],
            &asdir_in[0], &asdif_in[0], &aldir_in[0], &aldif_in[0], &coszen_in[0], &self.adjes, &self.dyofyr,
            &self.scon, &inflgsw, &iceflgsw, &liqflgsw, &cldfr_in[0,0], &taucld_sw_in[0,0,0], &ssacld_sw_in[0,0,0],
            &asmcld_sw_in[0,0,0], &fsfcld_sw_in[0,0,0], &cicewp_in[0,0], &cliqwp_in[0,0], &reice_in[0,0], &reliq_in[0,0],
            &tauaer_sw_in[0,0,0], &ssaaer_sw_in[0,0,0], &asmaer_sw_in[0,0,0], &ecaer_sw_in[0,0,0], &uflx_sw_out[0,0],
            &dflx_sw_out[0,0], &hr_sw_out[0,0], &uflxc_sw_out[0,0], &dflxc_sw_out[0,0], &hrc_sw_out[0,0])



        with nogil:
            for k in xrange(nlayers):
                self.dTdt_rad_lw[k] = (hr_lw_out[0,k] )/86400.0
                self.dTdt_rad_sw[k] = (hr_sw_out[0,k]) /86400.0
            for k in xrange(nlevels):
                self.uflux_lw[k] = uflx_lw_out[0,k]
                self.uflux_sw[k] = uflx_sw_out[0,k]
                self.dflux_lw[k] = dflx_lw_out[0,k]
                self.dflux_sw[k] = dflx_sw_out[0,k]

        return

    cdef lapse_rate(self, double p, double T, double *qt ):
        if self.lapse_rate_type == 'saturated':
            return self.lapse_rate_saturated(p,T, &qt[0])
        elif self.lapse_rate_type =='subsaturated':
            return self.lapse_rate_subsaturated(p,T,&qt[0])
        else:
            return self.lapse_rate_mixed(p,T,&qt[0])


    cdef lapse_rate_saturated(self, double p, double T, double *qt ):
        cdef:
            double esoverp, Lambda, L, ratio, dTdp, rho
            double pv_star, pv, gamma, qv_star


        pv_star = self.CC.LT.fast_lookup(T)
        pv = fmin(pv_c(p,qt[0], qt[0]),p)
        esoverp = pv_star/p
        qv_star = qv_star_c(p,qt[0],pv_star)
        Lambda = self.Lambda_fp(T)
        L = self.L_fp(T, Lambda)
        ratio = L/T/Rv
        dTdp = (T / p * kappa * (1 + esoverp * ratio)  / (1 + kappa * (cpv / Rv + (ratio-1) * ratio) * esoverp))
        rho = p /T/ (Rd*(1.0-qv_star)+Rv*qv_star)
        gamma = dTdp * rho * g
        qt[0] = qv_star

        return gamma


    cdef lapse_rate_subsaturated(self, double p, double T, double *qt):
        cdef:
            double esoverp, Lambda, L, ratio, dTdp, rho
            double pv_star, pv, gamma, qv_star

        gamma= (1.0 + qt[0])/(1.0+ qt[0] * cpv/cpd) * g/cpd

        return gamma


    cdef lapse_rate_mixed(self, double p, double T, double *qt):
        cdef:
            double esoverp, Lambda, L, ratio, dTdp, rho
            double pv_star, pv, gamma, qv_star


        pv_star = self.CC.LT.fast_lookup(T)
        pv = fmin(pv_c(p,qt[0], qt[0]),p)
        if pv_star > pv:
            gamma= (1.0 + qt[0])/(1.0+ qt[0] * cpv/cpd) * g/cpd
        else:
            esoverp = pv_star/p
            qv_star = qv_star_c(p,qt[0],pv_star)
            Lambda = self.Lambda_fp(T)
            L = self.L_fp(T, Lambda)
            ratio = L/T/Rv
            dTdp = (T / p * kappa * (1 + esoverp * ratio)  / (1 + kappa * (cpv / Rv + (ratio-1) * ratio) * esoverp))
            rho = p /T/ (Rd*(1.0-qv_star)+Rv*qv_star)
            gamma = dTdp * rho * g
            qt[0] = qv_star

        return gamma

    cpdef convective_adjustment(self):
        cdef:
            Py_ssize_t k, l, i
            Py_ssize_t nlv = self.nlevels
            double [:] T_l =np.zeros(nlv,dtype=np.double, order='c')
            double [:] p_l = np.zeros(nlv,dtype=np.double, order='c')
            double [:] q_l = np.zeros(nlv,dtype=np.double, order='c')
            double [:] Pi_l = np.zeros(nlv,dtype=np.double, order='c')
            double [:] beta_l = np.zeros(nlv,dtype=np.double, order='c')
            double [:] alpha_l = np.zeros(nlv,dtype=np.double, order='c')
            Py_ssize_t [:] n_k = np.zeros(nlv, dtype=np.int, order='c')
            double [:] s_k =np.zeros(nlv,dtype=np.double, order='c')
            double [:] t_k = np.zeros(nlv,dtype=np.double, order='c')
            double [:] theta_k = np.zeros(nlv,dtype=np.double, order='c')
            double [:] theta_new = np.zeros(nlv,dtype=np.double, order='c')
            double gamma_l, theta
            bint done = False
            Py_ssize_t n, count
            double dp = self.p_layers[0] - self.p_layers[1]
            double Rgas
            double pvg = self.CC.LT.fast_lookup(self.sst) * self.RH_surface
            double qtg = eps_v * pvg / (self.p_surface + (eps_v-1.0)*pvg)


        T_l[0] = self.sst
        p_l[0] = self.p_surface
        T_l[1:] = self.t_layers[0:]
        p_l[1:] = self.p_layers[0:]


        # Set up the array "beta"

        gamma_l = self.lapse_rate(p_l[0], T_l[0], &qtg)
        Rgas = Rd * (1.0-qtg) + Rv * qtg
        alpha_l[0] = Rgas * gamma_l/g
        # set Pi_l[0] = 1 (see equation 14, with indices shifted by 1)
        Pi_l[0] = 1.0
        beta_l[0] = 1/Pi_l[0]

        for k in xrange(1,nlv):
            gamma_l = self.lapse_rate(p_l[k], T_l[k], &qtg)
            Rgas = Rd * (1.0-qtg) + Rv * qtg
            alpha_l[k] = Rgas * gamma_l/g
            Pi_l[k] =  Pi_l[k-1] * np.power(p_l[k]/p_l[k-1],alpha_l[k-1])
            beta_l[k] = 1.0/Pi_l[k]

        q_l[0] = Pi_l[0] * 1.0 * 4.19e3 * 1000.0
        q_l[1] = Pi_l[1] * cpd/g * dp/2.0
        q_l[nlv-1] = Pi_l[nlv-1] * cpd * g * dp/2.0
        for k in xrange(2,nlv-1):
            q_l[k] = Pi_l[k] * cpd/g * dp


        # algorithm step 1 (remember all indices must be shifted by 1)
        k = 0
        n_k[0] = 1
        theta_k[0] = beta_l[0] * T_l[0]
        for l in xrange(1,nlv):
            #algorithm step 2
            n=1
            theta = beta_l[l] * T_l[l]
            done = False
            while not done:
                if theta_k[k] > theta: # unstable stratification
                    if n==1:
                        # algorithm step 3
                        s = q_l[l]
                        t =s * theta
                    if n_k[k] < 2:
                        # algorith step 4
                        # lower adjacent level is not an earlier-formed neutral layer
                        s_k[k] = q_l[l-n]
                        t_k[k] = s_k[k] * theta_k[k]
                    # algorithm step 5
                    n += n_k[k]
                    s += s_k[k]
                    s_k[k] = s
                    t += t_k[k]
                    t_k[k] = t
                    theta = t/s
                    if k==0:
                        #done checking the layers
                        done = True
                    else:
                        # go back through
                        k -= 1
                else: # stable stratification, move to next k
                    k+=1
                    done = True
                n_k[k] = n
                theta_k[k] = theta

        # update the temperatures
        count = 0
        for i in xrange(nlv):
            for k in xrange(n_k[i]):
                theta_new[count] = theta_k[i]
                count=np.minimum(count+1, nlv-1)
        self.sst = theta_new[0] * Pi_l[0]
        for i in xrange(1,nlv):
            self.t_layers[i-1] = theta_new[i] * Pi_l[i]


        return

    cpdef rce_fixed_toa(self, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t k, iter=0
            Py_ssize_t nly = self.nlayers
            double  net_surface, net_toa_old
            double slab_capacity = 1.0 * 1000.0 * 4.19e3
            double [:] T_old = np.zeros(self.nlevels, dtype=np.double, order='c')
            double [:] T_new= np.zeros(self.nlevels, dtype=np.double, order='c')
            double dt_rce_original = self.dt_rce
            double sst_original = self.sst
            double [:] t_layers_original = np.array(self.t_layers, copy=True)
            double [:] qv_layers_original=np.array(self.qv_layers, copy=True)
            bint converged = False

        while self.dt_rce > 1800.0 and not converged:
            self.net_toa_computed = 1000.0
            self.delta_T = self.delta_T_max * 100
            iter = 0
            self.sst = sst_original
            for k in xrange(self.nlayers):
                self.qv_layers[k] = qv_layers_original[k]
                self.t_layers[k] = t_layers_original[k]
            while iter < self.max_steps:
                if self.delta_T < self.delta_T_max  and np.abs(self.net_toa_target-self.net_toa_computed) < self.toa_error_max:
                    converged=True
                    plt.figure('Converged')
                    plt.plot(self.t_layers,self.p_layers)
                    plt.plot(self.sst, self.p_surface,'o')
                    plt.gca().invert_yaxis()
                    plt.show()
                    Pa.root_print('DONE! RCE converged at  '+str(iter))
                    Pa.root_print('--net_toa_target  '+ str(self.net_toa_target))
                    Pa.root_print('--net_toa_computed  '+str(self.net_toa_computed))
                    Pa.root_print('--ohu  '+str(self.ohu))
                    Pa.root_print('--sst ' + str(self.sst))
                    Pa.root_print('--delta T '+str(self.delta_T))
                    Pa.root_print('--dt_rce '+ str(self.dt_rce))
                    break

                T_old[0] = self.sst
                T_old[1:] = self.t_layers[0:]

                #update temperatures due to radiation
                self.compute_radiation()

                net_toa_old = self.net_toa_computed
                self.net_toa_computed = (self.dflux_sw[nly] - self.uflux_sw[nly]
                           - (self.uflux_lw[nly] - self.dflux_lw[nly]))
                net_surface = (self.dflux_sw[0] - self.uflux_sw[0]
                           - (self.uflux_lw[0] - self.dflux_lw[0]))

                self.sst += (net_surface-self.ohu)  * self.dt_rce/slab_capacity

                if self.net_toa_computed > self.net_toa_target and self.net_toa_computed> net_toa_old:
                    self.ohu += (self.net_toa_target-self.net_toa_computed)/slab_capacity * self.dt_rce
                elif self.net_toa_computed < self.net_toa_target and self.net_toa_computed < net_toa_old:
                    self.ohu += (self.net_toa_target-self.net_toa_computed)/slab_capacity * self.dt_rce


                if self.verbose:
                    Pa.root_print('iter, sst, net_toa '+ str(iter) + ', '
                                  + str(np.round(self.sst,4)) +  ',  '
                                  + str(np.round(self.net_toa_computed,4)) + ',  '
                                  + str(np.round(self.net_toa_target,4)))
                    Pa.root_print('new ohu '+ str(np.round(self.ohu,4)))

                for k in xrange(nly):
                    self.t_layers[k] += (self.dTdt_rad_lw[k] + self.dTdt_rad_sw[k]) * self.dt_rce

                self.convective_adjustment()

                self.qv_layers[0] = self.update_qv(self.p_layers[0],self.t_layers[0],self.RH_tropical, 1.0)
                for k in xrange(1,nly):
                    self.qv_layers[k] = self.update_qv(self.p_layers[k], self.t_layers[k],self.RH_tropical, self.qv_layers[k-1])

                T_new[0] = self.sst
                T_new[1:] = self.t_layers[0:]
                self.delta_T = np.amax(np.abs(np.subtract(T_new,T_old)))/self.dt_rce

                iter +=1
            if not converged:
                Pa.root_print('FAIL! Not converged with dt_rce ' + str(self.dt_rce))
                Pa.root_print('Performed  '+str(iter) + ' iterations')
                Pa.root_print('--net_toa_target  '+ str(self.net_toa_target))
                Pa.root_print('--net_toa_computed  '+str(self.net_toa_computed))
                Pa.root_print('--ohu  '+str(self.ohu))
                Pa.root_print('--sst ' + str(self.sst))
                Pa.root_print('--delta T '+str(self.delta_T))
                Pa.root_print('-- reducing dt_rce '+ str(self.dt_rce*0.75))
                self.dt_rce *=  0.75

        self.dt_rce = dt_rce_original

        return





    cpdef initialize(self, Grid Gr, ParallelMPI.ParallelMPI Pa, NetCDFIO_Stats NS, double  S_minus_L):
        if self.is_init:
            return

        cdef:
            double pv, pd
            Py_ssize_t k, index_h
            double maxval = 1.0
            Py_ssize_t nly = self.nlayers


        # pressure coordinate
        self.p_levels = np.linspace(self.p_surface, 0.0, num=self.nlevels, endpoint=True)
        self.p_layers = 0.5 * np.add(self.p_levels[1:],self.p_levels[:-1])
        self.pressure = self.p_layers
        ForcingReferenceBase.initialize(self, Gr, Pa, NS, S_minus_L)

        self.initialize_radiation()

        if not self.read_pkl:

            # THESE CONTAIN THE ACTUAL RCE SOLUTION
            self.t_layers = np.zeros(self.nlayers, dtype=np.double, order='c')
            self.qv_layers =np.zeros(self.nlayers, dtype=np.double, order='c')
            # Set up initial guesses
            self.sst = 325.0
            self.initialize_adiabat()
            k = 0
            while self.p_layers[k] > 300.0e2:
                index_h = k
                k+=1
            for k in xrange(index_h, self.nlayers):
                self.t_layers[k] = self.t_layers[index_h]
                self.qv_layers[k] =self.qv_layers[index_h]

            # Might as well do it on every processor, rather than communicate?
            # Solution should be unique...
            self.net_toa_target = S_minus_L
            self.ohu = S_minus_L
            self.rce_fixed_toa(Pa)
            Pa.root_print('Success! RCE converged.')
            Pa.root_print('net_toa_target  '+ str(self.net_toa_target))
            Pa.root_print('net_toa_computed  '+str(self.net_toa_computed))
            Pa.root_print('ohu  '+str(self.ohu))
            Pa.root_print('sst ' + str(self.sst))
            Pa.root_print('delta T '+str(self.delta_T))
            Pa.root_print('TOA_lw_down ' + str(np.round(self.dflux_lw[self.nlayers],2)))
            Pa.root_print('TOA_lw_up ' + str(np.round(self.uflux_lw[self.nlayers],4)))
            Pa.root_print('TOA_sw_down ' + str(np.round(self.dflux_sw[self.nlayers],4)))
            Pa.root_print('TOA_sw_up ' + str(np.round(self.uflux_sw[self.nlayers],4)))


            if Pa.rank==0:
                dict = {}
                dict['nlayers'] = self.nlayers
                dict['delta_T'] = self.delta_T
                dict['T_profile'] = np.asarray(self.t_layers)
                dict['qv_profile'] = np.asarray(self.qv_layers)
                dict['p_profile'] = np.asarray(self.p_layers)
                dict['p_tropo'] = self.p_layers[np.argmin(self.t_layers)]
                dict['sst'] = self.sst
                dict['ohu'] = self.ohu
                dict['S_minus_L'] = self.net_toa_computed
                dict['TOA_lw_down']= self.dflux_lw[nly]
                dict['TOA_lw_up']= self.uflux_lw[nly]
                dict['TOA_sw_down'] = self.dflux_sw[nly]
                dict['TOA_sw_up']= self.uflux_sw[nly]
                dict['surface_lw_down']= self.dflux_lw[0]
                dict['surface_lw_up']= self.uflux_lw[0]
                dict['surface_sw_down'] = self.dflux_sw[0]
                dict['surface_sw_up']= self.uflux_sw[0]
                pickle.dump(dict, open(self.out_dir+'/IRCE_TOA_'
                                       +str(np.round(self.net_toa_target,2)) +'_'+str(self.co2_factor)+'xCO2.pkl', "wb"  ))

            #################################################################
        else:

            file_handle = open(self.pkl_file, 'rb')
            read_dict = pickle.load(file_handle)
            if read_dict['nlayers'] != self.nlayers:
                Pa.root_print('Dict nlayers does not equal namelist nlayers!')
                Pa.kill()
            self.t_layers = read_dict['T_profile']
            self.qv_layers = read_dict['qv_profile']
            self.sst = read_dict['sst']

            # Now run radiation on the read-in profile
            self.compute_radiation()
            self.net_toa_computed = (self.dflux_sw[nly] - self.uflux_sw[nly]
                       - (self.uflux_lw[nly] - self.dflux_lw[nly]))
            self.net_toa_target = self.net_toa_computed
            self.ohu = (self.dflux_sw[0] - self.uflux_sw[0]
                       - (self.uflux_lw[0] - self.dflux_lw[0]))
            Pa.root_print('Read in pkl, computed TOA imbalance is '+str(np.round(self.net_toa_computed,5)))

        for k in xrange(self.npressure):
            self.temperature[k] = self.t_layers[k]
            if k >0:
                maxval = self.qt[k-1]
            self.qt[k] = self.update_qv(self.pressure[k],self.temperature[k],self.RH_subtrop,maxval)
            self.s[k] = self.entropy(self.pressure[k], self.temperature[k], self.qt[k], 0.0, 0.0)
            self.rv[k] = self.qt[k]/(1.0-self.qt[k])
            self.u[k] = fmin(-10.0 + (-7.0-(-10.0))/(750.0e2-1000.0e2)*(self.pressure[k]-1000.0e2),-4.0)

        NS.add_ts('tropical_sst', Gr, Pa)
        NS.add_ts('tropical_toa_imbalance', Gr, Pa)
        self.is_init = True

        return

    cpdef update_qv(self, double p, double t, double rh, double maxval):
        cdef double pv, pd, qv
        pv = self.CC.LT.fast_lookup(t) * rh
        pd = p - pv
        qv = fmin(pv/(pd * eps_vi + pv),maxval)
        return  qv

    cpdef initialize_adiabat(self):
        cdef:
            double pvg = self.CC.LT.fast_lookup(self.sst) * self.RH_tropical
            double qtg = eps_v * pvg / (self.p_surface + (eps_v-1.0)*pvg)
            double sg = self.entropy(self.p_surface, self.sst, qtg, 0.0, 0.0)
            double maxval = 1.0
            double temperature, ql, qi, pv
            Py_ssize_t k

        # Compute reference state thermodynamic profiles
        for k in xrange(self.nlayers):
            temperature, ql, qi = self.eos(self.p_layers[k], sg, qtg)
            qtg = fmax(qtg-ql, 1e-10)

            if np.isnan(temperature):
                self.t_layers[k] = 2.0 * self.t_layers[k-1] - self.t_layers[k-2]
            else:
                self.t_layers[k] = temperature
            if k > 0:
                maxval = self.qv_layers[k-1]
            self.qv_layers[k] = self.update_qv(self.p_layers[k], self.t_layers[k], self.RH_tropical, maxval)

        return

    cpdef update(self, ParallelMPI.ParallelMPI Pa,   double S_minus_L, TimeStepping TS):

        # Update with given timescale
        self.net_toa_target +=  (S_minus_L-self.net_toa_target)/self.toa_update_timescale * TS.dt * TS.acceleration_factor
        # Pa.root_print('net_toa_target '+str(np.round(self.net_toa_target,6))
        #               + ' net_toa_computed '+str(np.round(self.net_toa_computed,6)))



        # HACK TO Force convergence to prescribed, fixed S_minus_L
        if not self.adjust_S_minus_L:
            if fabs(self.net_toa_target-S_minus_L) < self.toa_update_criterion :
                self.net_toa_target = S_minus_L
                if fabs(self.net_toa_computed - self.net_toa_target) > self.toa_error_max*1.01:
                    self.net_toa_computed = 1000.0
                    Pa.root_print('Adjusted net_toa_computed for gap fix!')


        # check the change in S_minus_L
        if fabs(self.net_toa_computed - self.net_toa_target) < self.toa_update_criterion:
            return
        Pa.root_print('Updating Reference Profile for '+str(np.round(self.net_toa_target,5)))
        Pa.root_print('Computed toa was '+str(np.round(self.net_toa_computed,5)))

        self.ohu = self.net_toa_target
        self.rce_fixed_toa(Pa)
        Pa.root_print('Success! RCE converged.')
        Pa.root_print('net_toa_target  '+ str(self.net_toa_target))
        Pa.root_print('net_toa_computed  '+str(self.net_toa_computed))
        Pa.root_print('ohu  '+str(self.ohu))
        Pa.root_print('sst ' + str(self.sst))
        Pa.root_print('delta T '+str(self.delta_T))
        Pa.root_print('TOA_lw_down ' + str(np.round(self.dflux_lw[self.nlayers],2)))
        Pa.root_print('TOA_lw_up ' + str(np.round(self.uflux_lw[self.nlayers],4)))
        Pa.root_print('TOA_sw_down ' + str(np.round(self.dflux_sw[self.nlayers],4)))
        Pa.root_print('TOA_sw_up ' + str(np.round(self.uflux_sw[self.nlayers],4)))

        cdef:
            Py_ssize_t nly = self.nlayers
            double maxval = 1.0
        if Pa.rank==0:
            dict = {}
            dict['nlayers'] = self.nlayers
            dict['delta_T'] = self.delta_T
            dict['T_profile'] = np.asarray(self.t_layers)
            dict['qv_profile'] = np.asarray(self.qv_layers)
            dict['p_profile'] = np.asarray(self.p_layers)
            dict['p_tropo'] = self.p_layers[np.argmin(self.t_layers)]
            dict['sst'] = self.sst
            dict['ohu'] = self.ohu
            dict['S_minus_L'] = self.net_toa_computed
            dict['TOA_lw_down']= self.dflux_lw[nly]
            dict['TOA_lw_up']= self.uflux_lw[nly]
            dict['TOA_sw_down'] = self.dflux_sw[nly]
            dict['TOA_sw_up']= self.uflux_sw[nly]
            dict['surface_lw_down']= self.dflux_lw[0]
            dict['surface_lw_up']= self.uflux_lw[0]
            dict['surface_sw_down'] = self.dflux_sw[0]
            dict['surface_sw_up']= self.uflux_sw[0]
            pickle.dump(dict, open(self.out_dir+'/IRCE_TOA_'+str(np.round(self.net_toa_target,2)) +'_'+str(self.co2_factor)+'xCO2.pkl', "wb"  ))



            #################################################################

        # OLD--DOING INTERP
        # self.temperature = np.array(np.interp(pressure_array, self.p_layers[::-1], self.t_layers[::-1]), dtype=np.double, order='c')
        # self.qt = np.array(np.interp(pressure_array, self.p_layers[::-1], self.qv_layers[::-1]), dtype=np.double, order='c')
        # cdef:
        #     Py_ssize_t nz = np.shape(pressure_array)[0]
        #
        # for k in xrange(nz):
        #     self.s[k] = self.entropy(pressure_array[k], self.temperature[k], self.qt[k], 0.0, 0.0)
        #     self.rv[k] = self.qt[k]/(1.0-self.qt[k])
        #     self.u[k] = fmin(-10.0 + (-7.0-(-10.0))/(750.0e2-1000.0e2)*(pressure_array[k]-1000.0e2),-4.0)

        # NEW

        # NEW
        for k in xrange(self.npressure):
            self.temperature[k] = self.t_layers[k]
            if k > 0:
                maxval = self.qt[k-1]
            self.qt[k] = self.update_qv(self.pressure[k],self.temperature[k],self.RH_subtrop,maxval)
            self.s[k] = self.entropy(self.pressure[k], self.temperature[k], self.qt[k], 0.0, 0.0)
            self.rv[k] = self.qt[k]/(1.0-self.qt[k])
            # self.u[k] = fmin(-10.0 + (-7.0-(-10.0))/(750.0e2-1000.0e2)*(self.pressure[k]-1000.0e2),-4.0)

        return

    cpdef stats_io(self, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        NS.write_ts('tropical_sst', self.sst, Pa)
        NS.write_ts('tropical_toa_imbalance', self.net_toa_computed, Pa)
        return

    cpdef restart(self, Restart):

        Restart.restart_rce['nlayers'] = self.nlayers
        Restart.restart_rce['delta_T'] = self.delta_T
        Restart.restart_rce['T_profile'] = np.asarray(self.t_layers)
        Restart.restart_rce['qv_profile'] = np.asarray(self.qv_layers)
        Restart.restart_rce['p_profile'] = np.asarray(self.p_layers)
        Restart.restart_rce['p_tropo'] = self.p_layers[np.argmin(self.t_layers)]
        Restart.restart_rce['sst'] = self.sst
        Restart.restart_rce['ohu'] = self.ohu
        Restart.restart_rce['S_minus_L'] = self.net_toa_computed
        Restart.restart_rce['TOA_lw_down']= self.dflux_lw[self.nlayers]
        Restart.restart_rce['TOA_lw_up']= self.uflux_lw[self.nlayers]
        Restart.restart_rce['TOA_sw_down'] = self.dflux_sw[self.nlayers]
        Restart.restart_rce['TOA_sw_up']= self.uflux_sw[self.nlayers]
        Restart.restart_rce['surface_lw_down']= self.dflux_lw[0]
        Restart.restart_rce['surface_lw_up']= self.uflux_lw[0]
        Restart.restart_rce['surface_sw_down'] = self.dflux_sw[0]
        Restart.restart_rce['surface_sw_up']= self.uflux_sw[0]
        return



cdef class LookupProfiles:
    def __init__(self, Py_ssize_t nprofiles, Py_ssize_t nz):
        self.nprofiles = nprofiles
        self.nz = nz
        self.table_vals = np.zeros((nprofiles,nz),dtype=np.double, order='c')
        self.access_vals = np.zeros(nprofiles, dtype=np.double, order='c')
        self.profile_interp = np.zeros(nz, dtype=np.double, order='c')
        return

    cpdef lookup(self, double val):
        cdef:
            double min_ = self.access_vals[0]
            double max_ = self.access_vals[self.nprofiles-1]
            double del_ = self.access_vals[1] - self.access_vals[0]
            Py_ssize_t indx = int(np.floor((val - min_)/del_))
            double x1 = self.access_vals[indx]
            double y1, y2
            Py_ssize_t k

        with nogil:
            for k in xrange(self.nz):
                y1 = self.table_vals[indx,k]
                y2 = self.table_vals[indx+1, k]
                self.profile_interp[k] = y1 + (val - x1) * (y2 - y1)/del_
        return

    cpdef communicate(self,  ParallelMPI.ParallelMPI Pa):
        cdef:
            double [:] global_vals
            Py_ssize_t iprof, k

        for iprof in xrange(self.nprofiles):
           global_vals = Pa.domain_vector_sum(self.table_vals[iprof,:], self.nz)
           with nogil:
            for k in xrange(self.nz):
                self.table_vals[iprof,k] = global_vals[k]
        return





