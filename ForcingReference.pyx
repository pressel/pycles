#!python
#cython: boundscheck=False
#cython: initializedcheck=False
#cython: cdivision=True

import netCDF4 as nc
from thermodynamic_functions cimport cpm_c, pv_c, pd_c, exner_c
from entropies cimport sv_c, sd_c, s_tendency_c
import numpy as np
import cython
from libc.math cimport fabs, sin, cos, fmax, fmin, log
from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI
cimport Lookup
from Thermodynamics cimport LatentHeat, ClausiusClapeyron

import pylab as plt
include 'parameters.pxi'



cdef extern from "thermodynamics_sa.h":
    void eos_c(Lookup.LookupStruct *LT, double(*lam_fp)(double), double(*L_fp)(double, double), double p0, double s, double qt, double *T, double *qv, double *ql, double *qi) nogil
cdef extern from "thermodynamic_functions.h":
    inline double pd_c(double p0, double qt, double qv) nogil
    inline double pv_c(double p0, double qt, double qv) nogil
cdef extern from "entropies.h":
    inline double sd_c(double pd, double T) nogil
    inline double sv_c(double pv, double T) nogil
    inline double sc_c(double L, double T) nogil

# These classes compute or read in the reference profiles needed for ZGILS cases
# The base class
cdef class ForcingReferenceBase:
    def __init__(self):
        return
    cpdef initialize(self,  ParallelMPI.ParallelMPI Pa, double [:] pressure_array, double Pg, double Tg, double RH):
        cdef Py_ssize_t nz = len(pressure_array)
        self.s = np.zeros(nz, dtype=np.double, order='c')
        self.qt = np.zeros(nz, dtype=np.double, order='c')
        self.temperature = np.zeros(nz, dtype=np.double, order='c')
        self.rv = np.zeros(nz, dtype=np.double, order='c')
        self.u = np.zeros(nz, dtype=np.double, order='c')
        self.v = np.zeros(nz, dtype=np.double, order='c')
        return

# Climate change simulations use profiles based on radiative--convective equilibrium solutions obtained as described in
# Zhihong Tan's dissertation (Section 2.6). Zhihong has provided his reference profiles to be archived with the code, so
# this class just reads in the data and interpolates to the simulation pressure grid

cdef class ReferenceRCE(ForcingReferenceBase):
    def __init__(self, filename):
        self.filename = filename
        return
    @cython.wraparound(True)
    cpdef initialize(self,  ParallelMPI.ParallelMPI Pa, double [:] pressure_array,double Pg, double Tg, double RH):

        ForcingReferenceBase.initialize(self,Pa, pressure_array,  Pg, Tg, RH)
        data = nc.Dataset(self.filename, 'r')
        # Arrays must be flipped (low to high pressure) to use numpy interp function
        pressure_ref = data.variables['p_full'][::-1]
        temperature_ref = data.variables['temp_rc'][::-1]
        qt_ref = data.variables['yv_rc'][::-1]
        u_ref = data.variables['u'][::-1]
        v_ref = data.variables['v'][::-1]

        self.temperature = np.array(np.interp(pressure_array, pressure_ref, temperature_ref),
                                    dtype=np.double, order='c')
        self.qt = np.array(np.interp(pressure_array, pressure_ref, qt_ref), dtype=np.double, order='c')
        self.u = np.array(np.interp(pressure_array, pressure_ref, u_ref), dtype=np.double, order='c')
        self.v = np.array(np.interp(pressure_array, pressure_ref, v_ref), dtype=np.double, order='c')


        cdef:
            double pd, pv
            Py_ssize_t k
        # computing entropy assuming sub-saturated
        for k in xrange(len(pressure_array)):
            pv = pv_c(pressure_array[k], self.qt[k], self.qt[k])
            pd = pd_c(pressure_array[k], self.qt[k], self.qt[k])

            self.rv[k] =  self.qt[k]/(1.0-self.qt[k])
            self.s[k] = (sd_c(pd, self.temperature[k]) * (1.0 - self.qt[k])
                         + sv_c(pv, self.temperature[k]) * self.qt[k])
            self.u[k] = self.u[k]*0.5 - 5.0

        return

# Here we implement the RCE solution algorithm described in Section 2.6 of Zhihong Tan's thesis to allow updates
# of the Reference profiles as the SST changes
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

cdef class InteractiveReferenceRCE(ForcingReferenceBase):
    def __init__(self,namelist,  LatentHeat LH, ParallelMPI.ParallelMPI Pa ):
        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp
        self.CC = ClausiusClapeyron()
        self.CC.initialize(namelist, LH, Pa)

        # Radiation parameters
        #--Namelist options related to gas concentrations
        try:
            self.co2_factor = namelist['radiation']['RRTM']['co2_factor']
        except:
            self.co2_factor = 1.0
        #--Namelist options related to insolation
        try:
            self.dyofyr = namelist['radiation']['RRTM']['dyofyr']
        except:
            self.dyofyr = 0
        try:
            self.adjes = namelist['radiation']['RRTM']['adjes']
        except:
            self.adjes = 0.5
        try:
            self.scon = namelist['radiation']['RRTM']['solar_constant']
        except:
            self.scon = 1360.0
        try:
            self.coszen =namelist['radiation']['RRTM']['coszen']
        except:
            self.coszen = 2.0/pi
        try:
            self.adif = namelist['radiation']['RRTM']['adif']
        except:
            self.adif = 0.06
        try:
            self.adir = namelist['radiation']['RRTM']['adir']
        except:
            if (self.coszen > 0.0):
                self.adir = (.026/(self.coszen**1.7 + .065)+(.15*(self.coszen-0.10)*(self.coszen-0.50)*(self.coszen- 1.00)))
            else:
                self.adir = 0.0
        return

    cpdef get_pv_star(self, double t):
        return self.CC.LT.fast_lookup(t)
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
                trace[1,:] = lw_absorber[:,i].reshape(1,lw_np) * self.co2_factor
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
                tmpTrace[k,i] = g/(self.p_levels[k] - self.p_levels[k+1])*(trpath[k+1,i]-trpath[k,i])

        self.o3vmr  = np.array(tmpTrace[:,0],dtype=np.double, order='F')
        self.co2vmr = np.array(tmpTrace[:,1],dtype=np.double, order='F')
        self.ch4vmr =  np.array(tmpTrace[:,2],dtype=np.double, order='F')
        self.n2ovmr =  np.array(tmpTrace[:,3],dtype=np.double, order='F')
        self.o2vmr  =  np.array(tmpTrace[:,4],dtype=np.double, order='F')
        self.cfc11vmr =  np.array(tmpTrace[:,5],dtype=np.double, order='F')
        self.cfc12vmr =  np.array(tmpTrace[:,6],dtype=np.double, order='F')
        self.cfc22vmr = np.array( tmpTrace[:,7],dtype=np.double, order='F')
        self.ccl4vmr  =  np.array(tmpTrace[:,8],dtype=np.double, order='F')



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
            # self.lw_up_srf = uflx_lw_out[0,0]
            # self.lw_down_srf = dflx_lw_out[0,0]
            # self.sw_up_srf  =  uflx_sw_out[0,0]
            # self.sw_down_srf = dflx_sw_out[0,0]
            for k in xrange(nlayers):
                self.t_tend_rad[k] = (hr_lw_out[0,k] + hr_sw_out[0,k])/86400.0


        return


    cpdef initialize(self,  ParallelMPI.ParallelMPI Pa, double [:] pressure_array, double Pg, double Tg, double RH):

        ForcingReferenceBase.initialize(self,Pa, pressure_array,  Pg, Tg, RH)
        self.sst = Tg
        self.dt_rce  =3600.0 #1 hr?

        # pressure coordinates
        self.nlayers = 205
        self.nlevels = self.nlayers + 1
        self.p_levels = np.linspace(Pg, 0.0, num=self.nlevels, endpoint=True)
        self.p_layers = 0.5 * np.add(self.p_levels[1:],self.p_levels[:-1])

        self.initialize_radiation()


        self.t_layers = np.linspace(290.0, 260.0, self.nlayers)
        self.qv_layers =np.zeros(np.shape(self.p_layers), dtype=np.double, order='c')

        cdef:
            Py_ssize_t k
            double  pv

        for k in xrange(self.nlayers):
            pv = self.get_pv_star(self.t_layers[k]) * RH
            self.qv_layers[k] = fmax(eps_v * pv / (pressure_array[k] + (eps_v-1.0)*pv),0.0)

            # print(self.t_layers[k])


        self.delta_t = np.ones(np.shape(self.t_layers),dtype =np.double, order='c')
        self.t_tend_rad = np.zeros(np.shape(self.t_layers),dtype =np.double, order='c')

        #initialize the lookup table
        t_table = LookupProfiles(75,self.nlayers)
        qv_table = LookupProfiles(75, self.nlayers)


        cdef:
            Py_ssize_t iter_count = -1
            Py_ssize_t p_index_count = -1

        for tropo_p_index in xrange(5,80):
            p_index_count +=1
            iter_count = 0
            while np.max(self.delta_t) > 0.01 and iter_count < 10000:
                # print('RCE iteration ', count)
                self.rce_step()
                iter_count += 1
                # print(count,np.max(self.delta_t) )
            t_table.table_vals[p_index_count,:] = self.t_layers[:]

        print("interactive RCE ", iter_count, np.max(self.delta_t))

        data = nc.Dataset('./CGILSdata/RCE_1xCO2.nc', 'r')
        # Arrays must be flipped (low to high pressure) to use numpy interp function
        pressure_ref = data.variables['p_full'][:]
        temperature_ref = data.variables['temp_rc'][:]


        plt.figure(iter_count)
        plt.plot(self.t_layers, self.p_layers[:],'-b')
        plt.plot(temperature_ref, pressure_ref,'-r')
        plt.gca().invert_yaxis()
        plt.show()



        self.temperature = np.array(np.interp(pressure_array, self.p_layers, self.t_layers), dtype=np.double, order='c')
        self.qt = np.array(np.interp(pressure_array, self.p_layers, self.qv_layers), dtype=np.double, order='c')
        cdef Py_ssize_t nz = np.shape(pressure_array)[0]
        for k in xrange(nz):
            self.s[k] = self.entropy(pressure_array[k], self.temperature[k], self.qt[k], 0.0, 0.0)
            self.rv[k] = self.qt[k]/(1.0-self.qt[k])
            self.u[k] = min(-10.0 + (-7.0-(-10.0))/(750.0e2-1000.0e2)*(pressure_array[k]-1000.0e2),-4.0)


        return

    cpdef update_qv(self, double p, double t, double rh):
        cdef double pv, pd, qv
        pv = self.get_pv_star(t) * rh
        pd = p - pv
        qv = pv/(pd * eps_vi + pv)
        return  qv

# Tropical temperature is fixed deltaT over LES domain SST
# set a trial tropopause
# set temperatures as moist adiabat below tropopause
# compute radiative equilibrium above tropopause
# move tropopause height until the radiative equilibrium temperature at H = moist adiabat temperature at H
# I think this means that if Trad < Tmoist, H should increase
# if Tmoist > Trad, H should decrease


    cpdef rce_step(self):
        # update temperatures due to radiation

        self.compute_radiation()

        cdef:
            Py_ssize_t k
            double [:] t_0 = np.zeros(np.shape(self.t_layers),dtype =np.double, order='c')
            double [:] t_1 = np.zeros(np.shape(self.t_layers),dtype =np.double, order='c')
            double [:] t_2 = np.zeros(np.shape(self.t_layers),dtype =np.double, order='c')

        with nogil:
            for k in xrange(self.nlayers):
                t_0[k] = self.t_layers[k] + self.t_tend_rad[k] * self.dt_rce



        with nogil:
            for k in xrange(self.nlayers):
                self.delta_t[k] = fabs(t_2[k] - self.t_layers[k])
                self.t_layers[k] = t_2[k]

        return

# Control simulations use AdjustedMoistAdiabat
# Reference temperature profile correspondends to a moist adiabat
# Reference moisture profile corresponds to a fixed relative humidity given the reference temperature profile
cdef class AdjustedMoistAdiabat(ForcingReferenceBase):
    def __init__(self,namelist,  LatentHeat LH, ParallelMPI.ParallelMPI Pa ):

        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp
        self.CC = ClausiusClapeyron()
        self.CC.initialize(namelist, LH, Pa)

        return
    cpdef get_pv_star(self, t):
        return self.CC.LT.fast_lookup(t)

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



    cpdef initialize(self, ParallelMPI.ParallelMPI Pa, double [:] pressure_array,  double Pg, double Tg, double RH):
        '''
        Initialize the forcing reference profiles. These profiles use the temperature corresponding to a moist adiabat,
        but modify the water vapor content to have a given relative humidity. Thus entropy and qt are not conserved.
        '''
        ForcingReferenceBase.initialize(self,Pa, pressure_array, Pg, Tg, RH)

        cdef double pvg = self.get_pv_star(Tg)
        cdef double qtg = eps_v * pvg / (Pg + (eps_v-1.0)*pvg)
        cdef double sg = self.entropy(Pg, Tg, qtg, 0.0, 0.0)


        cdef double temperature, ql, qi, pv
        cdef Py_ssize_t n_levels = np.shape(pressure_array)[0]

        # Compute reference state thermodynamic profiles
        for k in xrange(n_levels):
            temperature, ql, qi = self.eos(pressure_array[k], sg, qtg)
            pv = self.get_pv_star(temperature) * RH
            self.qt[k] = eps_v * pv / (pressure_array[k] + (eps_v-1.0)*pv)
            self.s[k] = self.entropy(pressure_array[k],temperature, self.qt[k] , 0.0, 0.0)
            self.temperature[k] = temperature
            self.rv[k] = self.qt[k]/(1.0-self.qt[k])
            self.u[k] =  min(-10.0 + (-7.0-(-10.0))/(750.0e2-1000.0e2)*(pressure_array[k]-1000.0e2),-4.0)

        return

cdef class LookupProfiles:
    def __init__(self, Py_ssize_t nprofiles, Py_ssize_t nz):
        self.nprofiles = nprofiles
        self.nz = nz
        self.table_vals = np.zeros((nprofiles,nz),dtype=np.double, order='c')
        self.access_vals = np.zeros(nprofiles, dtype=np.double, order='c')
        return



