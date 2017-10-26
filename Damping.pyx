#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

from libc.math cimport fmin, fmax, sin
import cython
import netCDF4 as nc
import numpy as np
cimport ParallelMPI as ParallelMPI
cimport PrognosticVariables as PrognosticVariables
cimport Grid as Grid
cimport ReferenceState
cimport DiagnosticVariables
cimport numpy as np
from thermodynamic_functions cimport pd_c, pv_c
from entropies cimport sv_c, sd_c
from scipy.special import erf
import cPickle
cimport TimeStepping
from scipy.interpolate import pchip
from thermodynamic_functions cimport cpm_c

include 'parameters.pxi'

cdef class Damping:
    def __init__(self, namelist, ParallelMPI.ParallelMPI Pa):
        if(namelist['damping']['scheme'] == 'None'):
            self.scheme = Dummy(namelist, Pa)
            Pa.root_print('No Damping!')
        elif(namelist['damping']['scheme'] == 'Rayleigh'):
            casename = namelist['meta']['casename']
            if casename == 'GCMMean':
                self.scheme = RayleighGCMMeanNudge(namelist, Pa)
            elif casename == 'GCMVarying':
                self.scheme = RayleighGCMVarying(namelist, Pa)
            else:
                self.scheme = Rayleigh(namelist, Pa)
                Pa.root_print('Using Rayleigh Damping')


        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS):
        self.scheme.initialize(Gr, RS)
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):
        self.scheme.update(Gr, RS, PV, DV, Pa, TS)
        return

cdef class Dummy:
    def __init__(self, namelist, ParallelMPI.ParallelMPI Pa):
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):
        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS):
        return

cdef class RayleighGCMMeanNudge:
    def __init__(self, namelist, ParallelMPI.ParallelMPI Pa):

        try:
            self.z_d = namelist['damping']['Rayleigh']['z_d']
        except:
            Pa.root_print('Rayleigh damping z_d not given in namelist')
            Pa.root_print('Killing simulation now!')
            Pa.kill()

        try:
            self.gamma_r = namelist['damping']['Rayleigh']['gamma_r']
        except:
            Pa.root_print('Rayleigh damping gamm_r not given in namelist')
            Pa.root_print('Killing simulation now!')
            Pa.kill()

        self.file = str(namelist['gcm']['file'])
        self.gcm_profiles_initialized = False
        self.t_indx = 0

        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS):
        cdef:
            int k
            double z_top

        self.gamma_zhalf = np.zeros(
            (Gr.dims.nlg[2]),
            dtype=np.double,
            order='c')
        self.gamma_z = np.zeros((Gr.dims.nlg[2]), dtype=np.double, order='c')
        z_top = Gr.zpl[Gr.dims.nlg[2] - Gr.dims.gw]


        self.z_d = 20000.0
        with nogil:
            for k in range(Gr.dims.nlg[2]):
                if Gr.zpl_half[k] >= z_top - self.z_d:
                    self.gamma_zhalf[
                        k] = self.gamma_r * sin((pi / 2.0) * (1.0 - (z_top - Gr.zpl_half[k]) / self.z_d))**2.0
                if Gr.zpl[k] >= z_top - self.z_d:
                    self.gamma_z[
                        k] = self.gamma_r * sin((pi / 2.0) * (1.0 - (z_top - Gr.zpl[k]) / self.z_d))**2.0


#        import pylab as plt
#        plt.figure()
#        plt.plot(self.gamma_z)
#        plt.show()
#        import sys; sys.exit()


        #Set up tendency damping using error function
        fh = open(self.file, 'r')
        input_data_tv = cPickle.load(fh)
        fh.close()


        #Compute height for daimping profiles
        #dt_qg_conv = np.mean(input_data_tv['dt_qg_param'][:,::-1],axis=0)
        zfull = np.mean(input_data_tv['zfull'][:,::-1], axis=0)
        temp = np.mean(input_data_tv['temp'][:,::-1],axis=0)
        temp = interp_pchip(Gr.zp_half, zfull, temp)
        #import pylab as plt
        #plt.plot(np.abs(dt_qg_conv))

        print temp
        for i in range(temp.shape[0]-1, -1, -1):
            if temp[i]  >  220.0:
            #if np.abs(dt_qg_conv[i]) > cutoff:
                self.tend_flat_z_d = z_top - Gr.zp_half[i]
                break

        print 'Convective top', z_top - self.tend_flat_z_d


        z_damp = z_top - self.tend_flat_z_d
        z = (np.array(Gr.zp) - z_damp)/( self.tend_flat_z_d*0.15)
        z_half = (np.array(Gr.zp_half) - z_damp)/( self.tend_flat_z_d*0.15)

        tend_flat = erf(z)
        tend_flat[tend_flat < 0.0] = 0.0
        tend_flat = 1.0 - tend_flat
        self.tend_flat = tend_flat
        tend_flat = erf(z_half)
        tend_flat[tend_flat < 0.0] = 0.0
        tend_flat = 1.0 - tend_flat
        self.tend_flat_half = tend_flat


        #import pylab as plt
        #plt.plot(self.tend_flat_half, np.array(Gr.zpl_half)/1000.0,'-ok')
        #plt.plot(dt_tg_rad, zfull,'-ok')
        #plt.grid()
        #plt.savefig('sigmoid.pdf')
        #plt.show()
        #import sys; sys.exit()



        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):
        cdef:
            Py_ssize_t var_shift
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i, j, k, ishift, jshift, ijk
            double[:] domain_mean
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t s_shift
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            double pd, pv, qt, qv, p0, rho0, t
            double weight


        if not self.gcm_profiles_initialized or int(TS.t // (3600.0 * 6.0)) > self.t_indx:
            self.t_indx = int(TS.t // (3600.0 * 6.0))
            self.gcm_profiles_initialized = True
            Pa.root_print('Updating Total Tendencies in damping!')

            fh = open(self.file, 'r')
            input_data_tv = cPickle.load(fh)
            fh.close()

            #zfull = input_data_tv['zfull'][self.t_indx,::-1]
            #temp_dt_total = input_data_tv['temp_total'][self.t_indx,::-1]
            #shum_dt_total = input_data_tv['dt_qg_total'][self.t_indx,::-1]

            zfull = np.mean(input_data_tv['zfull'][:,::-1], axis=0)
            temp_dt_total = np.mean(input_data_tv['temp_total'][:,::-1], axis=0)
            shum_dt_total = np.mean(input_data_tv['dt_qg_total'][:,::-1], axis=0)

            self.dt_tg_total = interp_pchip(Gr.zp_half, zfull, temp_dt_total)
            self.dt_qg_total =  interp_pchip(Gr.zp_half, zfull, shum_dt_total)




        for var_name in PV.name_index:
            var_shift = PV.get_varshift(Gr, var_name)
            domain_mean = Pa.HorizontalMean(Gr, & PV.values[var_shift])
            if var_name == 'w':
                with nogil:
                    for i in xrange(imin, imax):
                        ishift = i * istride
                        for j in xrange(jmin, jmax):
                            jshift = j * jstride
                            for k in xrange(kmin, kmax):
                                ijk = ishift + jshift + k
                                PV.tendencies[var_shift + ijk] -= (PV.values[var_shift + ijk] - 0.0) * self.gamma_zhalf[k]

            elif var_name == 'u' or var_name == 'v':
                with nogil:
                    for i in xrange(imin, imax):
                        ishift = i * istride
                        for j in xrange(jmin, jmax):
                            jshift = j * jstride
                            for k in xrange(kmin, kmax):
                                ijk = ishift + jshift + k
                                PV.tendencies[var_shift + ijk] -= (PV.values[var_shift + ijk] - domain_mean[k]) * self.gamma_z[k]
            else:
                with nogil:
                    for i in xrange(imin, imax):
                        ishift = i * istride
                        for j in xrange(jmin, jmax):
                            jshift = j * jstride
                            for k in xrange(kmin, kmax):
                                ijk = ishift + jshift + k
                                #PV.tendencies[var_shift + ijk] *= self.tend_flat[k]
                                PV.tendencies[var_shift + ijk] -= (PV.values[var_shift + ijk] - domain_mean[k]) * self.gamma_z[k]

        # if 's' in PV.name_index:
        #     s_shift = PV.get_varshift(Gr, 's')
        # else:
        #     s_shift = PV.get_varshift(Gr, 'thli')
        #
        # with nogil:
        #     for i in xrange(gw,imax):
        #         ishift = i * istride
        #         for j in xrange(gw,jmax):
        #             jshift = j * jstride
        #             for k in xrange(gw,kmax):
        #
        #                 weight = self.tend_flat_half[k]
        #
        #                 ijk = ishift + jshift + k
        #                 p0 = RS.p0_half[k]
        #                 rho0 = RS.rho0_half[k]
        #                 qt = PV.values[qt_shift + ijk]
        #                 qv = qt - DV.values[ql_shift + ijk]
        #                 pd = pd_c(p0,qt,qv)
        #                 pv = pv_c(p0,qt,qv)
        #                 t  = DV.values[t_shift + ijk]
        #                 PV.tendencies[s_shift + ijk] =   (weight)*PV.tendencies[s_shift + ijk]
        #                 #PV.tendencies[s_shift + ijk] += (sv_c(pv,t) - sd_c(pd,t)) * (self.dt_qg_total[k]* (1.0 -weight) ) + (1.0 - weight) * (cpm_c(qt) * (self.dt_tg_total[k]))/t
        #                 #PV.tendencies[qt_shift + ijk] = self.dt_qg_total[k]* (1.0 - weight) + PV.tendencies[qt_shift + ijk] *(weight)
        #                 PV.tendencies[qt_shift + ijk] = PV.tendencies[qt_shift + ijk] *(weight)
        #                 PV.tendencies[u_shift + ijk] = (weight) * PV.tendencies[u_shift + ijk]
        #                 PV.tendencies[v_shift + ijk] = (weight) * PV.tendencies[v_shift + ijk]

        return



cdef class RayleighGCMMean:
    def __init__(self, namelist, ParallelMPI.ParallelMPI Pa):

        try:
            self.z_d = namelist['damping']['Rayleigh']['z_d']
        except:
            Pa.root_print('Rayleigh damping z_d not given in namelist')
            Pa.root_print('Killing simulation now!')
            Pa.kill()

        try:
            self.gamma_r = namelist['damping']['Rayleigh']['gamma_r']
        except:
            Pa.root_print('Rayleigh damping gamm_r not given in namelist')
            Pa.root_print('Killing simulation now!')
            Pa.kill()

        self.file = str(namelist['gcm']['file'])
        self.gcm_profiles_initialized = False
        self.t_indx = 0

        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS):
        cdef:
            int k
            double z_top

        self.gamma_zhalf = np.zeros(
            (Gr.dims.nlg[2]),
            dtype=np.double,
            order='c')
        self.gamma_z = np.zeros((Gr.dims.nlg[2]), dtype=np.double, order='c')
        z_top = Gr.zpl[Gr.dims.nlg[2] - Gr.dims.gw]

        with nogil:
            for k in range(Gr.dims.nlg[2]):
                if Gr.zpl_half[k] >= z_top - self.z_d:
                    self.gamma_zhalf[
                        k] = self.gamma_r * sin((pi / 2.0) * (1.0 - (z_top - Gr.zpl_half[k]) / self.z_d))**2.0
                if Gr.zpl[k] >= z_top - self.z_d:
                    self.gamma_z[
                        k] = self.gamma_r * sin((pi / 2.0) * (1.0 - (z_top - Gr.zpl[k]) / self.z_d))**2.0



        #Set up tendency damping using error function
        fh = open(self.file, 'r')
        input_data_tv = cPickle.load(fh)
        fh.close()


        #Compute height for daimping profiles
        #dt_qg_conv = np.mean(input_data_tv['dt_qg_param'][:,::-1],axis=0)
        zfull = np.mean(input_data_tv['zfull'][:,::-1], axis=0)
        temp = np.mean(input_data_tv['temp'][:,::-1],axis=0)
        temp = interp_pchip(Gr.zp_half, zfull, temp)
        #import pylab as plt
        #plt.plot(np.abs(dt_qg_conv))

        print temp
        for i in range(temp.shape[0]-1, -1, -1):
            if temp[i]  >  220.0:
            #if np.abs(dt_qg_conv[i]) > cutoff:
                self.tend_flat_z_d = z_top - Gr.zp_half[i]
                break

        print 'Convective top', z_top - self.tend_flat_z_d


        z_damp = z_top - self.tend_flat_z_d
        z = (np.array(Gr.zp) - z_damp)/( self.tend_flat_z_d*0.15)
        z_half = (np.array(Gr.zp_half) - z_damp)/( self.tend_flat_z_d*0.15)

        tend_flat = erf(z)
        tend_flat[tend_flat < 0.0] = 0.0
        tend_flat = 1.0 - tend_flat
        self.tend_flat = tend_flat
        tend_flat = erf(z_half)
        tend_flat[tend_flat < 0.0] = 0.0
        tend_flat = 1.0 - tend_flat
        self.tend_flat_half = tend_flat


        #import pylab as plt
        #plt.plot(self.tend_flat_half, np.array(Gr.zpl_half)/1000.0,'-ok')
        #plt.plot(dt_tg_rad, zfull,'-ok')
        #plt.grid()
        #plt.savefig('sigmoid.pdf')
        #plt.show()
        #import sys; sys.exit()



        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):
        cdef:
            Py_ssize_t var_shift
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i, j, k, ishift, jshift, ijk
            double[:] domain_mean
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t s_shift
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            double pd, pv, qt, qv, p0, rho0, t
            double weight


        if not self.gcm_profiles_initialized or int(TS.t // (3600.0 * 6.0)) > self.t_indx:
            self.t_indx = int(TS.t // (3600.0 * 6.0))
            self.gcm_profiles_initialized = True
            Pa.root_print('Updating Total Tendencies in damping!')

            fh = open(self.file, 'r')
            input_data_tv = cPickle.load(fh)
            fh.close()

            #zfull = input_data_tv['zfull'][self.t_indx,::-1]
            #temp_dt_total = input_data_tv['temp_total'][self.t_indx,::-1]
            #shum_dt_total = input_data_tv['dt_qg_total'][self.t_indx,::-1]

            zfull = np.mean(input_data_tv['zfull'][:,::-1], axis=0)
            temp_dt_total = np.mean(input_data_tv['temp_total'][:,::-1], axis=0)
            shum_dt_total = np.mean(input_data_tv['dt_qg_total'][:,::-1], axis=0)

            self.dt_tg_total = interp_pchip(Gr.zp_half, zfull, temp_dt_total)
            self.dt_qg_total =  interp_pchip(Gr.zp_half, zfull, shum_dt_total)




        for var_name in PV.name_index:
            var_shift = PV.get_varshift(Gr, var_name)
            domain_mean = Pa.HorizontalMean(Gr, & PV.values[var_shift])
            if var_name == 'w':
                with nogil:
                    for i in xrange(imin, imax):
                        ishift = i * istride
                        for j in xrange(jmin, jmax):
                            jshift = j * jstride
                            for k in xrange(kmin, kmax):
                                ijk = ishift + jshift + k
                                PV.tendencies[var_shift + ijk] -= (PV.values[var_shift + ijk] - 0.0) * self.gamma_zhalf[k] 

            elif var_name == 'u' or var_name == 'v':
                with nogil:
                    for i in xrange(imin, imax):
                        ishift = i * istride
                        for j in xrange(jmin, jmax):
                            jshift = j * jstride
                            for k in xrange(kmin, kmax):
                                ijk = ishift + jshift + k
                                PV.tendencies[var_shift + ijk] -= (PV.values[var_shift + ijk] - domain_mean[k]) * self.gamma_z[k]
            else:
                with nogil:
                    for i in xrange(imin, imax):
                        ishift = i * istride
                        for j in xrange(jmin, jmax):
                            jshift = j * jstride
                            for k in xrange(kmin, kmax):
                                ijk = ishift + jshift + k
                                #PV.tendencies[var_shift + ijk] *= self.tend_flat[k]
                                PV.tendencies[var_shift + ijk] -= (PV.values[var_shift + ijk] - domain_mean[k]) * self.gamma_z[k]

        if 's' in PV.name_index:
            s_shift = PV.get_varshift(Gr, 's')
        else:
            s_shift = PV.get_varshift(Gr, 'thli')

        with nogil:
            for i in xrange(gw,imax):
                ishift = i * istride
                for j in xrange(gw,jmax):
                    jshift = j * jstride
                    for k in xrange(gw,kmax):

                        weight = self.tend_flat_half[k]

                        ijk = ishift + jshift + k
                        p0 = RS.p0_half[k]
                        rho0 = RS.rho0_half[k]
                        qt = PV.values[qt_shift + ijk]
                        qv = qt - DV.values[ql_shift + ijk]
                        pd = pd_c(p0,qt,qv)
                        pv = pv_c(p0,qt,qv)
                        t  = DV.values[t_shift + ijk]
                        PV.tendencies[s_shift + ijk] =   (weight)*PV.tendencies[s_shift + ijk]
                        #PV.tendencies[s_shift + ijk] += (sv_c(pv,t) - sd_c(pd,t)) * (self.dt_qg_total[k]* (1.0 -weight) ) + (1.0 - weight) * (cpm_c(qt) * (self.dt_tg_total[k]))/t
                        #PV.tendencies[qt_shift + ijk] = self.dt_qg_total[k]* (1.0 - weight) + PV.tendencies[qt_shift + ijk] *(weight)
                        PV.tendencies[qt_shift + ijk] = PV.tendencies[qt_shift + ijk] *(weight)
                        PV.tendencies[u_shift + ijk] = (weight) * PV.tendencies[u_shift + ijk]
                        PV.tendencies[v_shift + ijk] = (weight) * PV.tendencies[v_shift + ijk] 

        return



cdef class RayleighGCMVarying:
    def __init__(self, namelist, ParallelMPI.ParallelMPI Pa):

        try:
            self.z_d = namelist['damping']['Rayleigh']['z_d']
        except:
            Pa.root_print('Rayleigh damping z_d not given in namelist')
            Pa.root_print('Killing simulation now!')
            Pa.kill()

        try:
            self.gamma_r = namelist['damping']['Rayleigh']['gamma_r']
        except:
            Pa.root_print('Rayleigh damping gamm_r not given in namelist')
            Pa.root_print('Killing simulation now!')
            Pa.kill()

        self.file = str(namelist['gcm']['file'])
        self.gcm_profiles_initialized = False
        self.t_indx = 0

        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS):
        cdef:
            int k
            double z_top

        self.gamma_zhalf = np.zeros(
            (Gr.dims.nlg[2]),
            dtype=np.double,
            order='c')
        self.gamma_z = np.zeros((Gr.dims.nlg[2]), dtype=np.double, order='c')
        z_top = Gr.zpl[Gr.dims.nlg[2] - Gr.dims.gw]

        with nogil:
            for k in range(Gr.dims.nlg[2]):
                if Gr.zpl_half[k] >= z_top - self.z_d:
                    self.gamma_zhalf[
                        k] = self.gamma_r * sin((pi / 2.0) * (1.0 - (z_top - Gr.zpl_half[k]) / self.z_d))**2.0
                if Gr.zpl[k] >= z_top - self.z_d:
                    self.gamma_z[
                        k] = self.gamma_r * sin((pi / 2.0) * (1.0 - (z_top - Gr.zpl[k]) / self.z_d))**2.0



        #Set up tendency damping using error function


        fh = open(self.file, 'r')
        input_data_tv = cPickle.load(fh)
        fh.close()


        #Compute height for daimping profiles
        #dt_qg_conv = np.mean(input_data_tv['dt_qg_param'][:,::-1],axis=0)
        zfull = np.mean(input_data_tv['zfull'][:,::-1], axis=0)
        temp = np.mean(input_data_tv['temp'][:,::-1],axis=0)
        temp = interp_pchip(Gr.zp_half, zfull, temp)
        #import pylab as plt
        #plt.plot(np.abs(dt_qg_conv))

        print temp
        for i in range(temp.shape[0]-1, -1, -1):
            if temp[i]  >  230.0:
            #if np.abs(dt_qg_conv[i]) > cutoff:
                self.tend_flat_z_d = z_top - Gr.zp_half[i]
                break

        print 'Convective top', z_top - self.tend_flat_z_d


        z_damp = z_top - self.tend_flat_z_d
        z = (np.array(Gr.zp) - z_damp)/( self.tend_flat_z_d*0.15)
        z_half = (np.array(Gr.zp_half) - z_damp)/( self.tend_flat_z_d*0.15)

        tend_flat = erf(z)
        tend_flat[tend_flat < 0.0] = 0.0
        tend_flat = 1.0 - tend_flat
        self.tend_flat = tend_flat
        tend_flat = erf(z_half)
        tend_flat[tend_flat < 0.0] = 0.0
        tend_flat = 1.0 - tend_flat
        self.tend_flat_half = tend_flat



        #import pylab as plt
        #plt.plot(self.tend_flat_half, np.array(Gr.zpl_half)/1000.0,'-ok')
        #plt.plot(dt_tg_rad, zfull,'-ok')
        #plt.grid()
        #plt.savefig('sigmoid.pdf')
        #plt.show()
        #import sys; sys.exit()



        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):
        cdef:
            Py_ssize_t var_shift
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i, j, k, ishift, jshift, ijk
            double[:] domain_mean
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            double pd, pv, qt, qv, p0, rho0, t
            double weight


        if not self.gcm_profiles_initialized or int(TS.t // (3600.0 * 6.0)) > self.t_indx:
            self.t_indx = int(TS.t // (3600.0 * 6.0))
            self.gcm_profiles_initialized = True
            Pa.root_print('Updating Total Tendencies in damping!')

            fh = open(self.file, 'r')
            input_data_tv = cPickle.load(fh)
            fh.close()


            zfull = input_data_tv['zfull'][self.t_indx ,::-1]
            temp_dt_total = input_data_tv['temp_total'][self.t_indx ,::-1]
            shum_dt_total = input_data_tv['dt_qg_total'][self.t_indx ,::-1]

            self.dt_tg_total = interp_pchip(Gr.zp_half, zfull, temp_dt_total)
            self.dt_qg_total =  interp_pchip(Gr.zp_half, zfull, shum_dt_total)


        for var_name in PV.name_index:
            var_shift = PV.get_varshift(Gr, var_name)
            domain_mean = Pa.HorizontalMean(Gr, & PV.values[var_shift])
            if var_name == 'w':
                with nogil:
                    for i in xrange(imin, imax):
                        ishift = i * istride
                        for j in xrange(jmin, jmax):
                            jshift = j * jstride
                            for k in xrange(kmin, kmax):
                                ijk = ishift + jshift + k
                                PV.tendencies[var_shift + ijk] -= (PV.values[var_shift + ijk] - 0.0) * self.gamma_zhalf[k]

            elif var_name == 'u' or var_name == 'v':
                with nogil:
                    for i in xrange(imin, imax):
                        ishift = i * istride
                        for j in xrange(jmin, jmax):
                            jshift = j * jstride
                            for k in xrange(kmin, kmax):
                                ijk = ishift + jshift + k
                                PV.tendencies[var_shift + ijk] -= (PV.values[var_shift + ijk] - domain_mean[k]) * self.gamma_z[k]
            else:
                with nogil:
                    for i in xrange(imin, imax):
                        ishift = i * istride
                        for j in xrange(jmin, jmax):
                            jshift = j * jstride
                            for k in xrange(kmin, kmax):
                                ijk = ishift + jshift + k
                                #PV.tendencies[var_shift + ijk] *= self.tend_flat[k]
                                PV.tendencies[var_shift + ijk] -= (PV.values[var_shift + ijk] - domain_mean[k]) * self.gamma_z[k]


        with nogil:
            for i in xrange(gw,imax):
                ishift = i * istride
                for j in xrange(gw,jmax):
                    jshift = j * jstride
                    for k in xrange(gw,kmax):

                        weight = self.tend_flat_half[k]

                        ijk = ishift + jshift + k
                        p0 = RS.p0_half[k]
                        rho0 = RS.rho0_half[k]
                        qt = PV.values[qt_shift + ijk]
                        qv = qt - DV.values[ql_shift + ijk]
                        pd = pd_c(p0,qt,qv)
                        pv = pv_c(p0,qt,qv)
                        t  = DV.values[t_shift + ijk]
                        #PV.tendencies[s_shift + ijk] =   (weight)*PV.tendencies[s_shift + ijk]
                        #PV.tendencies[s_shift + ijk] += (sv_c(pv,t) - sd_c(pd,t)) * (self.dt_qg_total[k]* (1.0 -weight) ) + (1.0 - weight) * (cpm_c(qt) * (self.dt_tg_total[k]))/t
                        #PV.tendencies[qt_shift + ijk] = self.dt_qg_total[k]* (1.0 - weight) + PV.tendencies[qt_shift + ijk] *(weight)
                        PV.tendencies[s_shift + ijk] *= weight
                        PV.tendencies[qt_shift + ijk] *= weight
                        PV.tendencies[u_shift + ijk] = (weight) * PV.tendencies[u_shift + ijk]
                        PV.tendencies[v_shift + ijk] = (weight) * PV.tendencies[v_shift + ijk]

        return



cdef class Rayleigh:
    def __init__(self, namelist, ParallelMPI.ParallelMPI Pa):


        try:
            self.z_d = namelist['damping']['Rayleigh']['z_d']
        except:
            Pa.root_print('Rayleigh damping z_d not given in namelist')
            Pa.root_print('Killing simulation now!')
            Pa.kill()

        try:
            self.gamma_r = namelist['damping']['Rayleigh']['gamma_r']
        except:
            Pa.root_print('Rayleigh damping gamm_r not given in namelist')
            Pa.root_print('Killing simulation now!')
            Pa.kill()
        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS):
        cdef:
            int k
            double z_top

        self.gamma_zhalf = np.zeros(
            (Gr.dims.nlg[2]),
            dtype=np.double,
            order='c')
        self.gamma_z = np.zeros((Gr.dims.nlg[2]), dtype=np.double, order='c')
        z_top = Gr.zpl[Gr.dims.nlg[2] - Gr.dims.gw]

        with nogil:
            for k in range(Gr.dims.nlg[2]):
                if Gr.zpl_half[k] >= z_top - self.z_d:
                    self.gamma_zhalf[
                        k] = self.gamma_r * sin((pi / 2.0) * (1.0 - (z_top - Gr.zpl_half[k]) / self.z_d))**2.0
                if Gr.zpl[k] >= z_top - self.z_d:
                    self.gamma_z[
                        k] = self.gamma_r * sin((pi / 2.0) * (1.0 - (z_top - Gr.zpl[k]) / self.z_d))**2.0



        #Set up tendency damping using error function
        z_damp = z_top - self.z_d
        z = (np.array(Gr.zp) - z_damp)/( self.z_d*0.5)
        z_half = (np.array(Gr.zp_half) - z_damp)/( self.z_d*0.5)

        tend_flat = erf(z)
        tend_flat[tend_flat < 0.0] = 0.0
        tend_flat = 1.0 - tend_flat
        self.tend_flat = tend_flat
        tend_flat = erf(z_half)
        tend_flat[tend_flat < 0.0] = 0.0
        tend_flat = 1.0 - tend_flat
        self.tend_flat_half = tend_flat


        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):
        cdef:
            Py_ssize_t var_shift
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i, j, k, ishift, jshift, ijk
            double[:] domain_mean


        for var_name in PV.name_index:
            var_shift = PV.get_varshift(Gr, var_name)
            domain_mean = Pa.HorizontalMean(Gr, & PV.values[var_shift])
            if var_name == 'w':
                with nogil:
                    for i in xrange(imin, imax):
                        ishift = i * istride
                        for j in xrange(jmin, jmax):
                            jshift = j * jstride
                            for k in xrange(kmin, kmax):
                                ijk = ishift + jshift + k
                                PV.tendencies[var_shift + ijk] *= self.tend_flat_half[k]

            elif var_name == 'u' or var_name == 'v':
                with nogil:
                    for i in xrange(imin, imax):
                        ishift = i * istride
                        for j in xrange(jmin, jmax):
                            jshift = j * jstride
                            for k in xrange(kmin, kmax):
                                ijk = ishift + jshift + k
                                PV.tendencies[var_shift + ijk] -= (PV.values[var_shift + ijk] - domain_mean[k]) * self.gamma_z[k]
            else:
                with nogil:
                    for i in xrange(imin, imax):
                        ishift = i * istride
                        for j in xrange(jmin, jmax):
                            jshift = j * jstride
                            for k in xrange(kmin, kmax):
                                ijk = ishift + jshift + k
                                PV.tendencies[var_shift + ijk] *= self.tend_flat[k]
                                PV.tendencies[var_shift + ijk] -= (PV.values[var_shift + ijk] - domain_mean[k]) * self.gamma_z[k]
        return


from scipy.interpolate import pchip
def interp_pchip(z_out, z_in, v_in, pchip_type=False):
    if pchip_type:
        p = pchip(z_in, v_in, extrapolate=True)
        return p(z_out)
    else:
        return np.interp(z_out, z_in, v_in)