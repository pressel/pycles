#!python
#cython: boundscheck=False
#cython: wraparound=True
#cython: initializedcheck=False
#cython: cdivision=True

from libc.math cimport fmin, fmax, sin, cos
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


include 'parameters.pxi'

cdef class Damping:
    def __init__(self, namelist, ParallelMPI.ParallelMPI Pa):
        if(namelist['damping']['scheme'] == 'None'):
            self.scheme = Dummy(namelist, Pa)
            Pa.root_print('No Damping!')
        elif(namelist['damping']['scheme'] == 'Rayleigh'):
            self.scheme = Rayleigh(namelist, Pa)
            Pa.root_print('Using Rayleigh Damping')
        elif(namelist['damping']['scheme'] == 'NudgeCGILS'):
            self.scheme = NudgeCGILS(namelist,Pa)
        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS):
        self.scheme.initialize(Gr, RS)
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        self.scheme.update(Gr, RS, PV, DV, Pa)
        return

cdef class Dummy:
    def __init__(self, namelist, ParallelMPI.ParallelMPI Pa):
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS):
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
        z_top = Gr.dims.dx[2] * Gr.dims.n[2]
        with nogil:
            for k in range(Gr.dims.nlg[2]):
                if Gr.zl_half[k] >= z_top - self.z_d:
                    self.gamma_zhalf[
                        k] = self.gamma_r * sin((pi / 2.0) * (1.0 - (z_top - Gr.zl_half[k]) / self.z_d))**2.0
                if Gr.zl[k] >= z_top - self.z_d:
                    self.gamma_z[
                        k] = self.gamma_r * sin((pi / 2.0) * (1.0 - (z_top - Gr.zl[k]) / self.z_d))**2.0
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
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
                                PV.tendencies[var_shift + ijk] -= (PV.values[var_shift + ijk] - domain_mean[k]) * self.gamma_zhalf[k]
            else:
                with nogil:
                    for i in xrange(imin, imax):
                        ishift = i * istride
                        for j in xrange(jmin, jmax):
                            jshift = j * jstride
                            for k in xrange(kmin, kmax):
                                ijk = ishift + jshift + k
                                PV.tendencies[var_shift + ijk] -= (PV.values[var_shift + ijk] - domain_mean[k]) * self.gamma_z[k]
        return





cdef class NudgeCGILS:
    def __init__(self, namelist, ParallelMPI.ParallelMPI Pa):

        try:
            self.loc = namelist['meta']['CGILS']['location']
            if self.loc !=12 and self.loc != 11 and self.loc != 6:
                Pa.root_print('Invalid CGILS location (must be 6, 11, or 12)')
                Pa.kill()
        except:
            Pa.root_print('Must provide a CGILS location (6/11/12) in namelist')
            Pa.kill()
        try:
            self.is_p2 = namelist['meta']['CGILS']['P2']
        except:
            Pa.root_print('Must specify if CGILS run is perturbed')
            Pa.kill()

        if self.loc == 12:
            self.z_relax = 1200.0
            self.z_relax_plus = 1500.0
        elif self.loc == 11:
            self.z_relax = 2500.0
            self.z_relax_plus = 3000.0
        elif self.loc == 6:
            self.z_relax = 4000.0
            self.z_relax_plus = 4800.0

        self.tau_inverse = 1.0/(60.0*60.0) # inverse of  max nudging timescale, 1 hr, for all cases
        self.tau_vel_inverse = 1.0/(10.0*60.0) # nudging timescale of horizontal winds


        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS):

        cdef double [:] nudge_temperature

        if self.is_p2:
            file = './CGILSdata/p2k_s'+str(self.loc)+'.nc'
        else:
            file = './CGILSdata/ctl_s'+str(self.loc)+'.nc'

        print("DAMPING " + file)
        data = nc.Dataset(file, 'r')
        # Get the profile information we need from the data file to set up nudging profiles
        # Since we are sure the the range of LES pressures that are nudged are within the range of the
        # pressure data, simple linear interpolation will suffice
        pressure_data = data.variables['lev'][:]
        temperature_data = data.variables['T'][0,:,0,0]
        q_data = data.variables['q'][0,:,0,0]
        u_data = data.variables['u'][0,:,0,0]
        v_data = data.variables['v'][0,:,0,0]
        Ps = data.variables['Ps'][0,0,0]
        n_data = np.shape(pressure_data)[0] - 1

        temperature_right = (temperature_data[n_data-1] - temperature_data[n_data])/(pressure_data[n_data-1]-pressure_data[n_data])*(Ps-pressure_data[n_data]) + temperature_data[n_data]
        temperature_data = np.append(temperature_data, temperature_right)

        q_right = (q_data[n_data-1] - q_data[n_data])/(pressure_data[n_data-1]-pressure_data[n_data])*(Ps-pressure_data[n_data]) + q_data[n_data]
        q_data = np.append(q_data, q_right)

        u_right = (u_data[n_data-1] - u_data[n_data])/(pressure_data[n_data-1]-pressure_data[n_data])*(Ps-pressure_data[n_data]) + u_data[n_data]
        u_data = np.append(u_data,u_right)

        v_right = (v_data[n_data-1] - v_data[n_data])/(pressure_data[n_data-1]-pressure_data[n_data])*(Ps-pressure_data[n_data]) + v_data[n_data]
        v_data = np.append(v_data, v_right)
        pressure_data = np.append(pressure_data, Ps)


        self.nudge_s = np.zeros((Gr.dims.nlg[2],),dtype=np.double, order='c')

        self.nudge_qt = np.array(np.interp(RS.p0_half, pressure_data, q_data),dtype=np.double, order='c')
        nudge_temperature = np.array(np.interp(RS.p0_half, pressure_data, temperature_data),dtype=np.double, order='c')
        self.nudge_u = np.array(np.interp(RS.p0_half, pressure_data, u_data),dtype=np.double, order='c')
        self.nudge_v = np.array(np.interp(RS.p0_half, pressure_data, v_data),dtype=np.double, order='c')

        # Obtain the moisture floor and find the max index corresponding to z <= 1300 m

        self.qt_floor = np.interp(1300.0, Gr.zl_half, self.nudge_qt)
        print("qt floor = " + str(self.qt_floor))
        cdef Py_ssize_t k = Gr.dims.gw

        for k in range(Gr.dims.gw, Gr.dims.nlg[2]-Gr.dims.gw):
            if Gr.zl_half[k] > 1300.0:
                break
            self.floor_index = k

        print("floor imdex" + str(self.floor_index))
        print(str(Gr.zl_half[self.floor_index]))




        # Nudging profile is unsaturated, so we can calculate entropy simply
        cdef double qv, pd, pv


        with nogil:
            for k in range(Gr.dims.nlg[2]):
                pd = pd_c(RS.p0_half[k], self.nudge_qt[k], self.nudge_qt[k])
                pv = pv_c(RS.p0_half[k], self.nudge_qt[k], self.nudge_qt[k])
                self.nudge_s[k] = sd_c(pd, nudge_temperature[k]) * (1.0 - self.nudge_qt[k]) \
                                  + sv_c(pv, nudge_temperature[k]) * self.nudge_qt[k]

        self.gamma_zhalf = np.zeros((Gr.dims.nlg[2]),dtype=np.double,order='c')
        self.gamma_z = np.zeros((Gr.dims.nlg[2]), dtype=np.double, order='c')

        with nogil:
            for k in range(Gr.dims.nlg[2]):
                if Gr.zl_half[k] < self.z_relax:
                    self.gamma_zhalf[k] = 0.0
                elif Gr.zl_half[k] > self.z_relax_plus:
                    self.gamma_zhalf[k] = self.tau_inverse
                else:
                    self.gamma_zhalf[k] = 0.5*self.tau_inverse * (1.0 - cos(pi* (Gr.zl_half[k]-self.z_relax)/(self.z_relax_plus-self.z_relax)))

                if Gr.zl[k] < self.z_relax:
                    self.gamma_z[k] = 0.0
                elif Gr.zl[k] > self.z_relax_plus:
                    self.gamma_z[k] = self.tau_inverse
                else:
                    self.gamma_z[k] = 0.5*self.tau_inverse * (1.0 - cos(pi* (Gr.zl[k]-self.z_relax)/(self.z_relax_plus-self.z_relax)))


        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        # Here we apply the free tropospheric damping without considering an entropy source due to moisture damping
        # However, we do apply an entropy source due to the in-BL moisture floor damping of the S12 case
        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t w_shift = PV.get_varshift(Gr, 'w')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr,'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr,'temperature')
            Py_ssize_t qv_shift = DV.get_varshift(Gr, 'qv')
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i, j, k, ishift, jshift, ijk
            double [:] u_mean = Pa.HorizontalMean(Gr, &PV.values[u_shift])
            double [:] v_mean = Pa.HorizontalMean(Gr, &PV.values[v_shift])

            double[:] domain_mean
            double qv, qt, pd, pv, t, qt_floor_nudge


        with nogil:
            for i in xrange(imin, imax):
                ishift = i * istride
                for j in xrange(jmin, jmax):
                    jshift = j * jstride
                    for k in xrange(kmin, kmax):
                        ijk = ishift + jshift + k
                        # Nudge w to zero in free troposphere
                        PV.tendencies[w_shift + ijk] -= (PV.values[w_shift + ijk]) * self.gamma_zhalf[k]
                        # Nudge s, qt to reference profiles in free troposphere
                        PV.tendencies[s_shift + ijk] -= (PV.values[s_shift + ijk] - self.nudge_s[k]) * self.gamma_z[k]
                        PV.tendencies[qt_shift + ijk] -= (PV.values[qt_shift + ijk] - self.nudge_qt[k]) * self.gamma_z[k]
                        # Nudge mean wind profiles through entire depth
                        PV.tendencies[u_shift + ijk] -= (u_mean[k] + RS.u0 - self.nudge_u[k]) * self.tau_vel_inverse
                        PV.tendencies[v_shift + ijk] -= (v_mean[k] + RS.v0 - self.nudge_v[k]) * self.tau_vel_inverse




        # Moisture floor nudging for S12 case
        if self.loc == 12:
            domain_mean = Pa.HorizontalMean(Gr, & PV.values[qt_shift])
            if np.amin(domain_mean[Gr.dims.gw:self.floor_index]) < self.qt_floor:
                with nogil:
                    for k in xrange(kmin, self.floor_index):
                        if domain_mean[k] < self.qt_floor:
                            qt_floor_nudge = -(domain_mean[k] - self.qt_floor) * self.tau_inverse
                            for i in xrange(imin, imax):
                                ishift = i * istride
                                for j in xrange(jmin, jmax):
                                    jshift = j * jstride
                                    ijk = ishift + jshift + k
                                    qt = PV.values[qt_shift + ijk]
                                    qv = DV.values[qv_shift + ijk]
                                    t = DV.values[t_shift + ijk]
                                    pv = pv_c(RS.p0_half[k],qt, qv)
                                    pd = pd_c(RS.p0_half[k],qt, qv)
                                    PV.tendencies[qt_shift + ijk] += qt_floor_nudge
                                    PV.tendencies[s_shift + ijk] += (sv_c(pv,t) - sd_c(pd,t)) * qt_floor_nudge


        return
