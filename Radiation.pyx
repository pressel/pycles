#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI
cimport TimeStepping


import numpy as np
cimport numpy as np
from libc.math cimport pow, cbrt, exp, sin, cos, sqrt, fmax
include 'parameters.pxi'

cdef class Radiation:
    def __init__(self, namelist, ParallelMPI.ParallelMPI Pa):
        casename = namelist['meta']['casename']
        if casename == 'DYCOMS_RF01':
            self.scheme = RadiationDyCOMS_RF01()
        elif casename == 'DYCOMS_RF02':
            #Dycoms RF01 and RF02 use the same radiation
            self.scheme = RadiationDyCOMS_RF01()
        elif casename == 'SMOKE':
            self.scheme = RadiationSmoke()
        elif casename == 'EUROCS_Sc':
            self.scheme = RadiationEUROCS_Sc()
        else:
            self.scheme = RadiationNone()
        return

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.scheme.initialize(Gr, NS, Pa)
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):
        self.scheme.update(Gr, Ref, PV, DV, Pa, TS)
        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                   PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.scheme.stats_io(Gr, PV, DV, NS, Pa)
        return


cdef class RadiationNone:
    def __init__(self):
        return
    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):
        return
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                   PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return


cdef class RadiationDyCOMS_RF01:
    def __init__(self):
        self.alpha_z = 1.0
        self.kap = 85.0
        self.f0 = 70.0
        self.f1 = 22.0
        self.divergence = 3.75e-6

        self.z_pencil = ParallelMPI.Pencil()
        return

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.z_pencil.initialize(Gr, Pa, 2)
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        cdef:
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw

            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw

            Py_ssize_t pi, i, j, k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t gw = Gr.dims.gw
            double [:, :] ql_pencils =  self.z_pencil.forward_double(&Gr.dims, Pa, &DV.values[ql_shift])
            double [:, :] qt_pencils =  self.z_pencil.forward_double(&Gr.dims, Pa, &PV.values[qt_shift])
            double[:, :] f_rad = np.empty((self.z_pencil.n_local_pencils, Gr.dims.n[2] + 1), dtype=np.double, order='c')
            double[:, :] f_heat = np.empty((self.z_pencil.n_local_pencils, Gr.dims.n[2]), dtype=np.double, order='c')
            double[:] heating_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double q_0
            double q_1

            double zi
            double rhoi
            double dz = Gr.dims.dx[2]
            double dzi = Gr.dims.dxi[2]
            double[:] z = Gr.z
            double[:] rho = Ref.rho0
            double[:] rho_half = Ref.rho0_half
            double cbrt_z = 0

        with nogil:
            for pi in xrange(self.z_pencil.n_local_pencils):

                # Compute zi (level of 8.0 g/kg isoline of qt)
                for k in xrange(Gr.dims.n[2]):
                    if qt_pencils[pi, k] > 8e-3:
                        zi = z[gw + k]
                        rhoi = rho_half[gw + k]

                # Now compute the third term on RHS of Stevens et al 2005
                # (equation 3)
                f_rad[pi, 0] = 0.0
                for k in xrange(Gr.dims.n[2]):
                    if z[gw + k] >= zi:
                        cbrt_z = cbrt(z[gw + k] - zi)
                        f_rad[pi, k + 1] = rhoi * cpd * self.divergence * self.alpha_z * (pow(cbrt_z,4)  / 4.0
                                                                                     + zi * cbrt_z)
                    else:
                        f_rad[pi, k + 1] = 0.0

                # Compute the second term on RHS of Stevens et al. 2005
                # (equation 3)
                q_1 = 0.0
                f_rad[pi, 0] += self.f1 * exp(-q_1)
                for k in xrange(1, Gr.dims.n[2] + 1):
                    q_1 += self.kap * \
                        rho_half[gw + k - 1] * ql_pencils[pi, k - 1] * dz
                    f_rad[pi, k] += self.f1 * exp(-q_1)

                # Compute the first term on RHS of Stevens et al. 2005
                # (equation 3)
                q_0 = 0.0
                f_rad[pi, Gr.dims.n[2]] += self.f0 * exp(-q_0)
                for k in xrange(Gr.dims.n[2] - 1, -1, -1):
                    q_0 += self.kap * rho_half[gw + k] * ql_pencils[pi, k] * dz
                    f_rad[pi, k] += self.f0 * exp(-q_0)

                for k in xrange(Gr.dims.n[2]):
                    f_heat[pi, k] = - \
                       (f_rad[pi, k + 1] - f_rad[pi, k]) * dzi / rho_half[k]

        # Now transpose the flux pencils
        self.z_pencil.reverse_double(&Gr.dims, Pa, f_heat, &heating_rate[0])


        # Now update entropy tendencies
        with nogil:
            for i in xrange(imin, imax):
                ishift = i * istride
                for j in xrange(jmin, jmax):
                    jshift = j * jstride
                    for k in xrange(kmin, kmax):
                        ijk = ishift + jshift + k
                        PV.tendencies[
                            s_shift + ijk] +=  heating_rate[ijk] / DV.values[ijk + t_shift]

        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                   PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        return


cdef class RadiationSmoke:
    '''
    Radiation for the smoke cloud case

    Bretherton, C. S., and coauthors, 1999:
    An intercomparison of radiatively- driven entrainment and turbulence in a smoke cloud,
    as simulated by different numerical models. Quart. J. Roy. Meteor. Soc., 125, 391-423. Full text copy.

    '''


    def __init__(self):
        self.f0 = 60.0
        self.kap = 0.02
        self.z_pencil = ParallelMPI.Pencil()
        return

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.z_pencil.initialize(Gr, Pa, 2)
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        cdef:
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw

            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw

            Py_ssize_t pi, i, j, k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t smoke_shift = PV.get_varshift(Gr, 'smoke')
            Py_ssize_t gw = Gr.dims.gw
            double [:, :] smoke_pencils =  self.z_pencil.forward_double(&Gr.dims, Pa, &PV.values[smoke_shift])
            double[:, :] f_rad = np.zeros((self.z_pencil.n_local_pencils, Gr.dims.n[2] + 1), dtype=np.double, order='c')
            double[:, :] f_heat = np.zeros((self.z_pencil.n_local_pencils, Gr.dims.n[2]), dtype=np.double, order='c')
            double[:] heating_rate = np.zeros((Gr.dims.npg, ), dtype=np.double, order='c')
            double q_0

            double zi
            double rhoi
            double dz = Gr.dims.dx[2]
            double dzi = Gr.dims.dxi[2]
            double[:] z = Gr.z
            double[:] rho = Ref.rho0
            double[:] rho_half = Ref.rho0_half
            double cbrt_z = 0
            Py_ssize_t kk


        with nogil:
            for pi in xrange(self.z_pencil.n_local_pencils):

                q_0 = 0.0
                f_rad[pi, Gr.dims.n[2]] = self.f0 * exp(-q_0)
                for k in xrange(Gr.dims.n[2] - 1, -1, -1):
                    q_0 += self.kap * rho_half[gw + k] * smoke_pencils[pi, k] * dz
                    f_rad[pi, k] = self.f0 * exp(-q_0)

                for k in xrange(Gr.dims.n[2]):
                    f_heat[pi, k] = - \
                       (f_rad[pi, k + 1] - f_rad[pi, k]) * dzi / rho_half[k]

        # Now transpose the flux pencils
        self.z_pencil.reverse_double(&Gr.dims, Pa, f_heat, &heating_rate[0])


        # Now update entropy tendencies
        with nogil:
            for i in xrange(imin, imax):
                ishift = i * istride
                for j in xrange(jmin, jmax):
                    jshift = j * jstride
                    for k in xrange(kmin, kmax):
                        ijk = ishift + jshift + k
                        PV.tendencies[
                            s_shift + ijk] +=  heating_rate[ijk] / DV.values[ijk + t_shift]

        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                   PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        return



cdef class RadiationEUROCS_Sc:
    def __init__(self):
        self.z_pencil = ParallelMPI.Pencil()
        # Parameters related to parameterized LW radiation
        self.a = 130.0 #m2/kg
        self.density = 1.14 # kg/m^3
        self.deltaFL = 70.0 # W/m^2
        # Temporal and location information for calculating solar cycle
        self.year = 1987
        self.month = 7
        self.day = 14
        self.hour = 8.0 # GMT
        self.latitude = 33.25
        self.longitude = 119.5
        # Parameters related to the delta Eddington parameterization for SW radiation
        self.omega_de = 0.9864
        self.F0_max = 1100.0
        self.reff = 1.0e-5 # assumed effective droplet radius of 10 micrometers
        self.asf = 0.06 # Surface albedo

        return

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.z_pencil.initialize(Gr, Pa, 2)
        Pa.root_print('Initialized EUROCS_Sc radiation')
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        cdef:
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw

            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw

            Py_ssize_t ip, i, j, k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t gw = Gr.dims.gw
            double [:, :] ql_pencils =  self.z_pencil.forward_double(&Gr.dims, Pa, &DV.values[ql_shift])
            double [:, :] f_rad_lw = np.zeros((self.z_pencil.n_local_pencils, Gr.dims.n[2] + 1), dtype=np.double, order='c')
            double [:, :] f_rad_sw = np.zeros((self.z_pencil.n_local_pencils, Gr.dims.n[2] + 1), dtype=np.double, order='c')
            double [:, :] f_heat = np.empty((self.z_pencil.n_local_pencils, Gr.dims.n[2]), dtype=np.double, order='c')
            double [:] heating_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double  lwp_down
            double rhoi
            double dz = Gr.dims.dx[2]
            double dzi = Gr.dims.dxi[2]
            double[:] z = Gr.z
            double a11,a12,a21,a22,b1,b2,C1,C2,det,tau,tau_tot
            double column_lwp
            double[:] rho_half = Ref.rho0_half


        if TS.rk_step == 0.0:
            self.hour = self.hour + TS.dt/3600.0
            if self.hour > 24.0:
                self.hour = self.hour - 24.0
                self.day = self.day + 1

        # Shortwave parameters depending on solar zenith angle
        cdef double amu0 = cosine_zenith_angle(self.year, self.month, self.day, self.hour, self.latitude, self.longitude)
        amu0 = fmax(amu0, 1.0e-5)
        Pa.root_print('Day = ' + str(self.day) + ' Hour = ' + str(self.hour) + 'cos(zen) = ' + str(np.round(amu0, decimals=2)))
        cdef double F0 = self.F0_max * amu0
        cdef double gp = self.g_de /(1.0 + self.g_de)
        cdef double omp = (1.0 - self.g_de * self.g_de)* self.omega_de/(1.0- self.omega_de * self.g_de * self.g_de)
        cdef double kap = sqrt(3.0 * (1.0 - self.omega_de) * (1.0 - self.omega_de * self.g_de))
        cdef double p = sqrt(3.0*(1.0 - self.omega_de)/(1.0 - self.omega_de * self.g_de))
        cdef double alpha = 0.75 * self.omega_de * amu0 * amu0 * (1.0 + self.g_de * (1.0 - self.omega_de))/(1.0- kap * kap * amu0 * amu0)
        cdef double beta = 0.75 * self.omega_de * amu0 * (1.0 + 3.0 * self.g_de *
                                                     (1.0 - self.omega_de) * amu0 * amu0)/(1.0- kap * kap * amu0 * amu0)



        with nogil:
            for ip in xrange(self.z_pencil.n_local_pencils):

                # Compute the net UPWARD longwave
                lwp_down = 0.0
                f_rad_lw[ip, Gr.dims.n[2]] = 0.0
                for k in xrange(Gr.dims.n[2] - 1, -1, -1):
                    lwp_down += self.density * ql_pencils[ip, k] * dz
                    f_rad_lw[ip, k] = self.deltaFL * exp(-self.a * lwp_down)

                column_lwp = lwp_down
                tau_tot = 1.5 * column_lwp/ (self.reff * liquid_density)
                a11 = (1.0 - self.asf - 2.0 * (1.0 + self.asf) * p / 3.0) * exp(-kap * tau_tot)
                a12 = (1.0 - self.asf + 2.0 * (1.0 + self.asf) * p / 3.0) *exp( kap*tau_tot)
                b1  = ((1.0 - self.asf) * alpha - 2.0 * (1.0 + self.asf) * beta/3.0  + self.asf * amu0) * exp(-tau_tot/amu0)
                a21 = 1.0 + 2.0/3.0 * p
                a22 = 1.0-2.0/3.0 * p
                b2  = alpha + 2.0/3.0 * beta
                det = a11 * a22 - a12 * a21
                C1 = (a22 * b1 - a12 * b2)/det
                C2 = (a11 * b2 - a21 * b1)/det

                lwp_down = 0.0
                # Compute the net DOWNWARD shortwave
                for k in xrange(Gr.dims.n[2], -1, -1):
                    lwp_down += self.density * ql_pencils[ip, k] * dz
                    tau = 1.5 * lwp_down / (self.reff * liquid_density)
                    f_rad_sw[ip, k] = (0.75 * F0 *(p * (C1 * exp(-kap*tau) - C2 * exp(kap * tau)) - beta * exp(-tau/amu0))
                                       + amu0 * F0 * exp(-tau/amu0))

                for k in xrange(Gr.dims.n[2]):
                    f_heat[ip, k] = -((f_rad_lw[ip, k+1] - f_rad_sw[ip, k+1]) - (f_rad_lw[ip, k] - f_rad_sw[ip, k])) * dzi / rho_half[k]

        # Now transpose the flux pencils
        self.z_pencil.reverse_double(&Gr.dims, Pa, f_heat, &heating_rate[0])


        # Now update entropy tendencies
        with nogil:
            for i in xrange(imin, imax):
                ishift = i * istride
                for j in xrange(jmin, jmax):
                    jshift = j * jstride
                    for k in xrange(kmin, kmax):
                        ijk = ishift + jshift + k
                        PV.tendencies[
                            s_shift + ijk] +=  heating_rate[ijk] / DV.values[ijk + t_shift]

        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                   PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        return



''' Calculation of cos(solar zenith angle) depending on date, time, lat, lon
    Adapted from STREAMER (zenith.f)
    Original reference:    Woolf, H.M., NASA TM X-1646.
                           ON THE COMPUTATION OF SOLAR ELEVATION
                           ANGLES AND THE DETERMINATION OF SUNRISE
                           AND SUNSET TIMES.
 '''

cdef double cosine_zenith_angle(Py_ssize_t year, Py_ssize_t month, Py_ssize_t day, double hour_Z, double dlat, double dlon):
    # hour_Z is decimal hour GMT
    # dlat, dlon : Latitude (positive in northern hemisphere), longitude (0 to +- 180 OR 0 to 360, positive west of Prime Mer.)
    jday_list = [0.0, 31.0, 59.0, 90.0, 120.0, 151.0, 181.0,212.0, 243.0, 273.0, 304.0, 334.0]
    cdef:
        double jday = jday_list[month - 1]
        double sigma, ang, cozena
        double  dang                # angle measured from perihelion in radians
        double  homp                # true solar noon (Hours Of Meridian Passage)
        double  hang               #hour angle
        double  sindlt, cosdlt     # sine, cosine of declination angle
        double epsiln = 0.016733 # eccentricity of earth's orbit
        double sinob = 0.3978   # sine of obliquity of the ecliptic
        double dpy = 365.242   # days per year
        double dph = 15.0        # degrees per hour (360/24)



    if month > 2:
        if year%4 == 0 and year%100 != 0 or year%400 == 0:
           jday = jday + 1.0

    jday = jday + day

    dang = 2.0 * pi * (jday-1.0)/dpy

    homp = 12.0 + 0.12357 * sin(dang) - 0.004289 * cos(dang) +  0.153809 *sin(2.0*dang) + 0.060783*cos(2.0*dang)

    hang = dph * (hour_Z-homp) - dlon
    ang = 279.9348 * pi/180.0 + dang
    sigma = (ang * 180.0 / pi + 0.4087 * sin(ang) + 1.8724*cos(ang) - 0.0182*sin(2.0*ang)+0.0083 * cos(2.0*ang))*pi/180.0
    sindlt = sinob*sin(sigma)
    cosdlt = sqrt(1.0-sindlt*sindlt)
    cozena = sindlt*sin(pi/180.0*dlat) + cosdlt*cos(pi/180.0*dlat)*cos(pi/180.0*hang)


    return cozena
