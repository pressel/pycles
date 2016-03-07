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

import pylab as plt
import numpy as np
cimport numpy as np
from libc.math cimport pow, cbrt, exp, sin, cos, sqrt, fmax
include 'parameters.pxi'

cdef class Radiation:
    def __init__(self, namelist, ParallelMPI.ParallelMPI Pa):
        casename = namelist['meta']['casename']
        try:
            fixed_heating = namelist['radiation']['fixed_heating']
        except:
            fixed_heating = False
        if casename == 'DYCOMS_RF01':
            if fixed_heating:
                self.scheme = RadiationFixedHeatingProfile()
                Pa.root_print('FHP Radiation')
            else:
                self.scheme = RadiationDyCOMS_RF01()
                Pa.root_print('Standard DYCOMS Radiation')
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
        self.scheme.stats_io(Gr, Ref, PV, DV, NS, Pa)
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
        NS.add_profile('radiative_heating_rate', Gr, Pa)
        NS.add_profile('radiative_entropy_tendency', Gr, Pa)
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
            double [:] heating_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] entropy_tendency = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
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
            double[:] tmp

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
                        entropy_tendency[ijk] =  heating_rate[ijk] / DV.values[ijk + t_shift]



        tmp = Pa.HorizontalMean(Gr, &heating_rate[0])
        NS.write_profile('radiative_heating_rate', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &entropy_tendency[0])
        NS.write_profile('radiative_entropy_tendency', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

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
        NS.add_profile('radiative_heating_rate', Gr, Pa)
        NS.add_profile('radiative_entropy_tendency', Gr, Pa)
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
            double [:] heating_rate = np.zeros((Gr.dims.npg, ), dtype=np.double, order='c')
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
            double [:] heating_rate = np.zeros((Gr.dims.npg, ), dtype=np.double, order='c')
            double [:] entropy_tendency = np.zeros((Gr.dims.npg, ), dtype=np.double, order='c')
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
            double [:] tmp


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
                        entropy_tendency[ijk] =  heating_rate[ijk] / DV.values[ijk + t_shift]


        tmp = Pa.HorizontalMean(Gr, &heating_rate[0])
        NS.write_profile('radiative_heating_rate', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &entropy_tendency[0])
        NS.write_profile('radiative_entropy_tendency', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)


        return



cdef class RadiationEUROCS_Sc:
    def __init__(self):
        self.z_pencil = ParallelMPI.Pencil()
        # Parameters related to parameterized LW radiation

        # Temporal and location information for calculating solar cycle
        self.year = 1987
        self.month = 7
        self.day = 14
        self.hour = 8.0 # GMT
        self.latitude = 33.25
        self.longitude = 119.5
        # Parameters related to the delta Eddington parameterization for SW radiation

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


        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                   PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        return




cdef class RadiationFixedHeatingProfile:
    def __init__(self):
        self.dFdz_5m = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 -0.0110111162932, -0.0330333488796, -0.055055581466, -0.0770778140524, -0.113946316803,
                                 -0.165661089718, -0.217375862634, -0.269090635549, -0.305635225157, -0.327009631458,
                                 -0.348384037759, -0.369758444061, -0.362931181328, -0.327902249563, -0.292873317797,
                                 -0.257844386031, -0.220287337083, -0.180202170954, -0.140117004824, -0.100031838695,
                                 -0.0718207314612, -0.0554836831237, -0.0391466347862, -0.0228095864488, -0.0129951489849,
                                 -0.00970332239466, -0.00641149580441, -0.00311966921416, -0.00129965603319, -0.000951456261504,
                                 -0.000603256489817, -0.000255056718129, -7.11385481613e-05, -5.15019799131e-05, -3.18654116649e-05,
                                 -1.22288434168e-05, -2.11408084586e-06, -1.52112395219e-06, -9.28167058508e-07, -3.3521016483e-07,
                                 -3.39258469927e-08, -2.43141049965e-08, -1.47023630003e-08, -5.09062100414e-09,
                                 8.47229484465e-09, 2.5986384546e-08, 4.35004742474e-08, 6.10145639488e-08,
                                 2.26816959916e-05, 6.79055447571e-05, 0.000113129393523, 0.000158353242288,
                                 0.109555947736, 0.328305912874, 0.547055878013, 0.765805843151, 0.984510601853, 1.20317015412,
                                 1.42182970638, 1.64048925865, 1.65585853326, 1.46793753022, 1.28001652717, 1.09209552413,
                                 0.908318592065, 0.728685730983, 0.549052869901, 0.369420008819, 0.271269235152,
                                 0.2546005489, 0.237931862648, 0.221263176396, 0.198760256033, 0.170423101559, 0.142085947086,
                                 0.113748792612, 0.0969877066781, 0.0918026892849, 0.0866176718916, 0.0814326544984,
                                 0.0774081123178, 0.0745440453497, 0.0716799783817, 0.0688159114137, 0.066447694487,
                                 0.0645753276016, 0.0627029607162, 0.0608305938309, 0.0592262984832, 0.0578900746731,
                                 0.056553850863, 0.055217627053, 0.0540455581507, 0.0530376441564, 0.052029730162,
                                 0.0510218161676, 0.0501227701077, 0.0493325919823, 0.0485424138569, 0.0477522357315,
                                 0.04703843338, 0.0464010068025, 0.0457635802249, 0.0451261536473, 0.0445446075897,
                                 0.044018942052, 0.0434932765142, 0.0429676109765, 0.0424841907238, 0.0420430157561,
                                 0.0416018407884, 0.0411606658207, 0.0407522780686, 0.0403766775322, 0.0400010769958,
                                 0.0396254764593, 0.0392758777585, 0.0389522808932, 0.038628684028, 0.0383050871627,
                                 0.0380024868994, 0.0377208832381, 0.0374392795768, 0.0371576759155, 0.0368932894206,
                                 0.0366461200919, 0.0363989507633, 0.0361517814346, 0.0359189179506, 0.0357003603112,
                                 0.0354818026718, 0.0352632450324, 0.03505671167, 0.0348622025847, 0.0346676934994,
                                 0.034473184414, 0.034288884488, 0.0341147937214, 0.0339407029548, 0.0337666121882,
                                 0.0336012673067, 0.0334446683102, 0.0332880693138, 0.0331314703173, 0.0329824236294,
                                 0.0328409292498, 0.0326994348703, 0.0325579404908, 0.0324230145521, 0.0322946570544,
                                 0.0321662995566, 0.0320379420588, 0.0319153343006, 0.0317984762818, 0.0316816182631,
                                 0.0315647602444, 0.0314529648584, 0.0313462321052, 0.031239499352, 0.0311327665988,
                                 0.0310305153204, 0.0309327455169, 0.0308349757133, 0.0307372059098, 0.0306434225397,
                                 0.0305536256032, 0.0304638286667, 0.0303740317301, 0.0302877967941, 0.0302051238587,
                                 0.0301224509233, 0.0300397779878, 0.029960300977, 0.0298840198908, 0.0298077388046,
                                 0.0297314577184, 0.0296580550683, 0.0295875308543, 0.0295170066403, 0.0294464824263,
                                 0.0293785598891, 0.0293132390287, 0.0292479181684, 0.029182597308, 0.029119635723,
                                 0.0290590334134, 0.0289984311038, 0.0289378287941, 0.0288793725117, 0.0288230622564,
                                 0.0287667520012, 0.0287104417459, 0.0286560891422, 0.0286036941901, 0.028551299238,
                                 0.028498904286, 0.0284482999388, 0.0283994861965, 0.0283506724542, 0.028301858712,
                                 0.0282546869056, 0.0282091570352, 0.0281636271648, 0.0281180972944, 0.0280740765969,
                                 0.0280315650723, 0.0279890535476, 0.027946542023, 0.027905420733, 0.0278656896776, 0.0278259586222,
                                 0.0277862275668, 0.0277477798715, 0.0277106155362, 0.0276734512009, 0.0276362868656, 0.0276003095822,
                                 0.0275655193508, 0.0275307291194, 0.0274959388879, 0.0274622486882, 0.0274296585201, 0.027397068352,
                                 0.027364478184, 0.0273329092188, 0.0273023614564, 0.027271813694, 0.0272412659317, 0.0272116677907,
                                 0.0271830192711, 0.0271543707515, 0.0271257222319, 0.0270979581838, 0.0270710786071, 0.0270441990305,
                                 0.0270173194538, 0.0269912649225, 0.0269660354367, 0.0269408059508, 0.026915576465, 0.0268911177065,
                                 0.0268674296755, 0.0268437416445, 0.0268200536135, 0.0276269597196, 0.0292644599629, 0.0309019602062,
                                 0.0325394604494, 0.0320991973025, 0.0295811707654, 0.0270631442283, 0.0245451176912, 0.0232861044227, 0.0232861044227 ])
        d = 5.0
        self.h_5m =np.arange(d/2.0,1800 + d/2.0, d)

        return

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        self.dFdz = np.interp(Gr.zl_half,self.h_5m,self.dFdz_5m)
        NS.add_profile('radiative_heating_rate', Gr, Pa)
        NS.add_profile('radiative_entropy_tendency', Gr, Pa)


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
            Py_ssize_t gw = Gr.dims.gw
            double heating_rate
            double[:] rho_half = Ref.rho0_half


        # Now update entropy tendencies
        with nogil:
            for i in xrange(imin, imax):
                ishift = i * istride
                for j in xrange(jmin, jmax):
                    jshift = j * jstride
                    for k in xrange(kmin, kmax):
                        ijk = ishift + jshift + k
                        heating_rate = -self.dFdz[k] /rho_half[k]
                        PV.tendencies[
                            s_shift + ijk] +=  heating_rate / DV.values[ijk + t_shift]

        return
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                   PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):


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
            Py_ssize_t gw = Gr.dims.gw
            double [:] heating_rate = np.zeros((Gr.dims.npg, ), dtype=np.double, order='c')
            double [:] entropy_tendency = np.zeros((Gr.dims.npg, ), dtype=np.double, order='c')
            double [:] rho_half = Ref.rho0_half
            double [:] tmp


        # Now update entropy tendencies
        with nogil:
            for i in xrange(imin, imax):
                ishift = i * istride
                for j in xrange(jmin, jmax):
                    jshift = j * jstride
                    for k in xrange(kmin, kmax):
                        ijk = ishift + jshift + k
                        heating_rate[ijk] = -self.dFdz[k] /rho_half[k]
                        entropy_tendency[ijk] =  heating_rate[ijk] / DV.values[ijk + t_shift]


        tmp = Pa.HorizontalMean(Gr, &heating_rate[0])
        NS.write_profile('radiative_heating_rate', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &entropy_tendency[0])
        NS.write_profile('radiative_entropy_tendency', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

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
