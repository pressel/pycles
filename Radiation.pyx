cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI

import pylab as plt
import numpy as np
cimport numpy as np
from libc.math cimport pow, cbrt, exp

include 'parameters.pxi'

cdef class Radiation:
    def __init__(self, namelist, ParallelMPI.ParallelMPI Pa):
        casename = namelist['meta']['casename']
        if casename == 'DYCOMS_RF01':
            try:
                fixed_heating = namelist['radiation']['fixed_heating']
            except:
                fixed_heating = False
            if fixed_heating:
                self.scheme = RadiationDyCOMSFixedHeating()
            else:
                self.scheme = RadiationDyCOMS_RF01()
        elif casename == 'DYCOMS_RF02':
            #Dycoms RF01 and RF02 use the same radiation
            self.scheme = RadiationDyCOMS_RF01()
        elif casename == 'SMOKE':
            self.scheme = RadiationSmoke()
        else:
            self.scheme = RadiationNone()
        return

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.scheme.initialize(Gr, NS, Pa)
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 ParallelMPI.ParallelMPI Pa):
        self.scheme.update(Gr, Ref, PV, DV, Pa)
        return

    cpdef stats_io(self, Grid.Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.scheme.stats_io(Gr, DV, NS, Pa)
        return


cdef class RadiationNone:
    def __init__(self):
        return
    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 ParallelMPI.ParallelMPI Pa):
        return
    cpdef stats_io(self, Grid.Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
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
        self.heating_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        NS.add_profile('radiative_heating_rate', Gr, Pa)
        NS.add_profile('radiative_entropy_tendency', Gr, Pa)

        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw

            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw

            Py_ssize_t ipen, i, j, k, ijk, ishift, jshift
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
            for ipen in xrange(self.z_pencil.n_local_pencils):

                # Compute zi (level of 8.0 g/kg isoline of qt)
                for k in xrange(Gr.dims.n[2]):
                    if qt_pencils[ipen, k] > 8e-3:
                        zi = z[gw + k]
                        rhoi = rho_half[gw + k]

                # Now compute the third term on RHS of Stevens et al 2005
                # (equation 3)
                f_rad[ipen, 0] = 0.0
                for k in xrange(Gr.dims.n[2]):
                    if z[gw + k] >= zi:
                        cbrt_z = cbrt(z[gw + k] - zi)
                        f_rad[ipen, k + 1] = rhoi * cpd * self.divergence * self.alpha_z * (pow(cbrt_z,4)  / 4.0
                                                                                     + zi * cbrt_z)
                    else:
                        f_rad[ipen, k + 1] = 0.0

                # Compute the second term on RHS of Stevens et al. 2005
                # (equation 3)
                q_1 = 0.0
                f_rad[ipen, 0] += self.f1 * exp(-q_1)
                for k in xrange(1, Gr.dims.n[2] + 1):
                    q_1 += self.kap * \
                        rho_half[gw + k - 1] * ql_pencils[ipen, k - 1] * dz
                    f_rad[ipen, k] += self.f1 * exp(-q_1)

                # Compute the first term on RHS of Stevens et al. 2005
                # (equation 3)
                q_0 = 0.0
                f_rad[ipen, Gr.dims.n[2]] += self.f0 * exp(-q_0)
                for k in xrange(Gr.dims.n[2] - 1, -1, -1):
                    q_0 += self.kap * rho_half[gw + k] * ql_pencils[ipen, k] * dz
                    f_rad[ipen, k] += self.f0 * exp(-q_0)

                for k in xrange(Gr.dims.n[2]):
                    f_heat[ipen, k] = - \
                       (f_rad[ipen, k + 1] - f_rad[ipen, k]) * dzi / rho_half[k]

        # Now transpose the flux pencils
        self.z_pencil.reverse_double(&Gr.dims, Pa, f_heat, &self.heating_rate[0])


        # Now update entropy tendencies
        with nogil:
            for i in xrange(imin, imax):
                ishift = i * istride
                for j in xrange(jmin, jmax):
                    jshift = j * jstride
                    for k in xrange(kmin, kmax):
                        ijk = ishift + jshift + k
                        PV.tendencies[
                            s_shift + ijk] +=  self.heating_rate[ijk] / DV.values[ijk + t_shift]

        return

    cpdef stats_io(self, Grid.Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw

            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw

            Py_ssize_t i, j, k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]

            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            double [:] entropy_tendency = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] tmp

        # Now update entropy tendencies
        with nogil:
            for i in xrange(imin, imax):
                ishift = i * istride
                for j in xrange(jmin, jmax):
                    jshift = j * jstride
                    for k in xrange(kmin, kmax):
                        ijk = ishift + jshift + k
                        entropy_tendency[ijk] =  self.heating_rate[ijk] / DV.values[ijk + t_shift]

        tmp = Pa.HorizontalMean(Gr, &self.heating_rate[0])
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
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw

            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw

            Py_ssize_t ipen, i, j, k, ijk, ishift, jshift
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
            for ipen in xrange(self.z_pencil.n_local_pencils):

                q_0 = 0.0
                f_rad[ipen, Gr.dims.n[2]] = self.f0 * exp(-q_0)
                for k in xrange(Gr.dims.n[2] - 1, -1, -1):
                    q_0 += self.kap * rho_half[gw + k] * smoke_pencils[ipen, k] * dz
                    f_rad[ipen, k] = self.f0 * exp(-q_0)

                for k in xrange(Gr.dims.n[2]):
                    f_heat[ipen, k] = - \
                       (f_rad[ipen, k + 1] - f_rad[ipen, k]) * dzi / rho_half[k]

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

    cpdef stats_io(self, Grid.Grid Gr, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        return





cdef class RadiationDyCOMSFixedHeating:
    def __init__(self):
        alpha_z = 1.0
        kap = 85.0
        f0 = 70.0
        f1 = 22.0
        divergence = 3.75e-6
        dens = 1.13


        # Create a heating rate profile using the observed liquid amounts and assumed cloud top height
        ### obs = np.array([0.08943104,0.13815998,0.28252313,0.29262888])
        ### hobs =np.array( [632.1025,647.1799,768.43787,765.1738])
        # Fitting coefficients
        m = 0.00137814
        b = -0.76845122
        # Then plug this into the DYCOMS radiation parameterization
        zi = 840.0
        dz = 30.0
        h_faces = np.arange(0.0, 1800.0+dz, dz, dtype=np.double)
        nh = len(h_faces) - 1
        self.heating_grid = np.zeros((nh,),dtype=np.double, order='c')

        for i in range(0,nh):
            self.heating_grid[i] = (h_faces[i+1] + h_faces[i])/2.0
        # ql defined on centers (nh)
        ql  = np.zeros((nh,),dtype=np.double, order='c')
        # lwp, frad defined on faces (nh+1)
        lwp_up  = np.zeros((nh+1,),dtype=np.double, order='c')
        lwp_down = np.zeros((nh+1,),dtype=np.double, order='c')
        frad = np.zeros((nh+1,),dtype=np.double, order='c')
        self.heating_profile = np.zeros((nh,),dtype=np.double, order='c')

        for i in range(0,nh):
            ql[i] = np.maximum(m * self.heating_grid[i] + b, 0.0)
            if self.heating_grid[i] > zi:
                ql[i] = 0.0
            lwp_up[i+1] = lwp_up[i] + ql[i] * dz * dens

        for i in range(nh-1, -1, -1):
            lwp_down[i] = lwp_down[i+1] + ql[i] * dz * dens

        for i in range(0,nh+1):
            frad[i] = f0 * np.exp(-lwp_down[i]) + f1 * np.exp(-lwp_up[i])
            if h_faces[i] >= zi:
                frad[i] += dens*1015.0*divergence*alpha_z*( 0.25*(h_faces[i]-zi)**(4.0/3.0) + zi*(h_faces[i]-zi)**(1.0/3.0))



        for i in range(0,nh):
            self.heating_profile[i] = -(frad[i+1]-frad[i])/dz/1.13

        return

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        interped_profile = np.interp(Gr.zl_half,self.heating_grid,self.heating_profile)

        self.heating_rate = np.array(interped_profile, dtype=np.double, copy=True, order='c')

        NS.add_profile('radiative_heating_rate', Gr, Pa)
        NS.add_profile('radiative_entropy_tendency', Gr, Pa)

        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw

            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw

            Py_ssize_t  i, j, k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')


        # Now update entropy tendencies
        with nogil:
            for i in xrange(imin, imax):
                ishift = i * istride
                for j in xrange(jmin, jmax):
                    jshift = j * jstride
                    for k in xrange(kmin, kmax):
                        ijk = ishift + jshift + k
                        PV.tendencies[
                            s_shift + ijk] +=  self.heating_rate[k] / DV.values[ijk + t_shift]

        return

    cpdef stats_io(self, Grid.Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw

            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw

            Py_ssize_t i, j, k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]

            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            double [:] entropy_tendency = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] tmp

        # Now compute entropy tendencies
        with nogil:
            for i in xrange(imin, imax):
                ishift = i * istride
                for j in xrange(jmin, jmax):
                    jshift = j * jstride
                    for k in xrange(kmin, kmax):
                        ijk = ishift + jshift + k
                        entropy_tendency[ijk] =  self.heating_rate[k] / DV.values[ijk + t_shift]

        NS.write_profile('radiative_heating_rate', self.heating_rate[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &entropy_tendency[0])
        NS.write_profile('radiative_entropy_tendency', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)


        return
