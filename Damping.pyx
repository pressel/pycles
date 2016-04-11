#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

from libc.math cimport fmin, fmax, sin
import cython
import numpy as np
cimport ParallelMPI as ParallelMPI
cimport PrognosticVariables as PrognosticVariables
cimport Grid as Grid
cimport numpy as np


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

    cpdef initialize(self, Grid.Grid Gr):
        self.scheme.initialize(Gr)
        return

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI Pa):
        self.scheme.update(Gr, PV, Pa)
        return

cdef class Dummy:
    def __init__(self, namelist, ParallelMPI.ParallelMPI Pa):
        return

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI Pa):
        return

    cpdef initialize(self, Grid.Grid Gr):
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

    cpdef initialize(self, Grid.Grid Gr):
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

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI Pa):

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
            loc = namelist['meta']['CGILS']['location']
            if loc !=12 and loc != 11 and loc != 6:
                Pa.root_print('Invalid CGILS location (must be 6, 11, or 12)')
                Pa.kill()
        except:
            Pa.root_print('Must provide a CGILS location (6/11/12) in namelist')
            Pa.kill()
        try:
            is_p2 = namelist['meta']['CGILS']['P2']
        except:
            Pa.root_print('Must specify if CGILS run is perturbed')
            Pa.kill()

        if loc == 12:
            self.z_relax = 1200.0
            self.z_relax_plus = 1500.0
        elif loc == 11:
            self.z_relax = 2500.0
            self.z_relax_plus = 3000.0
        elif loc == 6:
            self.z_relax = 4000.0
            self.z_relax_plus = 4800.0

        return

    cpdef initialize(self, Grid.Grid Gr):

        if is_p2:
            file = './CGILSdata/p2k_s'+str(loc)+'.nc'
        else:
            file = './CGILSdata/ctl_s'+str(loc)+'.nc'

        print(file)
        data = nc.Dataset(file, 'r')
        # Get the profile information we need from the data file to set up nudging profiles
        # Since we are sure the the range of LES pressures that are nudged are within the range of the
        # pressure data, simple linear interpolation will suffice
        pressure_data = data.variables['lev'][:]
        temperature_data = data.variables['T'][0,:,0,0]
        q_data = data.variables['q'][0,:,0,0]
        u_data = data.variables['u'][0,:,0,0]
        v_data = data.variables['v'][0,:,0,0]

        self.nudge_qt = np.zeros((Gr.dims.nlg[2],),dtype=np.double, order='c')
        self.nudge_s = np.zeros((Gr.dims.nlg[2],),dtype=np.double, order='c')
        self.nudge_u = np.zeros((Gr.dims.nlg[2],),dtype=np.double, order='c')
        self.nudge_v = np.zeros((Gr.dims.nlg[2],),dtype=np.double, order='c')



        # cdef:
        #     int k
        #     double z_top
        #
        # self.gamma_zhalf = np.zeros(
        #     (Gr.dims.nlg[2]),
        #     dtype=np.double,
        #     order='c')
        # self.gamma_z = np.zeros((Gr.dims.nlg[2]), dtype=np.double, order='c')
        # z_top = Gr.dims.dx[2] * Gr.dims.n[2]
        # with nogil:
        #     for k in range(Gr.dims.nlg[2]):
        #         if Gr.zl_half[k] >= z_top - self.z_d:
        #             self.gamma_zhalf[
        #                 k] = self.gamma_r * sin((pi / 2.0) * (1.0 - (z_top - Gr.zl_half[k]) / self.z_d))**2.0
        #         if Gr.zl[k] >= z_top - self.z_d:
        #             self.gamma_z[
        #                 k] = self.gamma_r * sin((pi / 2.0) * (1.0 - (z_top - Gr.zl[k]) / self.z_d))**2.0
        return

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI Pa):

        # cdef:
        #     Py_ssize_t var_shift
        #     Py_ssize_t imin = Gr.dims.gw
        #     Py_ssize_t jmin = Gr.dims.gw
        #     Py_ssize_t kmin = Gr.dims.gw
        #     Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
        #     Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
        #     Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw
        #     Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
        #     Py_ssize_t jstride = Gr.dims.nlg[2]
        #     Py_ssize_t i, j, k, ishift, jshift, ijk
        #     double[:] domain_mean
        #
        # for var_name in PV.name_index:
        #     var_shift = PV.get_varshift(Gr, var_name)
        #     domain_mean = Pa.HorizontalMean(Gr, & PV.values[var_shift])
        #     if var_name == 'w':
        #         with nogil:
        #             for i in xrange(imin, imax):
        #                 ishift = i * istride
        #                 for j in xrange(jmin, jmax):
        #                     jshift = j * jstride
        #                     for k in xrange(kmin, kmax):
        #                         ijk = ishift + jshift + k
        #                         PV.tendencies[var_shift + ijk] -= (PV.values[var_shift + ijk] - domain_mean[k]) * self.gamma_zhalf[k]
        #     else:
        #         with nogil:
        #             for i in xrange(imin, imax):
        #                 ishift = i * istride
        #                 for j in xrange(jmin, jmax):
        #                     jshift = j * jstride
        #                     for k in xrange(kmin, kmax):
        #                         ijk = ishift + jshift + k
        #                         PV.tendencies[var_shift + ijk] -= (PV.values[var_shift + ijk] - domain_mean[k]) * self.gamma_z[k]
        return
