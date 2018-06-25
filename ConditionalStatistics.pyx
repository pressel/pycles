#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

from scipy.fftpack import fft, ifft
cimport Grid
cimport ReferenceState
cimport DiagnosticVariables
cimport PrognosticVariables
cimport ParallelMPI
from NetCDFIO cimport NetCDFIO_CondStats
import cython
cimport numpy as np
import numpy as np
from libc.math cimport sqrt, ceil
from thermodynamic_functions cimport thetas_c
include "parameters.pxi"

cdef class ConditionalStatistics:
    def __init__(self, namelist):
        self.CondStatsClasses = []


    cpdef initialize(self, namelist, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                               DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_CondStats NC, ParallelMPI.ParallelMPI Pa):

        try:
            conditional_statistics = namelist['conditional_stats']['classes']
        except:
            conditional_statistics = ['Null']


        #Convert whatever is in twodimensional_statistics to list if not already
        if not type(conditional_statistics) == list:
            conditional_statistics = [conditional_statistics]

        #Build list of twodimensional statistics class instances
        if 'Spectra' in conditional_statistics:
            self.CondStatsClasses.append(SpectraStatistics(Gr,PV, DV, NC, Pa))
        if 'Null' in conditional_statistics:
            self.CondStatsClasses.append(NullCondStats())
        # # __
        # if 'NanStatistics' in conditional_statistics:
        #     self.CondStatsClasses.append(NanStatistics(Gr, PV, DV, NC, Pa))
        # # if 'Test' in conditional_statistics:
        # #     self.CondStatsClasses.append(TestStatistics(Gr, PV, DV, NC, Pa))
        # # __
        #
        # print('CondStatsClasses: ', self.CondStatsClasses)
        return


    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_CondStats NC, ParallelMPI.ParallelMPI Pa):

        #loop over class instances and class stats_io
        for _class in self.CondStatsClasses:
            _class.stats_io(Gr, RS, PV, DV, NC, Pa)

        return

cdef class NullCondStats:
    def __init__(self) :
        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,  NetCDFIO_CondStats NC, ParallelMPI.ParallelMPI Pa):
        return


cdef class SpectraStatistics:
    def __init__(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 NetCDFIO_CondStats NC, ParallelMPI.ParallelMPI Pa):
        Pa.root_print('SpectraStatistics initialized')
        cdef:
            Py_ssize_t ii, i,  jj, j
            double xi, yj

        # Set up the wavenumber vectors
        self.nwave = int( np.ceil(np.sqrt(2.0) * (Gr.dims.n[0] + 1.0) * 0.5 ) + 1.0)
        self.dk = 2.0 * pi/(Gr.dims.n[0]*Gr.dims.dx[0])
        self.wavenumbers = np.arange(self.nwave, dtype=np.double) * self.dk

        self.kx = np.zeros(Gr.dims.nl[0],dtype=np.double,order='c')
        self.ky = np.zeros(Gr.dims.nl[1],dtype=np.double,order='c')

        for ii in xrange(Gr.dims.nl[0]):
            i = Gr.dims.indx_lo[0] + ii
            if i <= (Gr.dims.n[0])/2:
                xi = np.double(i)
            else:
                xi = np.double(i - Gr.dims.n[0])
            self.kx[ii] = xi * self.dk
        for jj in xrange(Gr.dims.nl[1]):
            j = Gr.dims.indx_lo[1] + jj
            if j <= Gr.dims.n[1]/2:
                yj = np.double(j)
            else:
                yj = np.double(j-Gr.dims.n[1])
            self.ky[jj] = yj * self.dk

        NC.create_condstats_group('spectra','wavenumber', self.wavenumbers, Gr, Pa)


        # set up the names of the variables
        NC.add_condstat('energy_spectrum', 'spectra', 'wavenumber', Gr, Pa)
        if 's' in PV.name_index:
            NC.add_condstat('s_spectrum', 'spectra', 'wavenumber', Gr, Pa)
        if 'qt' in PV.name_index:
            NC.add_condstat('qt_spectrum', 'spectra', 'wavenumber', Gr, Pa)
        if 'theta_rho' in DV.name_index:
            NC.add_condstat('theta_rho_spectrum', 'spectra', 'wavenumber', Gr, Pa)
        if 'thetali' in DV.name_index:
            NC.add_condstat('thetali_spectrum', 'spectra', 'wavenumber', Gr, Pa)
        if 'theta' in DV.name_index:
            NC.add_condstat('theta_spectrum', 'spectra', 'wavenumber', Gr, Pa)
        if 'qt_variance' in DV.name_index:
            NC.add_condstat('qtvar_spectrum', 'spectra', 'wavenumber', Gr, Pa)
        if 'qt_variance_clip' in DV.name_index:
            NC.add_condstat('qtvarclip_spectrum', 'spectra', 'wavenumber', Gr, Pa)
        if 's_variance' in DV.name_index:
            NC.add_condstat('svar_spectrum', 'spectra', 'wavenumber', Gr, Pa)
        if 'covariance' in DV.name_index:
            NC.add_condstat('covar_spectrum', 'spectra', 'wavenumber', Gr, Pa)

        if 's' in PV.name_index and 'qt' in PV.name_index:
            NC.add_condstat('s_qt_cospectrum', 'spectra', 'wavenumber', Gr, Pa)


        #Instantiate classes used for Pencil communication/transposes
        self.X_Pencil = ParallelMPI.Pencil()
        self.Y_Pencil = ParallelMPI.Pencil()


        #Initialize classes used for Pencil communication/tranposes (here dim corresponds to the pencil direction)
        self.X_Pencil.initialize(Gr,Pa,dim=0)
        self.Y_Pencil.initialize(Gr,Pa,dim=1)


        # # _____
        # Pa.root_print('NanStatistics initialization')
        #
        # self.sk_arr = np.zeros((1,2),dtype=np.double)
        # self.qtk_arr = np.zeros((1,2),dtype=np.double)
        #
        # nz = np.arange(Gr.dims.n[2], dtype=np.double) * Gr.dims.dx[2]
        # # NC.create_condstats_group('nan_array','nz', nz, Gr, Pa)
        # # set up the names of the variables
        # NC.add_condstat('sk_arr', 'spectra', 'wavenumber', Gr, Pa)
        # NC.add_condstat('qtk_arr', 'spectra', 'wavenumber', Gr, Pa)


        return


    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,  NetCDFIO_CondStats NC, ParallelMPI.ParallelMPI Pa):

        Pa.root_print('calling ConditionalStatistics.SpectraStatistics stats_io')
        cdef:
            Py_ssize_t i, j, k,  ijk, var_shift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t ishift
            Py_ssize_t jshift
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t w_shift = PV.get_varshift(Gr, 'w')
            complex [:] data_fft= np.zeros(Gr.dims.npg,dtype=np.complex,order='c')
            complex [:] data_fft_s= np.zeros(Gr.dims.npg,dtype=np.complex,order='c')
            double [:] uc = np.zeros(Gr.dims.npg,dtype=np.double,order='c')
            double [:] vc = np.zeros(Gr.dims.npg,dtype=np.double,order='c')
            double [:] wc = np.zeros(Gr.dims.npg,dtype=np.double,order='c')
            Py_ssize_t npg = Gr.dims.npg
            Py_ssize_t gw = Gr.dims.gw
            double [:,:] spec_u, spec_v, spec_w, spec




        #Interpolate to cell centers
        with nogil:
            for i in xrange(1, Gr.dims.nlg[0]):
                ishift = i * istride
                for j in xrange(1, Gr.dims.nlg[1]):
                    jshift = j * jstride
                    for k in xrange(1, Gr.dims.nlg[2]):
                        ijk = ishift + jshift + k
                        uc[ijk] = 0.5 * (PV.values[u_shift + ijk - istride] + PV.values[u_shift + ijk])
                        vc[ijk] = 0.5 * (PV.values[v_shift + ijk - jstride] + PV.values[v_shift + ijk])
                        wc[ijk] = 0.5 * (PV.values[w_shift + ijk - 1] + PV.values[w_shift + ijk])


        self.fluctuation_forward_transform(Gr, Pa, uc[:], data_fft[:])
        spec_u = self.compute_spectrum(Gr, Pa,  data_fft[:])
        self.fluctuation_forward_transform(Gr, Pa, vc[:], data_fft[:])
        spec_v = self.compute_spectrum(Gr, Pa,  data_fft[:])
        self.fluctuation_forward_transform(Gr, Pa, wc[:], data_fft[:])
        spec_w = self.compute_spectrum(Gr, Pa,  data_fft[:])

        spec = np.add(np.add(spec_u,spec_v), spec_w)
        NC.write_condstat('energy_spectrum', 'spectra', spec[:,:], Pa)

        if 's' in PV.name_index:
            var_shift = PV.get_varshift(Gr, 's')
            self.fluctuation_forward_transform(Gr, Pa, PV.values[var_shift:var_shift+npg], data_fft_s[:])
            spec = self.compute_spectrum(Gr, Pa,  data_fft_s[:])
            NC.write_condstat('s_spectrum', 'spectra', spec[:,:], Pa)
        if 'qt' in PV.name_index:
            var_shift = PV.get_varshift(Gr, 'qt')
            self.fluctuation_forward_transform(Gr, Pa, PV.values[var_shift:var_shift+npg], data_fft[:])
            spec = self.compute_spectrum(Gr, Pa,  data_fft[:])
            NC.write_condstat('qt_spectrum', 'spectra', spec[:,:], Pa)
        if 's' in PV.name_index and 'qt' in PV.name_index:
            spec = self.compute_cospectrum(Gr, Pa, data_fft_s[:], data_fft[:])
            NC.write_condstat('s_qt_cospectrum', 'spectra', spec[:,:], Pa)


        if 'theta_rho' in DV.name_index:
            var_shift = DV.get_varshift(Gr, 'theta_rho')
            self.fluctuation_forward_transform(Gr, Pa, DV.values[var_shift:var_shift+npg], data_fft[:])
            spec = self.compute_spectrum(Gr, Pa,  data_fft[:])
            NC.write_condstat('theta_rho_spectrum', 'spectra', spec[:,:], Pa)
        if 'thetali' in DV.name_index:
            var_shift = DV.get_varshift(Gr, 'thetali')
            self.fluctuation_forward_transform(Gr, Pa, DV.values[var_shift:var_shift+npg], data_fft[:])
            spec = self.compute_spectrum(Gr, Pa,  data_fft[:])
            NC.write_condstat('thetali_spectrum', 'spectra', spec[:,:], Pa)
        if 'theta' in DV.name_index:
            var_shift = DV.get_varshift(Gr, 'theta')
            self.fluctuation_forward_transform(Gr, Pa, DV.values[var_shift:var_shift+npg], data_fft[:])
            spec = self.compute_spectrum(Gr, Pa,  data_fft[:])
            NC.write_condstat('theta_spectrum', 'spectra', spec[:,:], Pa)

        if 'qt_variance' in DV.name_index:
            var_shift = DV.get_varshift(Gr, 'qt_variance')
            self.fluctuation_forward_transform(Gr, Pa, DV.values[var_shift:var_shift+npg], data_fft[:])
            spec = self.compute_spectrum(Gr, Pa,  data_fft[:])
            NC.write_condstat('qtvar_spectrum', 'spectra', spec[:,:], Pa)

        if 'qt_variance_clip' in DV.name_index:
            var_shift = DV.get_varshift(Gr, 'qt_variance_clip')
            self.fluctuation_forward_transform(Gr, Pa, DV.values[var_shift:var_shift+npg], data_fft[:])
            spec = self.compute_spectrum(Gr, Pa,  data_fft[:])
            NC.write_condstat('qtvarclip_spectrum', 'spectra', spec[:,:], Pa)

        if 's_variance' in DV.name_index:
            var_shift = DV.get_varshift(Gr, 's_variance')
            self.fluctuation_forward_transform(Gr, Pa, DV.values[var_shift:var_shift+npg], data_fft[:])
            spec = self.compute_spectrum(Gr, Pa,  data_fft[:])
            NC.write_condstat('svar_spectrum', 'spectra', spec[:,:], Pa)

        if 'covariance' in DV.name_index:
            var_shift = DV.get_varshift(Gr, 'covariance')
            self.fluctuation_forward_transform(Gr, Pa, DV.values[var_shift:var_shift+npg], data_fft[:])
            spec = self.compute_spectrum(Gr, Pa,  data_fft[:])
            NC.write_condstat('covar_spectrum', 'spectra', spec[:,:], Pa)



        return

    cpdef forward_transform(self, Grid.Grid Gr,ParallelMPI.ParallelMPI Pa, double [:] data, complex [:] data_fft):
        cdef:
            double [:,:] x_pencil
            complex [:,:] x_pencil_fft,  y_pencil, y_pencil_fft


        #Do fft in x direction
        x_pencil = self.X_Pencil.forward_double(&Gr.dims, Pa, &data[0])
        x_pencil_fft = fft(x_pencil,axis=1)
        self.X_Pencil.reverse_complex(&Gr.dims, Pa, x_pencil_fft, &data_fft[0])

        #Do fft in y direction
        y_pencil = self.Y_Pencil.forward_complex(&Gr.dims, Pa, &data_fft[0])
        y_pencil_fft = fft(y_pencil,axis=1)
        self.Y_Pencil.reverse_complex(&Gr.dims, Pa, y_pencil_fft, &data_fft[0])

        return



    cpdef fluctuation_forward_transform(self, Grid.Grid Gr,ParallelMPI.ParallelMPI Pa, double [:] data, complex [:] data_fft):
        cdef:
            double [:,:] x_pencil
            complex [:,:] x_pencil_fft,  y_pencil, y_pencil_fft
            Py_ssize_t i, j, k,  ijk
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t ishift
            Py_ssize_t jshift
            double [:] fluctuation = np.zeros(Gr.dims.npg,dtype=np.double,order='c')
        cdef:
            double [:] data_mean = Pa.HorizontalMean(Gr, &data[0])

        with nogil:
            for i in xrange(1, Gr.dims.nlg[0]):
                ishift = i * istride
                for j in xrange(1, Gr.dims.nlg[1]):
                    jshift = j * jstride
                    for k in xrange(1, Gr.dims.nlg[2]):
                        ijk = ishift + jshift + k

                        #Compute fluctuations
                        fluctuation[ijk] = data[ijk] - data_mean[k]

        #Do fft in x direction
        x_pencil = self.X_Pencil.forward_double(&Gr.dims, Pa, &fluctuation[0])
        x_pencil_fft = fft(x_pencil,axis=1)
        self.X_Pencil.reverse_complex(&Gr.dims, Pa, x_pencil_fft, &data_fft[0])

        #Do fft in y direction
        y_pencil = self.Y_Pencil.forward_complex(&Gr.dims, Pa, &data_fft[0])
        y_pencil_fft = fft(y_pencil,axis=1)
        self.Y_Pencil.reverse_complex(&Gr.dims, Pa, y_pencil_fft, &data_fft[0])

        del fluctuation

        return





    cpdef compute_spectrum(self, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa, complex [:] data_fft ):
        cdef:
            Py_ssize_t i, j, k, ijk, ik, kg, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t nwave = self.nwave
            double [:] kx = self.kx
            double [:] ky = self.ky
            double dk = self.dk
            double kmag
            double [:,:] spec = np.zeros((Gr.dims.nl[2],self.nwave),dtype=np.double, order ='c')

        with nogil:
            for i in xrange(Gr.dims.nl[0]):
                ishift = (i + gw) * istride
                for j in xrange(Gr.dims.nl[1]):
                    jshift = (j + gw) * jstride
                    kmag = sqrt(kx[i]*kx[i] + ky[j]*ky[j])
                    ik = int(ceil(kmag/dk + 0.5) - 1.0)
                    for k in xrange(Gr.dims.nl[2]):
                        kg = k + gw
                        ijk = ishift + jshift + kg
                        spec[k, ik] += data_fft[ijk].real *  data_fft[ijk].real +  data_fft[ijk].imag *  data_fft[ijk].imag

        for k in xrange(Gr.dims.nl[2]):
            for ik in xrange(nwave):
                spec[k, ik] = Pa.domain_scalar_sum(spec[k,ik])

        return spec




    cpdef compute_cospectrum(self, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa, complex [:] data_fft_1,  complex [:] data_fft_2):
        cdef:
            Py_ssize_t i, j, k, ijk, ik, kg, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t nwave = self.nwave
            double [:] kx = self.kx
            double [:] ky = self.ky
            double dk = self.dk
            double kmag, R1, R2
            double [:,:] spec = np.zeros((Gr.dims.nl[2],self.nwave),dtype=np.double, order ='c')

        with nogil:
            for i in xrange(Gr.dims.nl[0]):
                ishift = (i + gw) * istride
                for j in xrange(Gr.dims.nl[1]):
                    jshift = (j + gw) * jstride
                    kmag = sqrt(kx[i]*kx[i] + ky[j]*ky[j])
                    ik = int(ceil(kmag/dk + 0.5) - 1.0)
                    for k in xrange(Gr.dims.nl[2]):
                        kg = k + gw
                        ijk = ishift + jshift + kg
                        R1 = sqrt(data_fft_1[ijk].real *  data_fft_1[ijk].real +  data_fft_1[ijk].imag *  data_fft_1[ijk].imag)
                        R2 = sqrt(data_fft_2[ijk].real *  data_fft_2[ijk].real +  data_fft_2[ijk].imag *  data_fft_2[ijk].imag)
                        spec[k, ik] += R1*R2

        for k in xrange(Gr.dims.nl[2]):
            for ik in xrange(nwave):
                spec[k, ik] = Pa.domain_scalar_sum(spec[k,ik])

        return spec

#
# # __________
# cdef class NanStatistics:
#     def __init__(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
#                  NetCDFIO_CondStats NC, ParallelMPI.ParallelMPI Pa):
#         Pa.root_print('NanStatistics initialized')
#         # cdef:
#         #    Py_ssize_t nz = Gr.dims.n[2]
#
#         # self.sk_arr = np.zeros((1,2),dtype=np.double)
#         # self.qtk_arr = np.zeros((1,2),dtype=np.double)
#         self.sk_arr = np.zeros((Gr.dims.npd),dtype=np.double)
#         self.qtk_arr = np.zeros((Gr.dims.npd),dtype=np.double)
#
#         # nz = np.arange(Gr.dims.n[2], dtype=np.double) * Gr.dims.dx[2]
#         # NC.create_condstats_group('nan_array','nz', nz, Gr, Pa)
#         nz = np.arange(Gr.dims.npd, dtype=np.double)
#         NC.create_condstats_group('nan_array','nz',nz, Gr, Pa)
#         # set up the names of the variables
#         NC.add_condstat('sk_arr', 'nan_array', 'nz', Gr, Pa)
#         NC.add_condstat('qtk_arr', 'nan_array', 'nz', Gr, Pa)
#
#
#         ## from NetCDFIO_CondStats:
#         # root_grp = nc.Dataset(self.path_plus_file, 'w', format='NETCDF4')
#         # sub_grp = root_grp.createGroup(groupname)
#         # sub_grp.createDimension('z', Gr.dims.n[2])
#         # sub_grp.createDimension(dimname, len(dimval))
#         # sub_grp.createDimension('t', None)
#         # z = sub_grp.createVariable('z', 'f8', ('z'))
#         # z[:] = np.array(Gr.z[Gr.dims.gw:-Gr.dims.gw])
#         # dim = sub_grp.createVariable(dimname, 'f8', (dimname))
#         # dim[:] = np.array(dimval[:])
#         # sub_grp.createVariable('t', 'f8', ('t'))
#
#         return
#
#
#     cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
#                  DiagnosticVariables.DiagnosticVariables DV,  NetCDFIO_CondStats NC, ParallelMPI.ParallelMPI Pa):
#
#         Pa.root_print('!!!! calling ConditionalStatistics.NanStatistics stats_io')
#
#         # _____
#         message = 'hi'
#         print('sk_arr before:', self.sk_arr)
#         self.nan_checking(message,Gr,PV,DV,NC,Pa)
#
#
#         print('sk_arr after:', self.sk_arr)
#         if 's' in PV.name_index:
#             NC.write_condstat('sk_arr', 'nan_array', self.sk_arr[:,:], Pa)
#         if 'qt' in PV.name_index:
#             NC.write_condstat('qtk_arr', 'nan_array', self.qt_arr[:,:], Pa)
#
#         return
#
#
#
#     # def debug_tend(self,message):
#     cpdef nan_checking(self,message, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
#                        DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_CondStats NC, ParallelMPI.ParallelMPI Pa):
#         print('nan_checking')
#         cdef:
#             PrognosticVariables.PrognosticVariables PV_ = PV
#             DiagnosticVariables.DiagnosticVariables DV_ = DV
#             Grid.Grid Gr_ = Gr
#
#         cdef:
#             Py_ssize_t u_varshift = PV_.get_varshift(Gr,'u')
#             Py_ssize_t v_varshift = PV_.get_varshift(Gr,'v')
#             Py_ssize_t w_varshift = PV_.get_varshift(Gr,'w')
#             Py_ssize_t s_varshift = PV_.get_varshift(Gr,'s')
#
#             Py_ssize_t istride = Gr_.dims.nlg[1] * Gr_.dims.nlg[2]
#             Py_ssize_t jstride = Gr_.dims.nlg[2]
#             Py_ssize_t imax = Gr_.dims.nlg[0]
#             Py_ssize_t jmax = Gr_.dims.nlg[1]
#             Py_ssize_t kmax = Gr_.dims.nlg[2]
#             Py_ssize_t ijk_max = imax*istride + jmax*jstride + kmax
#
#             Py_ssize_t i, j, k, ijk, ishift, jshift
#             Py_ssize_t imin = 0#Gr_.dims.gw
#             Py_ssize_t jmin = 0#Gr_.dims.gw
#             Py_ssize_t kmin = 0#Gr_.dims.gw
#
#         # __
#         PV_.values[u_varshift+1] = np.nan
#         # __
#
#         u_max = np.nanmax(PV_.tendencies[u_varshift:v_varshift])
#         uk_max = np.nanargmax(PV_.tendencies[u_varshift:v_varshift])
#         u_min = np.nanmin(PV_.tendencies[u_varshift:v_varshift])
#         uk_min = np.nanargmin(PV_.tendencies[u_varshift:v_varshift])
#         v_max = np.nanmax(PV_.tendencies[v_varshift:w_varshift])
#         vk_max = np.nanargmax(PV_.tendencies[v_varshift:w_varshift])
#         v_min = np.nanmin(PV_.tendencies[v_varshift:w_varshift])
#         vk_min = np.nanargmin(PV_.tendencies[v_varshift:w_varshift])
#         w_max = np.nanmax(PV_.tendencies[w_varshift:s_varshift])
#         wk_max = np.nanargmax(PV_.tendencies[w_varshift:s_varshift])
#         w_min = np.nanmin(PV_.tendencies[w_varshift:s_varshift])
#         wk_min = np.nanargmin(PV_.tendencies[w_varshift:s_varshift])
#
#         u_nan = np.isnan(PV_.tendencies[u_varshift:v_varshift]).any()
#         uk_nan = np.argmax(PV_.tendencies[u_varshift:v_varshift])
#         v_nan = np.isnan(PV_.tendencies[v_varshift:w_varshift]).any()
#         vk_nan = np.argmax(PV_.tendencies[v_varshift:w_varshift])
#         w_nan = np.isnan(PV_.tendencies[w_varshift:s_varshift]).any()
#         wk_nan = np.argmax(PV_.tendencies[w_varshift:s_varshift])
#
#         if Pa.rank == 0:
#             print(message, 'debugging (max, min, nan): ')
#             print('shifts', u_varshift, v_varshift, w_varshift, s_varshift)
#             print('u tend: ', u_max, uk_max, u_min, uk_min, u_nan, uk_nan)
#             print('v tend: ', v_max, vk_max, v_min, vk_min, v_nan, vk_nan)
#             print('w tend: ', w_max, wk_max, w_min, wk_min, w_nan, wk_nan)
#
#         if 'qt' in PV_.name_index:
#             qt_varshift = PV_.get_varshift(Gr,'qt')
#             ql_varshift = DV_.get_varshift(Gr,'ql')
#
#             s_max = np.nanmax(PV_.tendencies[s_varshift:qt_varshift])
#             sk_max = np.nanargmax(PV_.tendencies[s_varshift:qt_varshift])
#             s_min = np.nanmin(PV_.tendencies[s_varshift:qt_varshift])
#             sk_min = np.nanargmin(PV_.tendencies[s_varshift:qt_varshift])
#             qt_max = np.nanmax(PV_.tendencies[qt_varshift:-1])
#             qtk_max = np.nanargmax(PV_.tendencies[qt_varshift:-1])
#             qt_min = np.nanmin(PV_.tendencies[qt_varshift:-1])
#             qtk_min = np.nanargmin(PV_.tendencies[qt_varshift:-1])
#
#             s_nan = np.isnan(PV_.tendencies[s_varshift:qt_varshift]).any()
#             sk_nan = np.argmax(PV_.tendencies[s_varshift:qt_varshift])
#             qt_nan = np.isnan(PV_.tendencies[qt_varshift:-1]).any()
#             qtk_nan = np.argmax(PV_.tendencies[qt_varshift:-1])
#
#             s_max_val= np.nanmax(PV_.values[s_varshift:qt_varshift])
#             sk_max_val = np.nanargmax(PV_.values[s_varshift:qt_varshift])
#             s_min_val = np.nanmin(PV_.values[s_varshift:qt_varshift])
#             sk_min_val = np.nanargmin(PV_.tendencies[s_varshift:qt_varshift])
#             s_nan_val = np.isnan(PV_.values[s_varshift:qt_varshift]).any()
#             sk_nan_val = np.argmax(PV_.values[s_varshift:qt_varshift])
#             qt_max_val = np.nanmax(PV_.values[qt_varshift:-1])
#             qtk_max_val = np.nanargmax(PV_.values[qt_varshift:-1])
#             qt_min_val = np.nanmin(PV_.values[qt_varshift:-1])
#             if qt_min_val < 0:
#                 Pa.root_print('qt val negative')
#             qtk_min_val = np.nanargmin(PV_.values[qt_varshift:-1])
#             qt_nan_val = np.isnan(PV_.values[qt_varshift:-1]).any()
#             qtk_nan_val = np.argmax(PV_.values[qt_varshift:-1])
#
#             ql_max_val = np.nanmax(DV_.values[ql_varshift:(ql_varshift+ijk_max)])
#             ql_min_val = np.nanmin(DV_.values[ql_varshift:(ql_varshift+ijk_max)])
#             qlk_max_val = np.nanargmax(DV_.values[ql_varshift:(ql_varshift+ijk_max)])
#             qlk_min_val = np.nanargmin(DV_.values[ql_varshift:(ql_varshift+ijk_max)])
#             ql_nan_val = np.isnan(DV_.values[ql_varshift:(ql_varshift+ijk_max)]).any()
#             qlk_nan_val = np.argmax(DV_.values[ql_varshift:(ql_varshift+ijk_max)])
#
#             if Pa.rank == 0:
#                 print('s tend: ', s_max, sk_max, s_min, sk_min, s_nan, sk_nan)
#                 print('s val: ', s_max_val, sk_max_val, s_min_val, sk_min_val, s_nan_val, sk_nan_val)
#                 print('qt tend: ', qt_max, qtk_max, qt_min, qtk_min, qt_nan, qtk_nan)
#                 print('qt val: ', qt_max_val, qtk_max_val, qt_min_val, qtk_min_val, qt_nan_val, qtk_nan_val)
#                 print('ql val: ', ql_max_val, qlk_max_val, ql_min_val, qlk_min_val, ql_nan_val, qlk_nan_val)
#
#
#         #for name in PV.name_index.keys():
#             # with nogil:
#             if 1 == 1:
#                 for i in range(imin, imax):
#                     ishift = i * istride
#                     for j in range(jmin, jmax):
#                         jshift = j * jstride
#                         for k in range(kmin, kmax):
#                             ijk = ishift + jshift + k
#                             if np.isnan(PV_.values[s_varshift+ijk]):
#                                 self.sk_arr = np.append(self.sk_arr,np.array([[ijk,k]]),axis=0)
#                             if np.isnan(PV_.values[qt_varshift+ijk]):
#                                 self.qtk_arr = np.append(self.qtk_arr,np.array([[ijk,k]]),axis=0)
#             if np.size(self.sk_arr) > 1 or np.size(self.qtk_arr) > 1:
#                 self.output_nan_array(Gr, PV, DV, NC, Pa)
#             # if np.size(self.sk_arr) > 1:
#             #     if self.Pa.rank == 0:
#             #         print('sk_arr size: ', self.sk_arr.shape)
#             #         print('sk_arr:', self.sk_arr)
#             # if np.size(self.qtk_arr) > 1:
#             #     if self.Pa.rank == 0:
#             #         print('qtk_arr size: ', self.qtk_arr.shape)
#             #         print('qtk_arr: ', self.qtk_arr)
#
#         else:
#             s_max = np.nanmax(PV_.tendencies[s_varshift:-1])
#             sk_max = np.nanargmax(PV_.tendencies[s_varshift:-1])
#             s_min = np.nanmin(PV_.tendencies[s_varshift:-1])
#             sk_min = np.nanargmin(PV_.tendencies[s_varshift:-1])
#             s_nan = np.isnan(PV_.tendencies[s_varshift:-1]).any()
#             sk_nan = np.argmax(PV_.tendencies[s_varshift:-1])
#
#             s_max_val= np.nanmax(PV_.values[s_varshift:-1])
#             sk_max_val = np.nanargmax(PV_.values[s_varshift:-1])
#             s_min_val = np.nanmin(PV_.values[s_varshift:-1])
#             sk_min_val = np.nanargmin(PV_.tendencies[s_varshift:-1])
#             s_nan_val = np.isnan(PV_.values[s_varshift:-1]).any()
#             sk_nan_val = np.argmax(PV_.values[s_varshift:-1])
#
#             if Pa.rank == 0:
#                 print('s tend: ', s_max, sk_max, s_min, sk_min, s_nan, sk_nan)
#                 print('s val: ', s_max_val, sk_max_val, s_min_val, sk_min_val, s_nan_val, sk_nan_val)
#
#
#             if 1 == 1:
#                 for i in range(imin, imax):
#                     ishift = i * istride
#                     for j in range(jmin, jmax):
#                         jshift = j * jstride
#                         for k in range(kmin, kmax):
#                             ijk = ishift + jshift + k
#                             if np.isnan(PV_.values[s_varshift+ijk]):
#                                 self.sk_arr = np.append(self.sk_arr,np.array([[ijk,k]]),axis=0)
#             if np.size(self.sk_arr) > 1:
#                 self.output_nan_array(Gr, PV, DV, NC, Pa)
# #                 if self.Pa.rank == 0:
# #                     print('sk_arr size: ', self.sk_arr.shape)
# #                     print('sk_arr:', self.sk_arr)
#         return
#
#
#     cpdef output_nan_array(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
#                  DiagnosticVariables.DiagnosticVariables DV,  NetCDFIO_CondStats NC, ParallelMPI.ParallelMPI Pa):
#
#         if 's' in PV.name_index:
#             NC.write_condstat('sk_arr', 'nan_array', self.sk_arr[:,:], Pa)
#         if 'qt' in PV.name_index:
#             NC.write_condstat('qtk_arr', 'nan_array', self.qt_arr[:,:], Pa)
#
#         return

