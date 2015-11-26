#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport DiagnosticVariables
cimport PrognosticVariables
cimport ParallelMPI
cimport MomentumAdvection
cimport MomentumDiffusion
from NetCDFIO cimport NetCDFIO_Stats
import cython
cimport numpy as np
import numpy as np
from libc.math cimport sqrt
from thermodynamic_functions cimport thetas_c
include "parameters.pxi"

class AuxiliaryStatistics:
    def __init__(self, namelist):
        self.AuxStatsClasses = []
        return

    def initialize(self, namelist, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                               DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        try:
            auxiliary_statistics = namelist['stats_io']['auxiliary']
        except:
            return

        #Convert whatever is in auxiliary_statistics to list if not already
        if not type(auxiliary_statistics) == list:
            auxiliary_statistics = [auxiliary_statistics]

        #Build list of auxilary statistics class instances
        if 'Cumulus' in auxiliary_statistics:
            self.AuxStatsClasses.append(CumulusStatistics(Gr,PV, DV, NS, Pa))
        if 'StableBL' in auxiliary_statistics:
            self.AuxStatsClasses.append(StableBLStatistics(Gr, NS, Pa))
        if 'SMOKE' in auxiliary_statistics:
            self.AuxStatsClasses.append(SmokeStatistics(Gr, NS, Pa))
        if 'DYCOMS' in auxiliary_statistics:
            self.AuxStatsClasses.append(DYCOMSStatistics(Gr, NS, Pa))

        return


    def stats_io(self, Grid.Grid Gr,  PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 MomentumAdvection.MomentumAdvection MA, MomentumDiffusion.MomentumDiffusion MD,  NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        #loop over class instances and class stats_io
        for aux_class in self.AuxStatsClasses:
            aux_class.stats_io(Gr, PV, DV, MA, MD, NS, Pa)

        return

class CumulusStatistics:
    def __init__(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        conditions = ['cloud','core']
        scalars = ['qt','ql','s', 'thetas']
        print('PV.name_index', PV.name_index)

        if 'qr' in PV.name_index:
            scalars.append('qr')
        if 'nr' in PV.name_index:
            scalars.append('nr')
        if 'theta_rho' in DV.name_index:
            scalars.append('theta_rho')
        if 'thetali' in DV.name_index:
            scalars.append('thetali')
        print('Aux scalars', scalars)

        for cond in conditions:
            NS.add_profile('fraction_'+cond,Gr,Pa)
            NS.add_profile('w_'+cond,Gr,Pa)
            NS.add_profile('w2_'+cond,Gr,Pa)
            for scalar in scalars:
                NS.add_profile(scalar+'_'+cond,Gr,Pa)
                NS.add_profile(scalar+'2_'+cond,Gr,Pa)

    def stats_io(self, Grid.Grid Gr,  PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 MomentumAdvection.MomentumAdvection MA, MomentumDiffusion.MomentumDiffusion MD,  NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            double [:] cloudmask = np.zeros(Gr.dims.npg,dtype=np.double, order='c')
            double [:] coremask = np.zeros(Gr.dims.npg,dtype=np.double, order='c')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            Py_ssize_t b_shift = DV.get_varshift(Gr,'buoyancy')
            Py_ssize_t i,j,k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw
            Py_ssize_t count
            double [:] mean_buoyancy

        mean_buoyancy = Pa.HorizontalMean(Gr, &DV.values[b_shift])

        with nogil:
            count = 0
            for i in range(imin,imax):
                ishift = i * istride
                for j in range(jmin,jmax):
                    jshift = j * jstride
                    for k in range(kmin,kmax):
                        ijk = ishift + jshift + k
                        if DV.values[ql_shift+ijk] > 0.0:
                            cloudmask[ijk] = 1.0
                            if DV.values[b_shift+ijk] > mean_buoyancy[k]:
                                coremask[ijk] = 1.0

        cdef double [:] tmp

        #Compute the statistics
        #-fractions        # cdef Py_ssize_t ths_shift = DV.get_varshift(Gr,'thetas')
        tmp = Pa.HorizontalMean(Gr, &cloudmask[0])
        NS.write_profile('fraction_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &coremask[0])
        NS.write_profile('fraction_core', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        #-w
        cdef Py_ssize_t shift = PV.get_varshift(Gr, 'w')
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[shift], &cloudmask[0])
        NS.write_profile('w_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &PV.values[shift], &PV.values[shift], &cloudmask[0])
        NS.write_profile('w2_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[shift], &coremask[0])
        NS.write_profile('w_core', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &PV.values[shift], &PV.values[shift], &coremask[0])
        NS.write_profile('w2_core', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        #-qt
        shift = PV.get_varshift(Gr, 'qt')
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[shift], &cloudmask[0])
        NS.write_profile('qt_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &PV.values[shift], &PV.values[shift], &cloudmask[0])
        NS.write_profile('qt2_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[shift], &coremask[0])
        NS.write_profile('qt_core', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &PV.values[shift], &PV.values[shift], &coremask[0])
        NS.write_profile('qt2_core', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        #-ql
        shift = DV.get_varshift(Gr, 'ql')
        tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[shift], &cloudmask[0])
        NS.write_profile('ql_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[shift], &DV.values[shift], &cloudmask[0])
        NS.write_profile('ql2_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[shift], &coremask[0])
        NS.write_profile('ql_core', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[shift], &DV.values[shift], &coremask[0])
        NS.write_profile('ql2_core', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        #-s
        shift = PV.get_varshift(Gr,'s')
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[shift],&cloudmask[0])
        NS.write_profile('s_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &PV.values[shift], &PV.values[shift], &cloudmask[0])
        NS.write_profile('s2_cloud', tmp[Gr.dims.gw:-Gr.dims.gw],Pa)
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[shift], &coremask[0])
        NS.write_profile('s_core', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &PV.values[shift], &PV.values[shift], &coremask[0])
        NS.write_profile('s2_core', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)


        if 'qr' in PV.name_index:
            shift = PV.get_varshift(Gr, 'qr')
            tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[shift], &cloudmask[0])
            NS.write_profile('qr_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[shift], &DV.values[shift], &cloudmask[0])
            NS.write_profile('qr2_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[shift], &coremask[0])
            NS.write_profile('qr_core', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[shift], &DV.values[shift], &coremask[0])
            NS.write_profile('qr2_core' ,tmp[Gr.dims.gw:-Gr.dims.gw],Pa)



        if 'nr' in PV.name_index:
            shift = PV.get_varshift(Gr, 'nr')
            tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[shift], &cloudmask[0])
            NS.write_profile('nr_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[shift], &DV.values[shift], &cloudmask[0])
            NS.write_profile('nr2_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[shift], &coremask[0])
            NS.write_profile('nr_core', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[shift], &DV.values[shift], &coremask[0])
            NS.write_profile('nr2_core' ,tmp[Gr.dims.gw:-Gr.dims.gw],Pa)


        if 'theta_rho' in DV.name_index:
            shift = DV.get_varshift(Gr, 'theta_rho')
            tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[shift], &cloudmask[0])
            NS.write_profile('theta_rho_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[shift], &DV.values[shift], &cloudmask[0])
            NS.write_profile('theta_rho2_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[shift], &coremask[0])
            NS.write_profile('theta_rho_core', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[shift], &DV.values[shift], &coremask[0])
            NS.write_profile('theta_rho2_core' ,tmp[Gr.dims.gw:-Gr.dims.gw],Pa)


        if 'thetali' in DV.name_index:
            shift = DV.get_varshift(Gr,'thetali')
            tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[shift], &cloudmask[0])
            NS.write_profile('thetali_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[shift], &DV.values[shift], &cloudmask[0])
            NS.write_profile('thetali2_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[shift], &coremask[0])
            NS.write_profile('thetali_core', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[shift], &DV.values[shift], &coremask[0])
            NS.write_profile('thetali2_core', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        cdef:
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            double[:] data = np.empty((Gr.dims.npg,), dtype=np.double, order='c')


        if 'thetas' in DV.name_index:
            shift = DV.get_varshift(Gr,'thetas')
            tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[shift], &cloudmask[0])
            NS.write_profile('thetas_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[shift], &DV.values[shift], &cloudmask[0])
            NS.write_profile('thetas2_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[shift], &coremask[0])
            NS.write_profile('thetas_core', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[shift], &DV.values[shift], &coremask[0])
            NS.write_profile('thetas2_core', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        else:
            with nogil:
                count = 0
                for i in range(imin, imax):
                    ishift = i * istride
                    for j in range(jmin, jmax):
                        jshift = j * jstride
                        for k in range(kmin, kmax):
                            ijk = ishift + jshift + k
                            data[count] = thetas_c(PV.values[s_shift + ijk], PV.values[qt_shift + ijk])

                            count += 1
            tmp = Pa.HorizontalMeanConditional(Gr, &data[0], &cloudmask[0])
            NS.write_profile('thetas_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &data[0], &data[0], &cloudmask[0])
            NS.write_profile('thetas2_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            tmp = Pa.HorizontalMeanConditional(Gr, &data[0], &coremask[0])
            NS.write_profile('thetas_core', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &data[0], &data[0], &coremask[0])
            NS.write_profile('thetas2_core', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        return


class StableBLStatistics:
    def __init__(self,Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
            #NS.add_ts('boundary_layer_height', Gr, Pa)
        return


    def stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, MomentumAdvection.MomentumAdvection MA,
                 MomentumDiffusion.MomentumDiffusion MD, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        # cdef:
        #     double [:] total_flux = np.zeros(Gr.dims.npg,dtype=np.double,order='c')
        #     Py_ssize_t d=2, i1, i,j,k, shift_flux
        #     double [:] flux_profile
        #
        # with nogil:
        #     for i1 in xrange(Gr.dims.dims-1):
        #         shift_flux = (i1*Gr.dims.dims + d) * Gr.dims.npg
        #         for i in xrange(Gr.dims.npg):
        #             total_flux[i] += (MA.flux[shift_flux + i] + MD.flux[shift_flux + i] ) *  (MA.flux[shift_flux + i] + MD.flux[shift_flux + i] )
        #
        #     for i in xrange(Gr.dims.npg):
        #         total_flux[i] = sqrt(total_flux[i])
        #
        # flux_profile = Pa.HorizontalMean(Gr,&total_flux[0])
        #
        # cdef:
        #     Py_ssize_t ustar_shift = DV.get_varshift_2d(Gr, 'friction_velocity')
        #     double flux_surface
        #     double [:] ustar2 = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        #
        # with nogil:
        #     for i in xrange(Gr.dims.nlg[0]*Gr.dims.nlg[1]):
        #         ustar2[i] = DV.values_2d[ustar_shift + i] * DV.values_2d[ustar_shift + i]
        #
        # flux_surface = Pa.HorizontalMeanSurface(Gr,&ustar2[0])
        #
        # k=Gr.dims.gw
        #
        # while k < Gr.dims.nlg[2]-Gr.dims.gw and flux_profile[k] > 0.05 * flux_surface:
        #     k += 1
        #
        # h05 = Gr.zl_half[k]
        # h0 = h05/0.95
        #
        # if np.isnan(h0):
        #     print('bl height is nan')
        #     h0 = 0.0
        #
        # NS.write_ts('boundary_layer_height', h0, Pa)

        return


class SmokeStatistics:
    def __init__(self,Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        NS.add_ts('boundary_layer_height', Gr, Pa)
        return


    def stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,
                 MomentumAdvection.MomentumAdvection MA, MomentumDiffusion.MomentumDiffusion MD, NetCDFIO_Stats NS,
                 ParallelMPI.ParallelMPI Pa):

        #Here we compute the boundary layer height consistent with Bretherton et al. 1999
        cdef:
            Py_ssize_t i, j, k, ij, ij2d, ijk
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t level_1
            Py_ssize_t level_2
            Py_ssize_t smoke_shift = PV.get_varshift(Gr, 'smoke')
            double [:] blh = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
            double blh_mean
            double smoke_1
            double smoke_2
            double z1
            double z2
            double dz

        with nogil:
            for i in xrange(Gr.dims.nlg[0]):
                for j in xrange(Gr.dims.nlg[1]):
                    ij = i * istride + j * jstride
                    ij2d = i * Gr.dims.nlg[1] + j
                    level_1 = 0
                    level_2 = 0
                    for k in xrange(Gr.dims.nlg[2]):
                        ijk = ij + k
                        if PV.values[smoke_shift + ijk] > 0.5:
                            level_1 = k
                    level_2 = level_1 + 1
                    smoke_1 = PV.values[smoke_shift + ij + level_1]
                    smoke_2 = PV.values[smoke_shift + ij + level_2]
                    z1 = Gr.zl_half[level_1]
                    z2 = Gr.zl_half[level_2]
                    dz = (0.5 - smoke_1)/(smoke_2 - smoke_1)*(z2 - z1)

                    blh[ij2d] = z1 + dz

        blh_mean = Pa.HorizontalMeanSurface(Gr, &blh[0])

        NS.write_ts('boundary_layer_height', blh_mean, Pa)

        return

class DYCOMSStatistics:

    def __init__(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        NS.add_ts('boundary_layer_height', Gr, Pa)
        return


    def stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
             DiagnosticVariables.DiagnosticVariables DV,
             MomentumAdvection.MomentumAdvection MA, MomentumDiffusion.MomentumDiffusion MD, NetCDFIO_Stats NS,
             ParallelMPI.ParallelMPI Pa):

        #Here we compute the boundary layer height consistent with Bretherton et al. 1999
        cdef:
            Py_ssize_t i, j, k, ij, ij2d, ijk
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t level_1
            Py_ssize_t level_2
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            double [:] blh = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
            double blh_mean
            double qt_1
            double qt_2
            double z1
            double z2
            double dz

        with nogil:
            for i in xrange(Gr.dims.nlg[0]):
                for j in xrange(Gr.dims.nlg[1]):
                    ij = i * istride + j * jstride
                    ij2d = i * Gr.dims.nlg[1] + j
                    level_1 = 0
                    level_2 = 0
                    for k in xrange(Gr.dims.nlg[2]):
                        ijk = ij + k
                        if PV.values[qt_shift+ ijk] >= 0.008:
                            level_1 = k
                    level_2 = level_1 + 1
                    qt_1 = PV.values[qt_shift + ij + level_1]
                    qt_2 = PV.values[qt_shift+ ij + level_2]
                    z1 = Gr.zl_half[level_1]
                    z2 = Gr.zl_half[level_2]
                    dz = (0.008 - qt_1)/(qt_2 - qt_1)*(z2 - z1)

                    blh[ij2d] = z1 + dz

        blh_mean = Pa.HorizontalMeanSurface(Gr, &blh[0])

        NS.write_ts('boundary_layer_height', blh_mean, Pa)

        return
