#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport ReferenceState
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
        if 'TKE' in auxiliary_statistics:
            self.AuxStatsClasses.append(TKEStatistics(Gr, NS, Pa))
        if 'Flux' in auxiliary_statistics:
            self.AuxStatsClasses.append(FluxStatistics(Gr,PV, DV, NS, Pa))
        return


    def stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 MomentumAdvection.MomentumAdvection MA, MomentumDiffusion.MomentumDiffusion MD,  NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        #loop over class instances and class stats_io
        for aux_class in self.AuxStatsClasses:
            aux_class.stats_io(Gr, RS, PV, DV, MA, MD, NS, Pa)

        return

class CumulusStatistics:
    def __init__(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        conditions = ['cloud','core']
        scalars = ['qt','ql','s', 'thetas']


        if 'qr' in PV.name_index:
            scalars.append('qr')
        if 'nr' in PV.name_index:
            scalars.append('nr')
        if 'theta_rho' in DV.name_index:
            scalars.append('theta_rho')
        if 'thetali' in DV.name_index:
            scalars.append('thetali')


        for cond in conditions:
            NS.add_profile('fraction_'+cond,Gr,Pa)
            NS.add_profile('w_'+cond,Gr,Pa)
            NS.add_profile('w2_'+cond,Gr,Pa)
            for scalar in scalars:
                NS.add_profile(scalar+'_'+cond,Gr,Pa)
                NS.add_profile(scalar+'2_'+cond,Gr,Pa)

    def stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
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


    def stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
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


    def stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
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


    def stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
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
                        if PV.values[qt_shift+ ijk] >= 0.005:
                            level_1 = k
                    level_2 = level_1 + 1
                    qt_1 = PV.values[qt_shift + ij + level_1]
                    qt_2 = PV.values[qt_shift+ ij + level_2]
                    z1 = Gr.zl_half[level_1]
                    z2 = Gr.zl_half[level_2]
                    dz = (0.005 - qt_1)/(qt_2 - qt_1)*(z2 - z1)

                    blh[ij2d] = z1 + dz

        blh_mean = Pa.HorizontalMeanSurface(Gr, &blh[0])

        NS.write_ts('boundary_layer_height', blh_mean, Pa)

        return

class TKEStatistics:

    def __init__(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        NS.add_ts('tke_int_z', Gr, Pa)
        NS.add_ts('tke_nd_int_z', Gr, Pa)

        NS.add_profile('tke_mean', Gr, Pa)
        NS.add_profile('tke_nd_mean', Gr, Pa)
        NS.add_profile('tke_prod_B', Gr, Pa)
        NS.add_profile('tke_prod_S', Gr, Pa)
        NS.add_profile('tke_prod_P', Gr, Pa)
        NS.add_profile('tke_prod_T', Gr, Pa)
        NS.add_profile('tke_prod_A', Gr, Pa)
        NS.add_profile('tke_prod_D', Gr, Pa)

        return

    def stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
             DiagnosticVariables.DiagnosticVariables DV,
             MomentumAdvection.MomentumAdvection MA, MomentumDiffusion.MomentumDiffusion MD, NetCDFIO_Stats NS,
             ParallelMPI.ParallelMPI Pa):

        #Here we compute the boundary layer height consistent with Bretherton et al. 1999
        cdef:
            Py_ssize_t i, j, k, ij, ij2d, ijk
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t ishift
            Py_ssize_t jshift
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t w_shift = PV.get_varshift(Gr, 'w')
            Py_ssize_t b_shift = DV.get_varshift(Gr,'buoyancy')
            Py_ssize_t p_shift = DV.get_varshift(Gr, 'dynamic_pressure')
            Py_ssize_t visc_shift = DV.get_varshift(Gr, 'viscosity')

            double [:] uc = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] vc = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] wc = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')

            double [:] up = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] vp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] wp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')

            double [:] ucp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] vcp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] wcp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')

            double [:] upup = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] upvp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] upwp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')

            double [:] vpvp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] vpwp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')

            double [:] wpwp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')

            double [:] uppp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] vppp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] wppp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')

            double [:] wpep = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] wpbp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')

            double [:] tke = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] tke_nd = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')

            #double [:] epup = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            #double [:] epvp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] epwp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')

            double [:] e_adv = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] e_dis = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')

            double [:] tke_S = np.zeros(Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] tke_P = np.zeros(Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] tke_T = np.zeros(Gr.dims.nlg[2], dtype=np.double, order='c')

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

        #Compute the horizontal means of the cell centered velocities
        cdef:
            double [:] ucmean = Pa.HorizontalMean(Gr, &uc[0])
            double [:] vcmean = Pa.HorizontalMean(Gr, &vc[0])
            double [:] wcmean = Pa.HorizontalMean(Gr, &wc[0])
            double [:] bmean = Pa.HorizontalMean(Gr, &DV.values[b_shift])
            double [:] pmean = Pa.HorizontalMean(Gr, &DV.values[p_shift])
            double  bp, pp

        #Compute the TKE
        with nogil:
            for i in xrange(1, Gr.dims.nlg[0]):
                ishift = i * istride
                for j in xrange(1, Gr.dims.nlg[1]):
                    jshift = j * jstride
                    for k in xrange(1, Gr.dims.nlg[2]):
                        ijk = ishift + jshift + k

                        #Compute fluctuations
                        up[ijk] = uc[ijk] - ucmean[k]
                        vp[ijk] = vc[ijk] - vcmean[k]
                        wp[ijk] = wc[ijk] - wcmean[k]
                        bp  = DV.values[b_shift + ijk] - bmean[k]
                        pp  = DV.values[p_shift + ijk] - pmean[k]

                        #Coumpute fluctuation products
                        upup[ijk] = up[ijk] * up[ijk]
                        upvp[ijk] = up[ijk] * vp[ijk]
                        upwp[ijk] = up[ijk] * wp[ijk]

                        vpvp[ijk] = vp[ijk] * vp[ijk]
                        vpwp[ijk] = vp[ijk] * wp[ijk]
                        wpwp[ijk] = wp[ijk] * wp[ijk]

                        uppp[ijk] = up[ijk] * pp
                        vppp[ijk] = vp[ijk] * pp
                        wppp[ijk] = wp[ijk] * pp

                        tke_nd[ijk] =  0.5 * (upup[ijk] + vpvp[ijk] + wpwp[ijk])
                        tke[ijk] = RS.rho0[k] * tke_nd[ijk]

                        wpbp[ijk] = wp[ijk] * bp

        cdef:
            double [:] upup_mean = Pa.HorizontalMean(Gr, &upup[0])
            double [:] upvp_mean = Pa.HorizontalMean(Gr, &upvp[0])
            double [:] upwp_mean = Pa.HorizontalMean(Gr, &upwp[0])
            double [:] vpvp_mean = Pa.HorizontalMean(Gr, &vpvp[0])
            double [:] vpwp_mean = Pa.HorizontalMean(Gr, &vpwp[0])
            double [:] wpwp_mean = Pa.HorizontalMean(Gr, &wpwp[0])
            double [:] wppp_mean = Pa.HorizontalMean(Gr, &wppp[0])
            double [:] tke_mean = Pa.HorizontalMean(Gr, &tke_nd[0])
            double [:] tkemean = Pa.HorizontalMean(Gr, &tke[0])
            double [:] tkendmean = Pa.HorizontalMean(Gr, &tke_nd[0])
            double [:] tke_B = Pa.HorizontalMean(Gr, &wpbp[0])

        #Compute the Shear Production
        with nogil:
            for k in xrange(1, Gr.dims.nlg[2]-1):
                tke_S[k] -= upwp_mean[k] * (ucmean[k+1] - ucmean[k-1]) * 0.5 * Gr.dims.dxi[2]
                tke_S[k] -= vpwp_mean[k] * (vcmean[k+1] - vcmean[k-1]) * 0.5 * Gr.dims.dxi[2]

        #Compute Pressure Work
        with nogil:
            for k in xrange(1, Gr.dims.nlg[2]-1):
                tke_P[k] -= (wppp_mean[k+1] * RS.alpha0[k+1] - wppp_mean[k-1]* RS.alpha0[k-1])* 0.5 * Gr.dims.dxi[2]

        #Compute the Turbulent transport
        with nogil:
            for i in xrange(1, Gr.dims.nlg[0]):
                ishift = i * istride
                for j in xrange(1, Gr.dims.nlg[1]):
                    jshift = j * jstride
                    for k in xrange(1, Gr.dims.nlg[2]):
                        ijk = ishift + jshift + k
                        epwp[ijk] = (wc[ijk] - wcmean[k])*(tke_nd[ijk] - tke_mean[k])

        cdef:
            double [:] epwp_mean = Pa.HorizontalMean(Gr, &epwp[0])

        #Compute Turbulent Transport
        with nogil:
            for k in xrange(1, Gr.dims.nlg[2] -1):
                tke_T[k] -= (epwp_mean[k+1] - epwp_mean[k-1]) * 0.5 * Gr.dims.dxi[2]

        #Compute Mean Advection
        with nogil:
            for i in xrange(1, Gr.dims.nlg[0]):
                ishift = i * istride
                for j in xrange(1, Gr.dims.nlg[1]):
                    jshift = j * jstride
                    for k in xrange(1, Gr.dims.nlg[2]):
                        ijk = ishift + jshift + k
                        e_adv[ijk] -= ucmean[k] * (tke_nd[ijk+istride] - tke_nd[ijk-istride])*0.5*Gr.dims.dxi[0]
                        e_adv[ijk] -= vcmean[k] * (tke_nd[ijk+jstride] - tke_nd[ijk-jstride])*0.5*Gr.dims.dxi[1]

        cdef:
            double [:] tke_A = Pa.HorizontalMean(Gr, &e_adv[0])
            double nu

        #Compute the dissipation of TKE
        with nogil:
            for i in xrange(1, Gr.dims.nlg[0]):
                ishift = i * istride
                for j in xrange(1, Gr.dims.nlg[1]):
                    jshift = j * jstride
                    for k in xrange(1, Gr.dims.nlg[2]):
                        ijk = ishift + jshift + k
                        nu = DV.values[visc_shift + ijk]
                        e_dis[ijk] += (up[ijk + istride] - up[ijk-istride]) * 0.5 * Gr.dims.dxi[0] * (up[ijk + istride] - up[ijk-istride]) * 0.5 * Gr.dims.dxi[0]
                        e_dis[ijk] += (vp[ijk + jstride] - vp[ijk-jstride]) * 0.5 * Gr.dims.dxi[1] * (vp[ijk + jstride] - vp[ijk-jstride]) * 0.5 * Gr.dims.dxi[1]
                        e_dis[ijk] += (wp[ijk + 1] - wp[ijk-1]) * 0.5 * Gr.dims.dxi[2] * (wp[ijk + 1] - wp[ijk-1]) * 0.5 * Gr.dims.dxi[2]
                        e_dis[ijk] *= nu

        cdef:
            double [:] tke_D = Pa.HorizontalMean(Gr, &e_dis[0])

        #Write data
        NS.write_profile('tke_mean', tkemean[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('tke_nd_mean', tkendmean[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_ts('tke_int_z', np.sum(tkemean[Gr.dims.gw:-Gr.dims.gw])*Gr.dims.dx[2], Pa)
        NS.write_ts('tke_nd_int_z', np.sum(tkendmean[Gr.dims.gw:-Gr.dims.gw])*Gr.dims.dx[2],Pa)
        NS.write_profile('tke_prod_B', tke_B[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('tke_prod_S', tke_S[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('tke_prod_P', tke_P[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('tke_prod_T', tke_T[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('tke_prod_A', tke_A[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('tke_prod_D', tke_D[Gr.dims.gw:-Gr.dims.gw], Pa)

        return


class FluxStatistics:
    def __init__(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        scalar_list = ['qt']

        if 'theta' in DV.name_index:
            scalar_list.append('theta')
        else:
            scalar_list.append('thetali')
        if 'buoyancy' in DV.name_index:
            scalar_list.append('buoyancy')

        for name in scalar_list:
            NS.add_profile('resolved_x_flux_'+name, Gr, Pa)
            NS.add_profile('resolved_y_flux_'+name, Gr, Pa)
            NS.add_profile('resolved_z_flux_'+name, Gr, Pa)

            NS.add_profile('sgs_x_flux_'+name, Gr, Pa)
            NS.add_profile('sgs_y_flux_'+name, Gr, Pa)
            NS.add_profile('sgs_z_flux_'+name, Gr, Pa)

        NS.add_profile('resolved_x_vel_flux', Gr, Pa)
        NS.add_profile('resolved_y_vel_flux', Gr, Pa)

        return

    def stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
             DiagnosticVariables.DiagnosticVariables DV,
             MomentumAdvection.MomentumAdvection MA, MomentumDiffusion.MomentumDiffusion MD, NetCDFIO_Stats NS,
             ParallelMPI.ParallelMPI Pa):

        #Here we compute the boundary layer height consistent with Bretherton et al. 1999
        cdef:
            Py_ssize_t i, j, k, ij, ij2d, ijk
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t ishift
            Py_ssize_t jshift
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t w_shift = PV.get_varshift(Gr, 'w')
            Py_ssize_t b_shift = DV.get_varshift(Gr, 'buoyancy')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t th_shift
            Py_ssize_t diff_shift = DV.get_varshift(Gr, 'diffusivity')
            double bp, thp, qtp


            double [:] uc = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] vc = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] wc = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')

            double [:] up = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] vp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] wp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')

            double [:] upwp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] vpwp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')

            double [:] bpup = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] bpvp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] bpwp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')


            double [:] thpup = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] thpvp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] thpwp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')

            double [:] qtpup = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] qtpvp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] qtpwp = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')


            double [:] b_xsgs = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] b_ysgs = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] b_zsgs = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')


            double [:] th_xsgs = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] th_ysgs = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] th_zsgs = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')

            double [:] qt_xsgs = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] qt_ysgs = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')
            double [:] qt_zsgs = np.zeros(Gr.dims.nlg[0]* Gr.dims.nlg[1]* Gr.dims.nlg[2], dtype=np.double, order='c')




        if 'theta' in DV.name_index:
            th_shift = DV.get_varshift(Gr,'theta')
        else:
            th_shift = DV.get_varshift(Gr,'thetali')


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


        #Compute the horizontal means of the cell centered velocities
        cdef:
            double [:] ucmean = Pa.HorizontalMean(Gr, &uc[0])
            double [:] vcmean = Pa.HorizontalMean(Gr, &vc[0])
            double [:] wcmean = Pa.HorizontalMean(Gr, &wc[0])
            double [:] bmean = Pa.HorizontalMean(Gr, &DV.values[b_shift])
            double [:] thmean = Pa.HorizontalMean(Gr, &DV.values[th_shift])
            double [:] qtmean = Pa.HorizontalMean(Gr, &PV.values[qt_shift])

        #Compute the fluxes
        with nogil:
            for i in xrange(1, Gr.dims.nlg[0]-1):
                ishift = i * istride
                for j in xrange(1, Gr.dims.nlg[1]-1):
                    jshift = j * jstride
                    for k in xrange(1, Gr.dims.nlg[2]-1):
                        ijk = ishift + jshift + k

                        #Compute fluctuations
                        up[ijk] = uc[ijk] - ucmean[k]
                        vp[ijk] = vc[ijk] - vcmean[k]
                        wp[ijk] = wc[ijk] - wcmean[k]
                        bp  = DV.values[b_shift + ijk] - bmean[k]
                        thp = DV.values[th_shift + ijk] - thmean[k]
                        qtp = PV.values[qt_shift + ijk] - qtmean[k]

                        upwp[ijk] = up[ijk] * wp[ijk]
                        vpwp[ijk] = vp[ijk] * wp[ijk]

                        bpup[ijk] = bp * up[ijk]
                        bpvp[ijk] = bp * vp[ijk]
                        bpwp[ijk] = bp * wp[ijk]

                        thpup[ijk] = thp * up[ijk]
                        thpvp[ijk] = thp * vp[ijk]
                        thpwp[ijk] = thp * wp[ijk]


                        qtpup[ijk] = qtp * up[ijk]
                        qtpvp[ijk] = qtp * vp[ijk]
                        qtpwp[ijk] = qtp * wp[ijk]


                        b_xsgs[ijk] = -DV.values[diff_shift+ijk] * (DV.values[b_shift + ijk + istride] - DV.values[b_shift + ijk -istride]) * Gr.dims.dxi[0] * 0.5
                        b_ysgs[ijk] = -DV.values[diff_shift+ijk] * (DV.values[b_shift + ijk + jstride] - DV.values[b_shift + ijk -jstride]) * Gr.dims.dxi[1] * 0.5
                        b_zsgs[ijk] = -DV.values[diff_shift+ijk] * (DV.values[b_shift + ijk + 1] - DV.values[b_shift + ijk -1]) * Gr.dims.dxi[2] * 0.5

                        th_xsgs[ijk] = -DV.values[diff_shift+ijk] * (DV.values[th_shift + ijk + istride] - DV.values[th_shift + ijk -istride]) * Gr.dims.dxi[0] * 0.5
                        th_ysgs[ijk] = -DV.values[diff_shift+ijk] * (DV.values[th_shift + ijk + jstride] - DV.values[th_shift + ijk -jstride]) * Gr.dims.dxi[1] * 0.5
                        th_zsgs[ijk] = -DV.values[diff_shift+ijk] * (DV.values[th_shift + ijk + 1] - DV.values[th_shift + ijk -1]) * Gr.dims.dxi[2] * 0.5

                        qt_xsgs[ijk] = -DV.values[diff_shift+ijk] * (PV.values[qt_shift + ijk + istride] - PV.values[qt_shift + ijk -istride]) * Gr.dims.dxi[0] * 0.5
                        qt_ysgs[ijk] = -DV.values[diff_shift+ijk] * (PV.values[qt_shift + ijk + jstride] - PV.values[qt_shift + ijk -jstride]) * Gr.dims.dxi[1] * 0.5
                        qt_zsgs[ijk] = -DV.values[diff_shift+ijk] * (PV.values[qt_shift + ijk + 1] - PV.values[qt_shift + ijk -1]) * Gr.dims.dxi[2] * 0.5


        cdef:
            double [:] thpup_mean = Pa.HorizontalMean(Gr, &thpup[0])
            double [:] thpvp_mean = Pa.HorizontalMean(Gr, &thpvp[0])
            double [:] thpwp_mean = Pa.HorizontalMean(Gr, &thpwp[0])


            double [:] qtpup_mean = Pa.HorizontalMean(Gr, &qtpup[0])
            double [:] qtpvp_mean = Pa.HorizontalMean(Gr, &qtpvp[0])
            double [:] qtpwp_mean = Pa.HorizontalMean(Gr, &qtpwp[0])

            double [:] bpup_mean = Pa.HorizontalMean(Gr, &bpup[0])
            double [:] bpvp_mean = Pa.HorizontalMean(Gr, &bpvp[0])
            double [:] bpwp_mean = Pa.HorizontalMean(Gr, &bpwp[0])

            double [:] upwp_mean = Pa.HorizontalMean(Gr, &upwp[0])
            double [:] vpwp_mean = Pa.HorizontalMean(Gr, &vpwp[0])

            double [:] th_xsgs_mean = Pa.HorizontalMean(Gr, &th_xsgs[0])
            double [:] th_ysgs_mean = Pa.HorizontalMean(Gr, &th_ysgs[0])
            double [:] th_zsgs_mean = Pa.HorizontalMean(Gr, &th_zsgs[0])

            double [:] qt_xsgs_mean = Pa.HorizontalMean(Gr, &qt_xsgs[0])
            double [:] qt_ysgs_mean = Pa.HorizontalMean(Gr, &qt_ysgs[0])
            double [:] qt_zsgs_mean = Pa.HorizontalMean(Gr, &qt_zsgs[0])


            double [:] b_xsgs_mean = Pa.HorizontalMean(Gr, &b_xsgs[0])
            double [:] b_ysgs_mean = Pa.HorizontalMean(Gr, &b_ysgs[0])
            double [:] b_zsgs_mean = Pa.HorizontalMean(Gr, &b_zsgs[0])

        if 'theta' in DV.name_index:
            NS.write_profile('resolved_x_flux_theta', thpup_mean[Gr.dims.gw:-Gr.dims.gw], Pa)
            NS.write_profile('resolved_y_flux_theta', thpvp_mean[Gr.dims.gw:-Gr.dims.gw], Pa)
            NS.write_profile('resolved_z_flux_theta', thpwp_mean[Gr.dims.gw:-Gr.dims.gw], Pa)


            NS.write_profile('sgs_x_flux_theta', th_xsgs_mean[Gr.dims.gw:-Gr.dims.gw], Pa)
            NS.write_profile('sgs_y_flux_theta', th_ysgs_mean[Gr.dims.gw:-Gr.dims.gw], Pa)
            NS.write_profile('sgs_z_flux_theta', th_zsgs_mean[Gr.dims.gw:-Gr.dims.gw], Pa)

        else:
            NS.write_profile('resolved_x_flux_thetali', thpup_mean[Gr.dims.gw:-Gr.dims.gw], Pa)
            NS.write_profile('resolved_y_flux_thetali', thpvp_mean[Gr.dims.gw:-Gr.dims.gw], Pa)
            NS.write_profile('resolved_z_flux_thetali', thpwp_mean[Gr.dims.gw:-Gr.dims.gw], Pa)


            NS.write_profile('sgs_x_flux_thetali', th_xsgs_mean[Gr.dims.gw:-Gr.dims.gw], Pa)
            NS.write_profile('sgs_y_flux_thetali', th_ysgs_mean[Gr.dims.gw:-Gr.dims.gw], Pa)
            NS.write_profile('sgs_z_flux_thetali', th_zsgs_mean[Gr.dims.gw:-Gr.dims.gw], Pa)


        NS.write_profile('resolved_x_flux_buoyancy', bpup_mean[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('resolved_y_flux_buoyancy', bpvp_mean[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('resolved_z_flux_buoyancy', bpwp_mean[Gr.dims.gw:-Gr.dims.gw], Pa)

        NS.write_profile('sgs_x_flux_buoyancy', b_xsgs_mean[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('sgs_y_flux_buoyancy', b_ysgs_mean[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('sgs_z_flux_buoyancy', b_zsgs_mean[Gr.dims.gw:-Gr.dims.gw], Pa)

        NS.write_profile('resolved_x_vel_flux', upwp_mean[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('resolved_y_vel_flux', vpwp_mean[Gr.dims.gw:-Gr.dims.gw], Pa)



        NS.write_profile('resolved_x_flux_qt', qtpup_mean[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('resolved_y_flux_qt', qtpvp_mean[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('resolved_z_flux_qt', qtpwp_mean[Gr.dims.gw:-Gr.dims.gw], Pa)

        NS.write_profile('sgs_x_flux_qt', qt_xsgs_mean[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('sgs_y_flux_qt', qt_ysgs_mean[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('sgs_z_flux_qt', qt_zsgs_mean[Gr.dims.gw:-Gr.dims.gw], Pa)



        return