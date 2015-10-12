#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport numpy as np
import numpy as np
cimport Grid
cimport Lookup
cimport PrognosticVariables
cimport DiagnosticVariables
cimport ReferenceState
cimport ParallelMPI
cimport TimeStepping
from NetCDFIO cimport NetCDFIO_Stats
from Thermodynamics cimport LatentHeat, ClausiusClapeyron
from libc.math cimport fmax, fmin


cdef class No_Microphysics_Dry:
    def __init__(self, ParallelMPI.ParallelMPI Par, LatentHeat LH, namelist):
        LH.Lambda_fp = lambda_constant
        LH.L_fp = latent_heat_constant
        self.thermodynamics_type = 'dry'
        return
    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS):
        return


cdef class No_Microphysics_SA:
    def __init__(self, ParallelMPI.ParallelMPI Par, LatentHeat LH, namelist):
        LH.Lambda_fp = lambda_constant
        LH.L_fp = latent_heat_constant
        self.thermodynamics_type = 'SA'
        return
    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS):
        return


cdef extern from "microphysics_sb.h":
    double sb_rain_shape_parameter_0(double density, double qr, double Dm) nogil
    double sb_rain_shape_parameter_1(double density, double qr, double Dm) nogil
    double sb_rain_shape_parameter_2(double density, double qr, double Dm) nogil
    double sb_rain_shape_parameter_4(double density, double qr, double Dm) nogil
    double sb_droplet_nu_0(double density, double ql) nogil
    double sb_droplet_nu_1(double density, double ql) nogil
    double sb_droplet_nu_2(double density, double ql) nogil
    void sb_sedimentation_velocity_rain(double (*rain_mu)(double,double,double),double density, double nr, double qr, double* nr_velocity, double* qr_velocity) nogil
    void sb_microphysics_sources(Lookup.LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                             double (*rain_mu)(double,double,double), double (*droplet_nu)(double,double),
                             double density, double p0, double temperature,  double qt, double ccn,
                             double ql, double nr, double qr, double dt, double* nr_tendency, double* qr_tendency) nogil





cdef class Microphysics_SB_Liquid:
    def __init__(self, ParallelMPI.ParallelMPI Par, LatentHeat LH, namelist):
        # Create the appropriate linkages to the bulk thermodynamics
        LH.Lambda_fp = lambda_constant
        LH.L_fp = latent_heat_constant
        self.thermodynamics_type = 'SA'
        #also set local versions
        self.Lambda_fp = lambda_constant
        self.L_fp = latent_heat_constant
        self.CC = ClausiusClapeyron()
        self.CC.initialize(namelist, LH, Par)


        # Extract case-specific parameter values from the namelist
        # Get number concentration of cloud condensation nuclei (1/m^3)
        try:
            self.ccn = namelist['microphysics']['SB_Liquid']['ccn']
        except:
            self.ccn = 100.0e6
        # Set option for calculation of mu (distribution shape parameter)
        try:
            mu_opt = namelist['microphysics']['SB_Liquid']['mu_rain']
            if mu_opt == 1:
                self.compute_rain_shape_parameter = sb_rain_shape_parameter_1
            elif mu_opt == 2:
                self.compute_rain_shape_parameter = sb_rain_shape_parameter_2
            elif mu_opt == 4:
                self.compute_rain_shape_parameter = sb_rain_shape_parameter_4
            elif mu_opt == 0:
                self.compute_rain_shape_parameter  = sb_rain_shape_parameter_0
            else:
                Par.root_print("SB_Liquid mu_rain option not recognized, defaulting to option 1")
                self.compute_rain_shape_parameter = sb_rain_shape_parameter_1
        except:
            self.compute_rain_shape_parameter = sb_rain_shape_parameter_1
        # Set option for calculation of nu parameter of droplet distribution
        try:
            nu_opt = namelist['microphysics']['SB_Liquid']['nu_droplet']
            if nu_opt == 0:
                self.compute_droplet_nu = sb_droplet_nu_0
            elif nu_opt == 1:
                self.compute_droplet_nu = sb_droplet_nu_1
            elif nu_opt ==2:
                self.compute_droplet_nu = sb_droplet_nu_2
            else:
                Par.root_print("SB_Liquid nu_droplet_option not recognized, defaulting to option 0")
                self.compute_droplet_nu = sb_droplet_nu_0
        except:
            self.compute_droplet_nu = sb_droplet_nu_0
        return

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        # add prognostic variables for mass and number of rain
        PV.add_variable('nr', '1/kg', 'sym','scalar',Pa)
        PV.add_variable('qr', 'kg/kg', 'sym','scalar',Pa)



        # add statistical output for the class
        NS.add_profile('qr_autoconversion', Gr, Pa)
        NS.add_profile('nr_autoconversion', Gr, Pa)
        NS.add_profile('nr_selfcollection', Gr, Pa)
        NS.add_profile('qr_accretion', Gr, Pa)
        NS.add_profile('nr_evaporation', Gr, Pa)
        NS.add_profile('qr_evaporation', Gr,Pa)
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS):
        cdef:
            Py_ssize_t i, j, k, ijk
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t kmax = Gr.dims.nlg[2]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t ishift, jshift

            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            Py_ssize_t nr_shift = PV.get_varshift(Gr, 'nr')
            Py_ssize_t qr_shift = PV.get_varshift(Gr, 'qr')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            double dt = TS.dt
            double [:] nr_velocity  = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
            double [:] qr_velocity  = np.zeros((Gr.dims.npg),dtype=np.double,order='c')



        with nogil:
            for i in xrange(gw, imax - gw):
                ishift = i * istride
                for j in xrange(gw, jmax - gw):
                    jshift = j * jstride
                    for k in xrange(gw, kmax-gw):
                        ijk = ishift + jshift + k
                        # PV.values[qr_shift + ijk] = fmax(PV.values[qr_shift + ijk], 0.0)
                        # PV.values[nr_shift + ijk] = fmax(PV.values[nr_shift + ijk], 0.0)


                        sb_microphysics_sources(&self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, self.compute_rain_shape_parameter,
                                                self.compute_droplet_nu, Ref.rho0_half[k],  Ref.p0_half[k], DV.values[t_shift + ijk],
                                                PV.values[qt_shift + ijk], self.ccn, DV.values[ql_shift + ijk], PV.values[nr_shift + ijk],
                                                PV.values[qr_shift + ijk], dt, &PV.tendencies[nr_shift + ijk], &PV.tendencies[qr_shift + ijk] )
            #
            # for i in xrange(imax):
            #     ishift = i * istride
            #     for j in xrange(jmax):
            #         jshift = j * jstride
            #         for k in xrange( kmax):
            #             ijk = ishift + jshift + k
            #
            #             sb_sedimentation_velocity_rain(self.compute_rain_shape_parameter,Ref.rho0_half[k],
            #                                            PV.values[nr_shift + ijk], PV.values[qr_shift + ijk], &nr_velocity[ijk], &qr_velocity[ijk])



        return


    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):



        return



def MicrophysicsFactory(namelist, LatentHeat LH, ParallelMPI.ParallelMPI Par):
    if(namelist['microphysics']['scheme'] == 'None_Dry'):
        return No_Microphysics_Dry(Par, LH, namelist)
    elif(namelist['microphysics']['scheme'] == 'None_SA'):
        return No_Microphysics_SA(Par, LH, namelist)
    elif(namelist['microphysics']['scheme'] == 'SB_Liquid'):
        return Microphysics_SB_Liquid(Par, LH, namelist)
