#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport ParallelMPI
from Thermodynamics cimport LatentHeat

cdef class No_Microphysics_Dry:
    def __init__(self, ParallelMPI.ParallelMPI Par, LatentHeat LH, namelist):
        LH.Lambda_fp = lambda_constant
        LH.L_fp = latent_heat_constant
        self.thermodynamics_type = 'dry'
        return

cdef class No_Microphysics_SA:
    def __init__(self, ParallelMPI.ParallelMPI Par, LatentHeat LH, namelist):
        LH.Lambda_fp = lambda_constant
        LH.L_fp = latent_heat_constant
        self.thermodynamics_type = 'SA'
        return



cdef class Microphysics_SB_Liquid:
    def __init__(self,ParallelMPI.ParallelMPI Par, LatentHeat LH, namelist):
        # Create the appropriate linkages to the bulk thermodynamics
        LH.Lambda_fp = lambda_constant
        LH.L_fp = latent_heat_constant
        self.thermodynamics_type = 'SA'
        # Extract case-specific parameter values from the namelist
        try:
            self.ccn = namelist['microphysics']['SB_Liquid']['ccn']
        except:
            self.ccn = 100.0e6
        try:
            mu_opt = namelist['microphysics']['SB_Liquid']['mu']
            if mu_opt == 1:
                self.compute_shape_parameter = sb_shape_parameter_1
            elif mu_opt == 2:
                self.compute_shape_parameter = sb_shape_parameter_2
            elif mu_opt == 4:
                self.compute_shape_parameter = sb_shape_parameter_4
            elif mu_opt == 0:
                self.compute_shape_parameter  = sb_shape_parameter_0
            else:
                print("SB_Liquid mu value not recognized, defaulting to mu =1")
                self.compute_shape_parameter = sb_shape_parameter_1
        except:
            mu_opt = 1
            self.compute_shape_parameter = sb_shape_parameter_1
        try:
            self.nuc_opt = namelist['microphysics']['SB_Liquid']['nuc_opt']

        except:
            self.nuc_opt = 0
        return

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        # add prognostic variables for mass and number of rain
        PV.add_variable('qr', 'kg/kg', 'sym','scalar',Pa)
        PV.add_variable('nr', 'kg/kg', 'sym','scalar',Pa)


        # add statistical output for the class
        NS.add_profile('rain_autoconversion', Gr, Pa)
        return

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)






def MicrophysicsFactory(namelist, LatentHeat LH, ParallelMPI.ParallelMPI Par):
    if(namelist['microphysics']['scheme'] == 'None_Dry'):
        return No_Microphysics_Dry(Par, LH, namelist)
    elif(namelist['microphysics']['scheme'] == 'None_SA'):
        return No_Microphysics_SA(Par, LH, namelist)
