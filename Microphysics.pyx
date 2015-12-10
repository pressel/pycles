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
        LH.L_fp = latent_heat_variable #latent_heat_constant
        self.thermodynamics_type = 'SA'
        return


cdef class No_Microphysics_DrySGS:
    def __init__(self, ParallelMPI.ParallelMPI Par, LatentHeat LH, namelist):
        LH.Lambda_fp = lambda_constant
        LH.L_fp = latent_heat_constant
        self.thermodynamics_type = 'dry_sgs'
        return

cdef class No_Microphysics_SA_SGS:
    def __init__(self, ParallelMPI.ParallelMPI Par, LatentHeat LH, namelist):
        LH.Lambda_fp = lambda_constant
        LH.L_fp = latent_heat_constant
        self.thermodynamics_type = 'SA_sgs'
        return


def MicrophysicsFactory(namelist, LatentHeat LH, ParallelMPI.ParallelMPI Par):
    try:
        sgs_flag = namelist['sgs']['sgs_condensation']
    except:
        sgs_flag = False


    if(namelist['microphysics']['scheme'] == 'None_Dry'):
        if sgs_flag:
            return No_Microphysics_DrySGS(Par, LH, namelist)
        else:
            return No_Microphysics_Dry(Par, LH, namelist)

    elif(namelist['microphysics']['scheme'] == 'None_SA'):
        if sgs_flag:
            return No_Microphysics_SA_SGS(Par, LH, namelist)
        else:
            return No_Microphysics_SA(Par, LH, namelist)

