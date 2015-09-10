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


def MicrophysicsFactory(namelist, LatentHeat LH, ParallelMPI.ParallelMPI Par):
    if(namelist['microphysics']['scheme'] == 'None_Dry'):
        return No_Microphysics_Dry(Par, LH, namelist)
    elif(namelist['microphysics']['scheme'] == 'None_SA'):
        return No_Microphysics_SA(Par, LH, namelist)
