cimport Lookup
cimport ParallelMPI
cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
from NetCDFIO cimport NetCDFIO_Fields, NetCDFIO_Stats


cdef class LatentHeat:
    cdef:
        #In the functions pointed to by the function pointer L* must not require gil
        float (*L_fp)(float T, float Lambda) nogil
        float (*Lambda_fp)(float T) nogil

    cpdef L(self, float T, float Lambda)

    cpdef Lambda(self, float T)


cdef class ClausiusClapeyron:
    cdef:
        Lookup.Lookup LT

    #def initialize(self,namelist,LatentHeat LH, ParallelMPI.ParallelMPI Par)
    cpdef finalize(self)


