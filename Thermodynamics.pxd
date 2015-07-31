cimport Lookup
cimport ParallelMPI


cdef class LatentHeat:
    cdef:
        #In the functions pointed to by the function pointer L* must not require gil
        double (*L_fp)(double T, double Lambda) nogil
        double (*Lambda_fp)(double T) nogil

    cpdef L(self, double T, double Lambda)

    cpdef Lambda(self, double T)


cdef class ClausiusClapeyron:
    cdef:
        Lookup.Lookup LT

    #def initialize(self,namelist,LatentHeat LH, ParallelMPI.ParallelMPI Par)
    cpdef finalize(self)




