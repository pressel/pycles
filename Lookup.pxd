
cdef extern from "lookup.h":
    struct LookupStruct:
        int n_table
        double y_max
        double y_min
        double x_max
        double x_min
        double dx
        double dxi
        double* table_values

    void init_table(LookupStruct *LT, long n, double *x, double *y) nogil
    void free_table(LookupStruct *LT) nogil
    inline double lookup(LookupStruct * LT, double x) nogil

cdef class Lookup:
    cdef:
        LookupStruct LookupStructC
    cpdef initialize(self, double[:] x, double[:] y)
    cpdef finalize(self)
    cpdef table_bounds(self)
    cpdef lookup(self, double x)
    cdef:
        inline double fast_lookup(self, double x) nogil
