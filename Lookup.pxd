
cdef extern from "lookup.h":
    struct LookupStruct:
        int n_table
        float y_max
        float y_min
        float x_max
        float x_min
        float dx
        float dxi
        float* table_values

    void init_table(LookupStruct *LT, long n, float *x, float *y) nogil
    void free_table(LookupStruct *LT) nogil
    inline float lookup(LookupStruct * LT, float x) nogil

cdef class Lookup:
    cdef:
        LookupStruct LookupStructC
    cpdef initialize(self, float[:] x, float[:] y)
    cpdef finalize(self)
    cpdef table_bounds(self)
    cpdef lookup(self, float x)
    cdef:
        inline float fast_lookup(self, float x) nogil
