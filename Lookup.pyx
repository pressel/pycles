#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

import numpy as np
cimport numpy as np


cdef class Lookup:
    def __init__(self):
        pass

    cpdef initialize(self, double[:] x, double[:] y):
        cdef long n = np.shape(y)[0]
        init_table( &self.LookupStructC, n, &x[0], &y[0])
        return

    cpdef finalize(self):
        free_table(&self.LookupStructC)
        return

    cpdef table_bounds(self):
        return self.LookupStructC.x_min, self.LookupStructC.x_max, self.LookupStructC.y_min, self.LookupStructC.y_max

    cpdef lookup(self, double x):
        return lookup(&self.LookupStructC, x)

    cdef inline double fast_lookup(self, double x) nogil:
        return lookup(&self.LookupStructC, x)
