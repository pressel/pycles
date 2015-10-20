#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
import cython

cdef class TDMA:

    def  __init__(self):
        pass

    cdef void initialize(self, Py_ssize_t n):

        self.n = n
        self.scratch = <double*> PyMem_Malloc(self.n * sizeof(double))

        return

    cdef inline void solve(self, double* x, double* a, double* b, double* c) nogil:

        cdef:
            Py_ssize_t i
            double m

        self.scratch[0] = c[0]/b[0]
        x[0] = x[0]/b[0]

        with nogil:
            for i in xrange(1,self.n):
                m = 1.0/(b[i] - a[i] * self.scratch[i-1])
                self.scratch[i] = c[i] * m
                x[i] = (x[i] - a[i] * x[i-1])*m


            for i in xrange(self.n-2,-1,-1):
                x[i] = x[i] - self.scratch[i] * x[i+1]

        return

    cdef void destroy(self):
        PyMem_Free(self.scratch)
        return

    def test(self):
        n = 4
        cdef double [:] a = np.array([0, -1, -1, -1], dtype=np.double)
        cdef double [:] b = np.array([4, 4, 4, 4], dtype=np.double)
        cdef double [:] c = np.array([-1, -1, -1, 0], dtype=np.double)
        cdef double [:] d = np.array([5, 5, 10, 23], dtype=np.double)

        self.initialize(4)
        self.solve(&d[0], &a[0], &b[0], &c[0])

        print(np.array(d))

        return