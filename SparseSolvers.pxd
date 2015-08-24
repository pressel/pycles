cdef class TDMA:

    cdef:
        double* scratch
        Py_ssize_t n
        void initialize(self, Py_ssize_t n)
        inline void solve(self, double* x, double* a, double* b, double* c) nogil
        void destroy(self)