cdef class TDMA:

    cdef:
        double* scratch
        long n
        void initialize(self, long n)
        inline void solve(self, double* x, double* a, double* b, double* c) nogil
        void destroy(self)