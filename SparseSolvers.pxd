cdef class TDMA:

    cdef:
        float* scratch
        Py_ssize_t n
        void initialize(self, Py_ssize_t n)
        inline void solve(self, float* x, float* a, float* b, float* c) nogil
        void destroy(self)