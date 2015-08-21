cimport Grid
cdef class ReferenceState:
    cdef:
        public double [:] p0
        public double [:] p0_half
        public double [:] alpha0
        public double [:] alpha0_half
        public double [:] rho0
        public double [:] rho0_half
        double sg

    cdef public:
        #These public values should be set in the case initialization routine
        double Tg  #Temperature at ground level
        double Pg  #Pressure at ground level
        double qtg #Surface total water mixing ratio




