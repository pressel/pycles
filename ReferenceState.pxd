cimport Grid
cimport Restart
cdef class ReferenceState:
    cdef:
        public float [:] p0
        public float [:] p0_half
        public float [:] alpha0
        public float [:] alpha0_half
        public float [:] rho0
        public float [:] rho0_half

        public float [:] p0_global
        public float [:] p0_half_global
        public float [:] alpha0_global
        public float [:] alpha0_half_global
        public float [:] rho0_global
        public float [:] rho0_half_global

        float sg

    cdef public:
        #These public values should be set in the case initialization routine
        float Tg  #Temperature at ground level
        float Pg  #Pressure at ground level
        float qtg #Surface total water mixing ratio
        float u0 #u velocity removed in Galilean transformation
        float v0 #v velocity removed in Galilean transformation

    cpdef restart(self, Grid.Grid Gr, Restart.Restart Re)
    cpdef init_from_restart(self, Grid.Grid Gr, Restart.Restart Re)

