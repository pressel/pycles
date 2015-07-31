cimport Grid
cimport PrognosticVariables
cdef class Kinematics:
    cdef:
        double [:] vgrad
        double [:] strain_rate
        double [:] strain_rate_mag
        int get_grad_shift(self, Grid.Grid Gr, int vel_i, int dx_j)

    cpdef initialize(self, Grid.Grid Gr)

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)
