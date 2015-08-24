cimport Grid
cimport PrognosticVariables
cdef class Kinematics:
    cdef:
        double [:] vgrad
        double [:] strain_rate
        double [:] strain_rate_mag
        Py_ssize_t get_grad_shift(self, Grid.Grid Gr, Py_ssize_t vel_i, Py_ssize_t dx_j)
    cpdef initialize(self, Grid.Grid Gr)
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)
