#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport PrognosticVariables
cimport ParallelMPI
cimport ReferenceState
from NetCDFIO cimport NetCDFIO_Stats

import numpy as np
cimport numpy as np

cdef extern from "kinematics.h":
    void compute_velocity_gradient(Grid.DimStruct *dims, double *v, double *vgrad, long d)
    void compute_strain_rate(Grid.DimStruct *dims, double *vgrad, double *strain_rate)
    void compute_strain_rate_mag(Grid.DimStruct *dims, double *vgrad, double *strain_rate)
    void compute_wind_speed_angle(Grid.DimStruct *dims, double *u, double *v, double *wind_speed, double *wind_angle, double u0, double v0)

cdef class Kinematics:
    def __init__(self):
        pass

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.vgrad = np.zeros(
            Gr.dims.npg * Gr.dims.dims * Gr.dims.dims, dtype=np.double, order='c')
        self.strain_rate = np.zeros(
            Gr.dims.npg * Gr.dims.dims * Gr.dims.dims, dtype=np.double, order='c')
        self.strain_rate_mag = np.zeros(Gr.dims.npg, dtype=np.double, order='c')

        self.wind_speed = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        self.wind_angle = np.zeros(Gr.dims.npg, dtype=np.double, order='c')

        NS.add_profile('strain_rate_magnitude', Gr, Pa)
        NS.add_profile('wind_speed', Gr, Pa)
        NS.add_profile('wind_angle', Gr, Pa)
        return

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV):

        cdef:
            Py_ssize_t vi1, d, count = 0
            Py_ssize_t shift_v1
            Py_ssize_t shift_v_grad, shift_strain

        for vi1 in xrange(Gr.dims.dims):
            shift_v1 = PV.velocity_directions[vi1] * Gr.dims.npg
            for d in xrange(Gr.dims.dims):
                shift_v_grad = Gr.dims.npg * count
                compute_velocity_gradient( & Gr.dims, & PV.values[shift_v1], & self.vgrad[shift_v_grad], d)
                count += 1

        compute_strain_rate( &Gr.dims, &self.vgrad[0], &self.strain_rate[0])
        compute_strain_rate_mag( &Gr.dims, &self.strain_rate[0], &self.strain_rate_mag[0])

        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        cdef:
            double [:] mean = np.empty((Gr.dims.nlg[2],),dtype=np.double,order='c')
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')


        mean = Pa.HorizontalMean(Gr,&self.strain_rate_mag[0])
        NS.write_profile('strain_rate_magnitude',mean[Gr.dims.gw:-Gr.dims.gw],Pa)

        compute_wind_speed_angle(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &self.wind_speed[0], &self.wind_angle[0],RS.u0, RS.v0)

        mean = Pa.HorizontalMean(Gr,&self.wind_speed[0])
        NS.write_profile('wind_speed',mean[Gr.dims.gw:-Gr.dims.gw],Pa)

        mean = Pa.HorizontalMean(Gr,&self.wind_angle[0])
        NS.write_profile('wind_angle',mean[Gr.dims.gw:-Gr.dims.gw],Pa)

        return


    cdef Py_ssize_t get_grad_shift(self, Grid.Grid Gr, Py_ssize_t vel_i, Py_ssize_t dx_i):
        return 3 * Gr.dims.npg * vel_i + Gr.dims.npg * dx_i
