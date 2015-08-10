cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport Kinematics
from libc.math cimport  fmax

import cython
cdef class SGS:
    def __init__(self,namelist):
        if(namelist['sgs']['scheme'] == 'UniformViscosity'):
            self.scheme = UniformViscosity()
        elif(namelist['sgs']['scheme'] == 'Smagorinsky'):
            self.scheme = Smagorinsky(namelist)

    cpdef initialize(self, Grid.Grid Gr):
        self.scheme.initialize(Gr)
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV,Kinematics.Kinematics Ke):

        self.scheme.update(Gr,RS,DV,PV,Ke)

        return


cdef class UniformViscosity:
    def __init__(self):
        return

    cpdef initialize(self, Grid.Grid Gr):

        return

    @cython.boundscheck(False)  #Turn off numpy array index bounds checking
    @cython.wraparound(False)   #Turn off numpy array wrap around indexing
    @cython.cdivision(True)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke):

        cdef:
            long diff_shift = DV.get_varshift(Gr,'diffusivity')
            long visc_shift = DV.get_varshift(Gr,'viscosity')
            long i


        with nogil:
            for i in xrange(Gr.dims.npg):
                DV.values[diff_shift + i] = 75.0
                DV.values[visc_shift + i] = 75.0

        return

cdef class Smagorinsky:
    def __init__(self,namelist):
        try:
            self.cs = namelist['sgs']['cs']
        except:
            self.cs = 0.17
        try:
            self.prt = namelist['sgs']['prt']
        except:
            self.prt = 1.0/3.0


        return

    cpdef initialize(self, Grid.Grid Gr):

        return

    @cython.boundscheck(False)  #Turn off numpy array index bounds checking
    @cython.wraparound(False)   #Turn off numpy array wrap around indexing
    @cython.cdivision(True)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke):

        cdef:
            long diff_shift = DV.get_varshift(Gr,'diffusivity')
            long visc_shift = DV.get_varshift(Gr,'viscosity')
            long bf_shift =DV.get_varshift(Gr, 'buoyancy_frequency')
            long i

            double delta = (Gr.dims.dx[0]*Gr.dims.dx[1]*Gr.dims.dx[2])**(1.0/3.0)
            double cs = self.cs
            double prt = self.prt
            double fb


        with nogil:
            for i in xrange(Gr.dims.npg):
                DV.values[visc_shift + i] = cs*cs*delta*delta*Ke.strain_rate_mag[i]
                DV.values[diff_shift + i] = DV.values[visc_shift + i]/prt
                if DV.values[bf_shift+i] > 0.0:
                    fb = fmax(1.0 - DV.values[bf_shift + i]/(prt*Ke.strain_rate_mag[i]*Ke.strain_rate_mag[i]),0.0)
                    DV.values[diff_shift + i] = DV.values[diff_shift + i] * fb
                    DV.values[visc_shift + i] = DV.values[visc_shift + i] * fb

        print('performed Smagorinsky update')


        return
