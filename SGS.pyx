#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport Kinematics
from libc.math cimport  fmax
import cython

cdef extern from "sgs.h":
    void smagorinsky_update(Grid.DimStruct* dims, double* visc, double* diff, double* buoy_freq,
                            double* strain_rate_mag, double cs, double prt)

cdef class SGS:
    def __init__(self,namelist):
        if(namelist['sgs']['scheme'] == 'UniformViscosity'):
            self.scheme = UniformViscosity(namelist)
        elif(namelist['sgs']['scheme'] == 'Smagorinsky'):
            self.scheme = Smagorinsky(namelist)
        return

    cpdef initialize(self, Grid.Grid Gr):
        self.scheme.initialize(Gr)
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV,Kinematics.Kinematics Ke):

        self.scheme.update(Gr,Ref,DV,PV,Ke)

        return

cdef class UniformViscosity:
    def __init__(self,namelist):
        try:
            self.const_diffusivity = namelist['sgs']['UniformViscosity']['diffusivity']
        except:
            self.const_diffusivity = 0.0

        try:
            self.const_viscosity = namelist['sgs']['UniformViscosity']['viscosity']
        except:
            self.const_viscosity = 0.0

        self.is_init = False 

        return

    cpdef initialize(self, Grid.Grid Gr):

        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke):

        cdef:
            Py_ssize_t diff_shift = DV.get_varshift(Gr,'diffusivity')
            Py_ssize_t visc_shift = DV.get_varshift(Gr,'viscosity')
            Py_ssize_t i


        with nogil:
            if not self.is_init: 
                for i in xrange(Gr.dims.npg):
                    DV.values[diff_shift + i] = self.const_diffusivity
                    DV.values[visc_shift + i] = self.const_viscosity
                    self.is_init = True 

        return

cdef class Smagorinsky:
    def __init__(self,namelist):
        try:
            self.cs = namelist['sgs']['Smagorinsky']['cs']
        except:
            self.cs = 0.17
        try:
            self.prt = namelist['sgs']['Smagorinsky']['prt']
        except:
            self.prt = 1.0/3.0

        return

    cpdef initialize(self, Grid.Grid Gr):

        return

    @cython.boundscheck(False)  #Turn off numpy array index bounds checking
    @cython.wraparound(False)   #Turn off numpy array wrap around indexing
    @cython.cdivision(True)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke):

        cdef:
            Py_ssize_t diff_shift = DV.get_varshift(Gr,'diffusivity')
            Py_ssize_t visc_shift = DV.get_varshift(Gr,'viscosity')
            Py_ssize_t bf_shift =DV.get_varshift(Gr, 'buoyancy_frequency')

        smagorinsky_update(&Gr.dims,&DV.values[visc_shift],&DV.values[diff_shift],&DV.values[bf_shift],&Ke.strain_rate_mag[0],self.cs,self.prt)

        return

