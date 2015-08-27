cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport Kinematics
cimport ParallelMPI

import cython

cdef extern from "sgs.h":
    void smagorinsky_update(Grid.DimStruct* dims, double* visc, double* diff, double* buoy_freq,
                            double* strain_rate_mag, double cs, double prt)
    void tke_viscosity_diffusivity(Grid.DimStruct *dims, double* e, double* buoy_freq,double* visc, double* diff,
                                   double cn, double ck)
    void tke_dissipation(Grid.DimStruct* dims, double* e, double* e_tendency, double* buoy_freq, double cn, double ck)

cdef class SGS:
    def __init__(self,namelist):
        if(namelist['sgs']['scheme'] == 'UniformViscosity'):
            self.scheme = UniformViscosity(namelist)
        elif(namelist['sgs']['scheme'] == 'Smagorinsky'):
            self.scheme = Smagorinsky(namelist)
        return

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI Pa):
        self.scheme.initialize(Gr,PV)
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

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI Pa):

        return

    @cython.boundscheck(False)  #Turn off numpy array index bounds checking
    @cython.wraparound(False)   #Turn off numpy array wrap around indexing
    @cython.cdivision(True)
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

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI Pa):

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

        smagorinsky_update(&Gr.dims,&DV.values[visc_shift],&DV.values[diff_shift],&DV.values[bf_shift],
                           &Ke.strain_rate_mag[0],self.cs,self.prt)

        return



cdef class TKE:
    def __init__(self,namelist):
        try:
            self.ck = namelist['sgs']['TKE']['ck']
        except:
            self.ck = 0.1
        try:
            self.cn = namelist['sgs']['TKE']['cn']
        except:
            self.prt = 0.76

        return

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI Pa):
        PV.add_variable('e', 'm^2/s^2', 'sym','scalar',Pa)

        return

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke):

        cdef:
            Py_ssize_t diff_shift = DV.get_varshift(Gr,'diffusivity')
            Py_ssize_t visc_shift = DV.get_varshift(Gr,'viscosity')
            Py_ssize_t bf_shift = DV.get_varshift(Gr,'buoyancy_frequency')
            Py_ssize_t e_shift = PV.get_varshift(Gr,'e')

        tke_viscosity_diffusivity(&Gr.dims, &PV.values[e_shift], &DV.values[bf_shift], &DV.values[visc_shift],
                                  &DV.values[diff_shift], self.cn, self.ck)
        tke_dissipation(&Gr.dims, &PV.values[e_shift], &PV.tendencies[e_shift], &DV.values[bf_shift], self.cn, self.ck)

        return
