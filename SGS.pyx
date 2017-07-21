#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport PrognosticVariables
cimport DiagnosticVariables
cimport Kinematics
cimport ParallelMPI
cimport Surface
from NetCDFIO cimport NetCDFIO_Stats
from libc.math cimport exp, sqrt
cimport numpy as np
import numpy as np
import cython

cdef extern from "sgs.h":
    void smagorinsky_update(Grid.DimStruct* dims, double* visc, double* diff, double* buoy_freq,
                            double* strain_rate_mag, double cs, double prt)
    void tke_viscosity_diffusivity(Grid.DimStruct *dims, double* e, double* buoy_freq,double* visc, double* diff,
                                   double cn, double ck)
    void tke_dissipation(Grid.DimStruct* dims, double* e, double* e_tendency, double* buoy_freq, double cn, double ck)
    void tke_shear_production(Grid.DimStruct *dims,  double* e_tendency, double* visc, double* strain_rate_mag)
    void tke_buoyant_production(Grid.DimStruct *dims,  double* e_tendency, double* diff, double* buoy_freq)
    void tke_surface(Grid.DimStruct *dims, double* e, double* lmo, double* ustar, double h_bl, double zb) nogil
    double tke_ell(double cn, double e, double buoy_freq, double delta) nogil
    void smagorinsky_update_wall(Grid.DimStruct* dims, double* zl_half, double* visc, double* diff, double* buoy_freq,
                            double* strain_rate_mag, double cs, double prt)
    void smagorinsky_update_iles(Grid.DimStruct* dims, double* zl_half, double* visc, double* diff, double* buoy_freq,
                            double* strain_rate_mag, double cs, double prt)
cdef class SGS:
    def __init__(self,namelist):
        if(namelist['sgs']['scheme'] == 'UniformViscosity'):
            self.scheme = UniformViscosity(namelist)
        elif(namelist['sgs']['scheme'] == 'Smagorinsky'):
            self.scheme = Smagorinsky(namelist)
        elif(namelist['sgs']['scheme'] == 'TKE'):
            self.scheme = TKE(namelist)

        return

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.scheme.initialize(Gr,PV,NS,Pa)
        return

    cpdef update(self, Grid.Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV,Kinematics.Kinematics Ke,Surface.SurfaceBase Sur, ParallelMPI.ParallelMPI Pa):

        self.scheme.update(Gr,DV,PV,Ke,Sur,Pa)

        return

    cpdef stats_io(self, Grid.Grid Gr, DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.scheme.stats_io(Gr,DV,PV,Ke,NS,Pa)
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

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        return


    cpdef update(self, Grid.Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke,Surface.SurfaceBase Sur, ParallelMPI.ParallelMPI Pa):


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

    cpdef stats_io(self, Grid.Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

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

        try:
            self.adjust_wall = namelist['sgs']['Smagorinsky']['wall']
            if self.adjust_wall:
                self.iles = False
        except:
            self.adjust_wall = False

        try:
            self.iles = namelist['sgs']['Smagorinsky']['iles']
            if self.iles:
                self.adjust_wall = False
        except:
            self.iles = False


        return

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        return

    cpdef update(self, Grid.Grid Gr, DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke, Surface.SurfaceBase Sur,  ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t diff_shift = DV.get_varshift(Gr,'diffusivity')
            Py_ssize_t visc_shift = DV.get_varshift(Gr,'viscosity')
            Py_ssize_t bf_shift =DV.get_varshift(Gr, 'buoyancy_frequency')

        if self.adjust_wall:
            smagorinsky_update_wall(&Gr.dims, &Gr.zpl_half[0], &DV.values[visc_shift],&DV.values[diff_shift],&DV.values[bf_shift],
                                    &Ke.strain_rate_mag[0],self.cs,self.prt)

        elif self.iles:
            smagorinsky_update_iles(&Gr.dims, &Gr.zpl_half[0], &DV.values[visc_shift],&DV.values[diff_shift],&DV.values[bf_shift],
                                    &Ke.strain_rate_mag[0],self.cs,self.prt)
        else:
            smagorinsky_update(&Gr.dims,&DV.values[visc_shift],&DV.values[diff_shift],&DV.values[bf_shift],
                               &Ke.strain_rate_mag[0],self.cs,self.prt)


        return

    cpdef stats_io(self, Grid.Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
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
            self.cn = 0.76

        return

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        PV.add_variable('e', 'm^2/s^2', 'sym','scalar',Pa)


        self.Z_Pencil = ParallelMPI.Pencil()
        self.Z_Pencil.initialize(Gr,Pa,dim=2)

        NS.add_profile('tke_dissipation_tendency', Gr, Pa)
        NS.add_profile('tke_shear_tendency', Gr, Pa)
        NS.add_profile('tke_buoyancy_tendency', Gr, Pa)
        NS.add_profile('tke_prandtl_number', Gr, Pa)
        NS.add_profile('tke_mixing_length', Gr, Pa)

        return


    cpdef update(self, Grid.Grid Gr,  DiagnosticVariables.DiagnosticVariables DV, PrognosticVariables.PrognosticVariables PV,
                 Kinematics.Kinematics Ke, Surface.SurfaceBase Sur, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t diff_shift = DV.get_varshift(Gr,'diffusivity')
            Py_ssize_t visc_shift = DV.get_varshift(Gr,'viscosity')
            Py_ssize_t bf_shift = DV.get_varshift(Gr,'buoyancy_frequency')
            Py_ssize_t e_shift = PV.get_varshift(Gr,'e')
            Py_ssize_t th_shift
            Py_ssize_t i,k
            double [:,:] theta_pencil
            double h_local = 0.0
            double h_global = 0.0
            double n_xy_i = 1.0/(Gr.dims.nlg[0]*Gr.dims.nlg[1])


        if 'theta_rho' in DV.name_index:
            th_shift = DV.get_varshift(Gr,'theta_rho')
        else:
            th_shift = DV.get_varshift(Gr,'theta')

        theta_pencil = self.Z_Pencil.forward_double(&Gr.dims, Pa, &DV.values[th_shift])

        for i in xrange(self.Z_Pencil.n_local_pencils):
            k=Gr.dims.gw
            while theta_pencil[i,k] <= theta_pencil[i,Gr.dims.gw]:
                k = k + 1
            h_local = h_local + Gr.z_half[k]
        h_global = Pa.domain_scalar_sum(h_local)/(Gr.dims.n[0]*Gr.dims.n[1])




        tke_viscosity_diffusivity(&Gr.dims, &PV.values[e_shift], &DV.values[bf_shift], &DV.values[visc_shift],
                                  &DV.values[diff_shift], self.cn, self.ck)

        tke_dissipation(&Gr.dims, &PV.values[e_shift], &PV.tendencies[e_shift], &DV.values[bf_shift], self.cn, self.ck)

        tke_shear_production(&Gr.dims,  &PV.tendencies[e_shift], &DV.values[visc_shift], &Ke.strain_rate_mag[0])

        tke_buoyant_production(&Gr.dims, &PV.tendencies[e_shift], &DV.values[diff_shift], &DV.values[bf_shift])


        if Pa.sub_z_rank == 0:
            tke_surface(&Gr.dims, &PV.values[e_shift], &Sur.obukhov_length[0], &Sur.friction_velocity[0] , h_global, Gr.zl_half[Gr.dims.gw])


        return


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef stats_io(self, Grid.Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t diff_shift = DV.get_varshift(Gr,'diffusivity')
            Py_ssize_t visc_shift = DV.get_varshift(Gr,'viscosity')
            Py_ssize_t bf_shift = DV.get_varshift(Gr,'buoyancy_frequency')
            Py_ssize_t e_shift = PV.get_varshift(Gr,'e')
            double [:] mean_tendency = np.empty((Gr.dims.nlg[2],),dtype=np.double,order='c')

            double [:] mean = np.empty((Gr.dims.nlg[2],),dtype=np.double,order='c')

        cdef double [:] tmp_tendency1  = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
        tke_dissipation(&Gr.dims, &PV.values[e_shift], &tmp_tendency1[0], &DV.values[bf_shift], self.cn, self.ck)
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency1[0])
        NS.write_profile('tke_dissipation_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        cdef double [:] tmp_tendency2  = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
        tke_shear_production(&Gr.dims,   &tmp_tendency2[0], &DV.values[visc_shift], &Ke.strain_rate_mag[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency2[0])
        NS.write_profile('tke_shear_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        cdef double [:] tmp_tendency3  = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
        tke_buoyant_production(&Gr.dims,  &tmp_tendency3[0], &DV.values[diff_shift], &DV.values[bf_shift])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency3[0])
        NS.write_profile('tke_buoyancy_tendency',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        cdef:
            Py_ssize_t i
            double delta = (Gr.dims.dx[0] * Gr.dims.dx[1] * Gr.dims.dx[2])**(1.0/3.0)
            double [:] prt  = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
            double [:] mixing_length = np.zeros((Gr.dims.npg),dtype=np.double,order='c')

        with nogil:
            for i in xrange(Gr.dims.npg):
                mixing_length[i] = tke_ell(self.cn, PV.values[e_shift+i], DV.values[bf_shift+i], delta)
                prt[i] = delta/(delta + 2.0*mixing_length[i])


        mean = Pa.HorizontalMean(Gr,&prt[0])
        NS.write_profile('tke_prandtl_number',mean[Gr.dims.gw:-Gr.dims.gw],Pa)

        mean = Pa.HorizontalMean(Gr,&mixing_length[0])
        NS.write_profile('tke_mixing_length',mean[Gr.dims.gw:-Gr.dims.gw],Pa)


        return
